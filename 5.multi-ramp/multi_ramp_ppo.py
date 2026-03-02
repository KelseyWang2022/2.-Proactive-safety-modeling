#!/usr/bin/env python3
"""
多Ramp PPO控制 - 完整版

基于单ramp代码改进:
1. 支持3个ramp协调控制
2. 清晰的奖励函数设计
3. 增强的观测空间
4. 支持baseline对比
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List
import json
from collections import deque

# ===========================
# SUMO 设置
# ===========================

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# =====================================================
# 多Ramp环境类
# =====================================================

class MultiRampEnv(gym.Env):
    """
    多Ramp Metering环境 (3个ramp)

    观测空间（26维）:
    [0-6]:   主线7段速度 / 30 (main_0 ~ main_6)
    [7-13]:  主线7段占有率 (main_0 ~ main_6)
    [14-17]: Ramp1状态 [排队/30, 等待/120, 上一步rate, 流入/1000]
    [18-21]: Ramp2状态 [排队/30, 等待/120, 上一步rate, 流入/1000]
    [22-25]: Ramp3状态 [排队/30, 等待/120, 上一步rate, 流入/1000]

    动作空间（3维）:
    - [rate1, rate2, rate3]: 每个ramp的放行率 [0,1]
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            sumo_cfg: str,
            max_steps: int = 120,  # 3600秒 / 30秒 = 120步
            delta_time: int = 30,  # 控制间隔30秒
            gui: bool = False,
            seed: int = 42,
            save_metrics: bool = False,
            metrics_dir: str = "./metrics"
    ):
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.gui = gui
        self.seed_value = seed
        self.save_metrics = save_metrics
        self.metrics_dir = metrics_dir

        if save_metrics:
            os.makedirs(metrics_dir, exist_ok=True)

        # ===== 网络配置 (3个ramp) =====
        self.MAINLINE_EDGES = ['main_0', 'main_1', 'main_2', 'main_3',
                               'main_4', 'main_5', 'main_6']
        self.RAMP_EDGES = [
            ['ramp1_0', 'ramp1_1'],  # Ramp1的两段
            ['ramp2_0', 'ramp2_1'],  # Ramp2的两段
            ['ramp3_0', 'ramp3_1']  # Ramp3的两段
        ]
        self.METER_IDS = ['meter1', 'meter2', 'meter3']

        # 关键路段
        self.MERGE_EDGES = ['main_1', 'main_3', 'main_5']  # 3个合流区
        self.BOTTLENECK_EDGES = ['main_2', 'main_4', 'main_6']  # 3个瓶颈区

        # 容量参数
        self.MAX_RAMP_QUEUE = 30
        self.MIN_RATE = 200.0  # veh/h
        self.MAX_RATE = 1200.0  # veh/h

        # ===== 观测和动作空间 =====
        # 观测: 26维
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(26,),
            dtype=np.float32
        )

        # 动作: 3维 (3个ramp的放行率)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # 状态变量
        self.current_step = 0
        self.episode_count = 0
        self.sumo_running = False
        self.last_rates = np.array([0.5, 0.5, 0.5])  # 上一步的放行率(归一化)

        # 历史信息
        self.ramp_inflows = [0.0, 0.0, 0.0]  # 各ramp的流入速率(veh/h)

        # 指标记录
        self.episode_metrics = {
            'reward': [],
            'mainline_speed': [],
            'ramp_queues': [[], [], []],  # 3个ramp的排队
            'throughput': [],
            'metering_rates': [[], [], []],  # 3个meter的放行率
            'total_delay': []
        }

    # -----------------------------
    # Gym 核心接口
    # -----------------------------

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

        sumo_cmd = [
            "sumo-gui" if self.gui else "sumo",
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--duration-log.disable",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed_value + self.episode_count)
        ]

        traci.start(sumo_cmd)
        self.sumo_running = True

        self.current_step = 0
        self.episode_count += 1
        self.last_rates = np.array([0.5, 0.5, 0.5])
        self.ramp_inflows = [0.0, 0.0, 0.0]

        # 清空指标
        for key in self.episode_metrics:
            if isinstance(self.episode_metrics[key], list):
                # 检查是否是嵌套列表(3个ramp的数据)
                if len(self.episode_metrics[key]) > 0 and isinstance(self.episode_metrics[key][0], list):
                    self.episode_metrics[key] = [[], [], []]
                else:
                    self.episode_metrics[key] = []

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """执行一步"""
        # 1) 动作解析和裁剪
        action = np.clip(action, 0.0, 1.0)

        # 2) 转换为实际放行率 (veh/h)
        rates = [self.MIN_RATE + a * (self.MAX_RATE - self.MIN_RATE)
                 for a in action]

        # 3) 应用安全约束
        safe_rates = [self._safety_filter(rates[i], i) for i in range(3)]

        # 4) 应用控制到3个meter
        for i, rate in enumerate(safe_rates):
            self._apply_metering(rate, i)

        # 5) 仿真前进delta_time步
        for _ in range(self.delta_time):
            traci.simulationStep()

        # 6) 更新流入速率统计
        self._update_inflows()

        # 7) 获取新观测
        obs = self._get_observation()

        # 8) 计算奖励 (这里是核心!)
        reward = self._calculate_reward(safe_rates)

        # 9) 判断是否结束
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 10) 记录指标
        self._record_metrics(reward, safe_rates)

        # 11) 更新last_rates
        self.last_rates = action.copy()

        # 12) 返回info
        info = {}
        if terminated:
            info = self._get_episode_summary()
            if self.episode_count % 5 == 0:
                self._print_episode_summary(info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """关闭环境"""
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

    # -----------------------------
    # 核心功能1: 观测
    # -----------------------------

    def _get_observation(self) -> np.ndarray:
        """
        获取26维观测
        """
        obs = np.zeros(26, dtype=np.float32)

        try:
            # [0-6] 主线7段速度 (归一化到[0,1])
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                obs[i] = speed / 30.0  # 30 m/s = 108 km/h

            # [7-13] 主线7段占有率
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                occ = traci.edge.getLastStepOccupancy(edge_id) / 100.0
                obs[7 + i] = occ

            # [14-25] 3个ramp的局部状态 (每个4维)
            for ramp_idx in range(3):
                base_idx = 14 + ramp_idx * 4
                ramp_state = self._get_ramp_state(ramp_idx)
                obs[base_idx:base_idx + 4] = ramp_state

        except Exception as e:
            print(f"Warning: Error getting observation: {e}")

        return obs

    def _get_ramp_state(self, ramp_idx: int) -> np.ndarray:
        """
        获取单个ramp的状态 (4维)
        返回: [排队长度, 平均等待, 上一步放行率, 流入速率]
        """
        state = np.zeros(4, dtype=np.float32)

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]

            # [0] 排队长度 (归一化: MAX=30)
            queue = sum(traci.edge.getLastStepHaltingNumber(e)
                        for e in ramp_edges)
            state[0] = min(queue / self.MAX_RAMP_QUEUE, 1.0)

            # [1] 平均等待时间 (归一化: MAX=120秒)
            wait_times = []
            for edge_id in ramp_edges:
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    wait_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id))
            avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
            state[1] = min(avg_wait / 120.0, 1.0)

            # [2] 上一步放行率 (已归一化)
            state[2] = self.last_rates[ramp_idx]

            # [3] 流入速率 (归一化: MAX=1000 veh/h)
            state[3] = min(self.ramp_inflows[ramp_idx] / 1000.0, 1.0)

        except Exception as e:
            print(f"Warning: Error getting ramp {ramp_idx} state: {e}")

        return state

    def _update_inflows(self):
        """更新各ramp的流入速率估计"""
        try:
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                # 统计本interval通过meter的车辆数
                vehs_count = sum(traci.edge.getLastStepVehicleNumber(e)
                                 for e in ramp_edges)
                # 转换为veh/h
                self.ramp_inflows[i] = (vehs_count / self.delta_time) * 3600
        except Exception as e:
            print(f"Warning: Error updating inflows: {e}")

    # -----------------------------
    # 核心功能2: 控制
    # -----------------------------

    def _safety_filter(self, rate: float, ramp_idx: int) -> float:
        """
        安全约束: 软化版本

        规则:
        1. 如果ramp排队过长 → 增加放行
        2. 如果下游拥堵 → 减少放行
        """
        rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]

            # 1) 检查ramp排队
            queue = sum(traci.edge.getLastStepHaltingNumber(e)
                        for e in ramp_edges)

            if queue > 0.7 * self.MAX_RAMP_QUEUE:
                # 排队接近上限,渐进增加放行率
                excess = (queue - 0.7 * self.MAX_RAMP_QUEUE) / (0.3 * self.MAX_RAMP_QUEUE)
                rate = rate + excess * (self.MAX_RATE - rate) * 0.5

            # 2) 检查下游占有率
            # Ramp1看main_1, Ramp2看main_3, Ramp3看main_5
            downstream_edge = self.MERGE_EDGES[ramp_idx]
            downstream_occ = traci.edge.getLastStepOccupancy(downstream_edge)

            if downstream_occ > 40.0:  # 占有率>40%
                # 下游拥堵,限制放行
                rate = min(rate, 0.6 * self.MAX_RATE)
            elif downstream_occ < 15.0:  # 占有率<15%
                # 下游很空,可以增加放行
                rate = max(rate, 0.7 * self.MAX_RATE)

        except Exception as e:
            print(f"Warning: Error in safety filter for ramp {ramp_idx}: {e}")

        return rate

    def _apply_metering(self, rate: float, ramp_idx: int):
        """应用控制到指定meter"""
        try:
            rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))
            meter_id = self.METER_IDS[ramp_idx]

            # 计算红绿灯时长
            headway = 3600.0 / rate  # 秒/车
            green_time = 2
            red_time = max(int(round(headway - green_time)), 1)

            logic = traci.trafficlight.Logic(
                programID=f"rl_control_{ramp_idx}",
                type=0,
                currentPhaseIndex=0,
                phases=[
                    traci.trafficlight.Phase(green_time, "G"),
                    traci.trafficlight.Phase(red_time, "r")
                ]
            )

            traci.trafficlight.setProgramLogic(meter_id, logic)
            traci.trafficlight.setProgram(meter_id, f"rl_control_{ramp_idx}")

        except Exception as e:
            print(f"Warning: Error applying metering to ramp {ramp_idx}: {e}")

    # -----------------------------
    # 核心功能3: 奖励函数 ⭐
    # -----------------------------

    def _calculate_reward(self, rates: List[float]) -> float:
        """
        奖励函数 - 清晰版本

        目标:
        1. 主线速度高 → 正奖励
        2. 吞吐量大 → 正奖励
        3. Ramp排队少 → 减少惩罚
        4. 公平性好 → 各ramp延误相近

        权重设计:
        - w_speed = 3.0      (主线速度最重要)
        - w_throughput = 1.0 (吞吐量次要)
        - w_queue = -1.5     (排队惩罚)
        - w_fairness = 0.5   (公平性奖励)
        """
        reward = 0.0

        try:
            # ===== 1. 主线速度奖励 =====
            # 关注瓶颈区的速度
            bottleneck_speeds = []
            for edge_id in self.BOTTLENECK_EDGES:
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                bottleneck_speeds.append(speed)

            avg_bottleneck_speed = float(np.mean(bottleneck_speeds))
            speed_reward = (avg_bottleneck_speed / 30.0) ** 2  # 平方强调高速

            # ===== 2. 吞吐量奖励 =====
            # 吞吐量 = 速度 × 车辆数
            throughput = 0.0
            for edge_id in self.MAINLINE_EDGES:
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                vehs = traci.edge.getLastStepVehicleNumber(edge_id)
                throughput += speed * vehs

            throughput_reward = throughput / 1500.0  # 归一化

            # ===== 3. Ramp排队惩罚 =====
            total_queue = 0
            ramp_queues = []
            for ramp_edges in self.RAMP_EDGES:
                queue = sum(traci.edge.getLastStepHaltingNumber(e)
                            for e in ramp_edges)
                total_queue += queue
                ramp_queues.append(queue)

            queue_penalty = (total_queue / (3 * self.MAX_RAMP_QUEUE)) ** 1.5

            # ===== 4. 公平性奖励 =====
            # 各ramp的等待时间标准差越小越好
            ramp_waits = []
            for ramp_edges in self.RAMP_EDGES:
                wait_times = []
                for edge_id in ramp_edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                        wait_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id))
                avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
                ramp_waits.append(avg_wait)

            fairness_reward = 1.0 - min(np.std(ramp_waits) / 60.0, 1.0)

            # ===== 5. 严重拥堵惩罚 =====
            congestion_penalty = 0.0
            if avg_bottleneck_speed < 8.0:
                congestion_penalty = 2.0  # 严重拥堵
            elif avg_bottleneck_speed < 12.0:
                congestion_penalty = 1.0  # 中度拥堵

            # ===== 6. 组合奖励 =====
            reward = (
                    3.0 * speed_reward +  # [0, ~3]
                    1.0 * throughput_reward -  # [0, ~1]
                    1.5 * queue_penalty -  # [0, ~1.5]
                    congestion_penalty +  # [0, 2]
                    0.5 * fairness_reward  # [0, 0.5]
            )

            # ===== 7. 平滑性惩罚 (避免频繁大幅调整) =====
            # 计算动作变化幅度
            if hasattr(self, 'last_rates'):
                rate_changes = [abs(rates[i] - self.last_rates[i] * (self.MAX_RATE - self.MIN_RATE) - self.MIN_RATE)
                                for i in range(3)]
                avg_change = np.mean(rate_changes) / self.MAX_RATE
                if avg_change > 0.3:  # 变化超过30%
                    reward -= 0.5

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            reward = 0.0

        return float(reward)

    # -----------------------------
    # 指标记录和输出
    # -----------------------------

    def _record_metrics(self, reward: float, rates: List[float]):
        """记录详细指标"""
        try:
            self.episode_metrics['reward'].append(reward)

            # 主线速度
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e)
                for e in self.MAINLINE_EDGES
            ])
            self.episode_metrics['mainline_speed'].append(mainline_speed)

            # 各ramp排队
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                queue = sum(traci.edge.getLastStepHaltingNumber(e)
                            for e in ramp_edges)
                self.episode_metrics['ramp_queues'][i].append(queue)

            # 吞吐量
            throughput = sum(
                traci.edge.getLastStepMeanSpeed(e) *
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            self.episode_metrics['throughput'].append(throughput)

            # 放行率
            for i, rate in enumerate(rates):
                self.episode_metrics['metering_rates'][i].append(rate)

            # 总延误
            total_delay = 0.0
            for edge_list in self.RAMP_EDGES + [self.MAINLINE_EDGES]:
                edges = edge_list if isinstance(edge_list, list) else [edge_list]
                for e in edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(e):
                        total_delay += traci.vehicle.getAccumulatedWaitingTime(veh_id)
            self.episode_metrics['total_delay'].append(total_delay)

        except Exception as e:
            print(f"Warning: Error recording metrics: {e}")

    def _get_episode_summary(self) -> Dict[str, Any]:
        """生成episode摘要"""
        summary = {}

        # 基础指标
        summary['total_reward'] = float(np.sum(self.episode_metrics['reward']))
        summary['avg_reward'] = float(np.mean(self.episode_metrics['reward']))
        summary['avg_mainline_speed'] = float(np.mean(self.episode_metrics['mainline_speed']))
        summary['avg_throughput'] = float(np.mean(self.episode_metrics['throughput']))

        # 各ramp排队
        for i in range(3):
            summary[f'avg_ramp{i + 1}_queue'] = float(np.mean(self.episode_metrics['ramp_queues'][i]))
            summary[f'avg_ramp{i + 1}_rate'] = float(np.mean(self.episode_metrics['metering_rates'][i]))

        # 总延误
        if len(self.episode_metrics['total_delay']) > 0:
            summary['total_delay'] = float(np.sum(self.episode_metrics['total_delay']))

        summary['episode_steps'] = self.current_step
        summary['episode_number'] = self.episode_count

        # 保存
        if self.save_metrics:
            filename = os.path.join(
                self.metrics_dir,
                f"episode_{self.episode_count:04d}.json"
            )
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)

        return summary

    def _print_episode_summary(self, info: Dict):
        """打印episode摘要"""
        print(f"\n{'=' * 70}")
        print(f"Episode {self.episode_count} Summary")
        print(f"{'=' * 70}")
        print(f"  Total Reward:       {info.get('total_reward', 0):8.2f}")
        print(f"  Avg Reward:         {info.get('avg_reward', 0):8.3f}")
        print(f"  Mainline Speed:     {info.get('avg_mainline_speed', 0):8.2f} m/s")
        print(f"  Throughput:         {info.get('avg_throughput', 0):8.1f}")
        print(f"  Ramp Queues:")
        for i in range(3):
            queue = info.get(f'avg_ramp{i + 1}_queue', 0)
            rate = info.get(f'avg_ramp{i + 1}_rate', 0)
            print(f"    Ramp{i + 1}: {queue:5.1f} veh (rate: {rate:6.1f} veh/h)")
        if 'total_delay' in info:
            print(f"  Total Delay:        {info.get('total_delay', 0):8.1f} veh·s")
        print(f"{'=' * 70}\n")


# =====================================================
# 训练回调
# =====================================================

class TrainingCallback(BaseCallback):
    """训练监控回调"""

    def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            avg_reward = float(np.mean(recent_rewards))

            if self.verbose:
                print(f"\n[Step {self.n_calls:,}] Avg Reward: {avg_reward:.3f}")

        return True


# =====================================================
# 训练函数
# =====================================================

def train_ppo(
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        total_timesteps: int = 200_000,
        n_envs: int = 4,
        model_save_path: str = "./models/ppo_multi_ramp",
        log_path: str = "./logs"
):
    """训练PPO模型"""

    print(f"\n{'=' * 70}")
    print("Training PPO Multi-Ramp Metering")
    print(f"{'=' * 70}")
    print(f"Config:         {sumo_cfg}")
    print(f"Timesteps:      {total_timesteps:,}")
    print(f"Envs:           {n_envs}")
    print(f"Control Freq:   30 seconds")
    print(f"Episode Length: 3600 seconds (120 steps)")
    print(f"{'=' * 70}\n")

    def make_env(rank):
        def _init():
            env = MultiRampEnv(
                sumo_cfg=sumo_cfg,
                gui=False,
                seed=1000 + rank,
                save_metrics=False,
                max_steps=120,
                delta_time=30
            )
            return Monitor(env)

        return _init

    # 创建并行环境
    if n_envs > 1 and os.name != "nt":
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callback = TrainingCallback(
        save_freq=1000,
        save_path=log_path,
        verbose=1
    )

    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    model.save(model_save_path)
    print(f"\n✅ Model saved to {model_save_path}.zip\n")

    env.close()
    return model


# =====================================================
# 评估函数
# =====================================================

def evaluate_policy(
        model_path: str = None,
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        n_episodes: int = 5,
        gui: bool = False,
        policy_type: str = "ppo"
):
    """
    评估策略

    policy_type:
    - "ppo": 训练好的PPO
    - "fixed": 固定放行率(600 veh/h)
    - "no_control": 无控制(最大放行)
    """

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {policy_type.upper()}")
    print(f"{'=' * 70}\n")

    # 加载模型(如果是PPO)
    model = None
    if policy_type == "ppo":
        if model_path is None:
            model_path = "./models/ppo_multi_ramp"
        model = PPO.load(model_path)
        print(f"Loaded model: {model_path}.zip\n")

    # 创建环境
    env = MultiRampEnv(
        sumo_cfg=sumo_cfg,
        gui=gui,
        seed=2000,
        save_metrics=True,
        metrics_dir=f"./eval_{policy_type}",
        max_steps=120,
        delta_time=30
    )

    all_summaries = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...")
        obs, info = env.reset()
        done = False

        while not done:
            # 选择动作
            if policy_type == "ppo":
                action, _ = model.predict(obs, deterministic=True)
            elif policy_type == "fixed":
                # 固定600 veh/h
                fixed_rate = (600 - env.MIN_RATE) / (env.MAX_RATE - env.MIN_RATE)
                action = np.array([fixed_rate, fixed_rate, fixed_rate])
            elif policy_type == "no_control":
                # 最大放行
                action = np.array([1.0, 1.0, 1.0])
            else:
                raise ValueError(f"Unknown policy: {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info:
            all_summaries.append(info)
            print(f"  Reward: {info.get('total_reward', 0):.2f}, "
                  f"Speed: {info.get('avg_mainline_speed', 0):.2f} m/s\n")

    env.close()

    # 统计结果
    print(f"\n{'=' * 70}")
    print(f"RESULTS - {policy_type.upper()}")
    print(f"{'=' * 70}")

    metrics = [
        ('avg_mainline_speed', 'Mainline Speed (m/s)'),
        ('avg_throughput', 'Throughput'),
        ('total_reward', 'Total Reward'),
        ('avg_ramp1_queue', 'Ramp1 Queue'),
        ('avg_ramp2_queue', 'Ramp2 Queue'),
        ('avg_ramp3_queue', 'Ramp3 Queue'),
        ('avg_waiting_time', 'Avg Waiting Time (s)'),

    ]

    results = {}
    for key, name in metrics:
        values = [s.get(key, 0) for s in all_summaries]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        results[key] = {'mean': mean_val, 'std': std_val}
        print(f"  {name:25s}: {mean_val:8.2f} ± {std_val:6.2f}")

    print(f"{'=' * 70}\n")

    # 保存
    output_file = f"eval_results_{policy_type}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'policy': policy_type,
            'n_episodes': n_episodes,
            'results': results,
            'summaries': all_summaries
        }, f, indent=2)

    print(f"✅ Results saved to {output_file}\n")
    return results


def compare_policies(
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        n_episodes: int = 5,
        model_path: str = "./models/ppo_multi_ramp"
):
    """对比所有策略"""

    print("\n" + "=" * 70)
    print("COMPARING POLICIES")
    print("=" * 70 + "\n")

    policies = ["no_control", "fixed", "ppo"]
    all_results = {}

    for policy in policies:
        results = evaluate_policy(
            model_path=model_path,
            sumo_cfg=sumo_cfg,
            n_episodes=n_episodes,
            gui=False,
            policy_type=policy
        )
        all_results[policy] = results

    # 对比表
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'No Control':>15} {'Fixed':>15} {'PPO':>15}")
    print("-" * 70)

    metrics = [
        'avg_mainline_speed',
        'avg_throughput',
        'total_reward',
        'avg_ramp1_queue',
    ]

    for metric in metrics:
        row = f"{metric:<25}"
        for policy in policies:
            val = all_results[policy][metric]['mean']
            row += f"{val:>15.2f}"
        print(row)

    print("=" * 70 + "\n")

    # 保存
    with open("comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✅ Comparison saved to comparison.json\n")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Ramp PPO Control")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Evaluate model")
    parser.add_argument("--compare", action="store_true", help="Compare all policies")
    parser.add_argument("--gui", action="store_true", help="Use GUI")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs")
    parser.add_argument("--n-episodes", type=int, default=5, help="Eval episodes")
    parser.add_argument("--policy", type=str, default="ppo",
                        choices=["ppo", "fixed", "no_control"])

    args = parser.parse_args()

    if args.train:
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs
        )
    elif args.eval:
        evaluate_policy(
            n_episodes=args.n_episodes,
            gui=args.gui,
            policy_type=args.policy
        )
    elif args.compare:
        compare_policies(n_episodes=args.n_episodes)
    else:
        print("Please specify --train, --eval, or --compare")
        parser.print_help()