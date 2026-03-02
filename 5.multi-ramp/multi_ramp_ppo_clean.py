#!/usr/bin/env python3
"""
多Ramp PPO控制 - 稳定版

改动概要：
1. 奖励函数简化为：瓶颈速度奖励 + 轻微排队惩罚，显著提高 PPO 可学性
2. safety_filter 简化，不再大幅覆盖 PPO 动作，只做极端保护
3. 控制频率提高：delta_time = 10s, max_steps = 360（仍然一小时）
4. PPO 超参数针对 SUMO 交通仿真重新调优
5. 增加 reward 各项及关键指标的逐步记录，并提供绘图函数
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
            max_steps: int = 360,   # 3600秒 / 10秒 = 360步
            delta_time: int = 10,   # 控制间隔10秒（更频繁控制）
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
            ['ramp3_0', 'ramp3_1']   # Ramp3的两段
        ]
        self.METER_IDS = ['meter1', 'meter2', 'meter3']

        # 关键路段
        self.MERGE_EDGES = ['main_1', 'main_3', 'main_5']       # 3个合流区
        self.BOTTLENECK_EDGES = ['main_2', 'main_4', 'main_6']  # 3个瓶颈区

        # 物理与容量参数
        self.FREE_FLOW_SPEED = 30.0   # m/s, 约108 km/h
        self.MAX_RAMP_QUEUE = 30      # veh, 由ramp长度/车长估计
        self.MAX_RAMP_WAIT = 300.0    # s, 最大容忍等待时间
        self.MIN_RATE = 200.0         # veh/h, 最小放行率
        self.MAX_RATE = 1200.0        # veh/h, 最大放行率
        self.MAX_RAMP_INFLOW = 1500.0 # veh/h, 峰值需求上界

        # ===== 观测和动作空间 =====
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(26,),
            dtype=np.float32
        )

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

        # 指标记录（加入 reward 分解 &瓶颈速度/队列等）
        self.episode_metrics = {
            'reward': [],
            'speed_term': [],
            'queue_term': [],
            'bottleneck_speed': [],
            'total_ramp_queue': [],
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

        # sumo_cmd = [
        #     "sumo-gui" if self.gui else "sumo",
        #     "-c", self.sumo_cfg,
        #     "--start",
        #     "--quit-on-end",
        #     "--no-warnings",
        #     "--duration-log.disable",
        #     "--time-to-teleport", "-1",
        #     "--seed", str(self.seed_value + self.episode_count)
        # ]
        sumo_cmd = [
            "sumo-gui" if self.gui else "sumo",
            "-n", "multi_ramp_network.net.xml",  # 👈 直接指定 net
            "-r", "multi_ramp_routes.rou.xml",  # 👈 直接指定 routes
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
        rates = [self.MIN_RATE + float(a) * (self.MAX_RATE - self.MIN_RATE)
                 for a in action]

        # 3) 应用温和的安全约束（不再强行覆盖 RL 动作，只在极端情况稍微调整）
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

        # 8) 计算奖励（返回 reward 和分解）
        reward, reward_info = self._calculate_reward(safe_rates)

        # 9) 判断是否结束
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 10) 记录指标
        self._record_metrics(reward, safe_rates, reward_info)

        # 11) 更新last_rates（归一化到[0,1]）
        self.last_rates = np.clip(action.copy(), 0.0, 1.0)

        # 12) 返回info
        info = {}
        if terminated:
            info = self._get_episode_summary()
            if self.episode_count % 5 == 0:
                self._print_episode_summary(info)

        return obs, float(reward), terminated, truncated, info

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
        获取26维观测:
        [0-6]   主线速度 / FREE_FLOW_SPEED
        [7-13]  主线占有率 (0-1)
        [14-25] 3个ramp状态，每个4维:
                [排队长度/MAX_RAMP_QUEUE,
                 平均等待时间/MAX_RAMP_WAIT,
                 上一步放行率(0-1),
                 流入速率/MAX_RAMP_INFLOW]
        """
        obs = np.zeros(26, dtype=np.float32)

        try:
            # [0-6] 主线7段速度 (归一化到[0,1])
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                v_norm = speed / self.FREE_FLOW_SPEED if self.FREE_FLOW_SPEED > 0 else 0.0
                obs[i] = float(np.clip(v_norm, 0.0, 1.0))

            # [7-13] 主线7段占有率 (0-1)
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                occ = traci.edge.getLastStepOccupancy(edge_id) / 100.0
                obs[7 + i] = float(np.clip(occ, 0.0, 1.0))

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
        返回: [排队长度/MAX_RAMP_QUEUE,
              平均等待时间/MAX_RAMP_WAIT,
              上一步放行率(0-1),
              流入速率/MAX_RAMP_INFLOW]
        """
        state = np.zeros(4, dtype=np.float32)

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]

            # [0] 排队长度
            queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)
            state[0] = float(np.clip(queue / self.MAX_RAMP_QUEUE, 0.0, 1.0))

            # [1] 平均等待时间
            wait_times = []
            for edge_id in ramp_edges:
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    wait_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id))
            avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
            state[1] = float(np.clip(avg_wait / self.MAX_RAMP_WAIT, 0.0, 1.0))

            # [2] 上一步放行率 (动作本身就是0-1)
            state[2] = float(self.last_rates[ramp_idx])

            # [3] 流入速率
            inflow = self.ramp_inflows[ramp_idx]
            state[3] = float(np.clip(inflow / self.MAX_RAMP_INFLOW, 0.0, 1.0))

        except Exception as e:
            print(f"Warning: Error getting ramp {ramp_idx} state: {e}")

        return state

    # -----------------------------
    # 核心功能2: 控制
    # -----------------------------

    def _update_inflows(self):
        """更新各ramp的流入速率估计"""
        try:
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                vehs_count = sum(
                    traci.edge.getLastStepVehicleNumber(e)
                    for e in ramp_edges
                )
                self.ramp_inflows[i] = (vehs_count / self.delta_time) * 3600
        except Exception as e:
            print(f"Warning: Error updating inflows: {e}")

    def _safety_filter(self, rate: float, ramp_idx: int) -> float:
        """
        简化的安全约束（不会强行覆盖 PPO 动作）：

        规则：
        1. 基本裁剪到 [MIN_RATE, MAX_RATE]
        2. 如果 ramp 排队非常接近上限 → 轻微往上推放行率
        （不再根据下游占有率硬剪，避免破坏 RL 学习）
        """
        rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]
            queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)

            # 排队 > 90% 上限时，温和提高放行率（只做一点“救火”，不完全覆盖 RL 决策）
            if queue > 0.9 * self.MAX_RAMP_QUEUE:
                # 向 MAX_RATE 推一点点（线性插值）
                alpha = min((queue - 0.9 * self.MAX_RAMP_QUEUE) / (0.1 * self.MAX_RAMP_QUEUE), 1.0)
                rate = (1 - 0.3 * alpha) * rate + 0.3 * alpha * self.MAX_RATE

        except Exception as e:
            print(f"Warning: Error in safety filter for ramp {ramp_idx}: {e}")

        return float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))

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
    # 核心功能3: 奖励函数（简化且可学习）⭐
    # -----------------------------

    def _calculate_reward(self, rates: List[float]) -> (float, Dict[str, float]):
        """
        奖励函数 - 简化稳定版

        目标（两项）：
        1. 提高主线瓶颈区平均速度（核心）
        2. 轻微惩罚总 ramp 排队（防止极端情况）

        形式：
            speed_term = avg_bottleneck_speed / FREE_FLOW_SPEED
            queue_term = total_queue / (3 * MAX_RAMP_QUEUE)
            reward = speed_term - 0.1 * queue_term
        """
        reward = 0.0
        info = {
            "speed_term": 0.0,
            "queue_term": 0.0,
            "bottleneck_speed": 0.0,
            "total_ramp_queue": 0.0,
        }

        try:
            # 1. 瓶颈速度
            bottleneck_speeds = []
            for edge_id in self.BOTTLENECK_EDGES:
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                bottleneck_speeds.append(speed)
            avg_bottleneck_speed = float(np.mean(bottleneck_speeds)) if bottleneck_speeds else 0.0
            speed_term = avg_bottleneck_speed / self.FREE_FLOW_SPEED if self.FREE_FLOW_SPEED > 0 else 0.0
            speed_term = float(np.clip(speed_term, 0.0, 1.5))  # 略放宽一点上限

            # 2. 总 ramp 排队
            total_queue = 0.0
            for ramp_edges in self.RAMP_EDGES:
                queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)
                total_queue += queue
            if self.MAX_RAMP_QUEUE > 0:
                queue_term = total_queue / (3.0 * self.MAX_RAMP_QUEUE)
            else:
                queue_term = 0.0
            queue_term = float(np.clip(queue_term, 0.0, 2.0))

            # 3. 组合奖励
            reward = speed_term - 0.1 * queue_term

            # 填充 info 方便记录和画图
            info["speed_term"] = speed_term
            info["queue_term"] = queue_term
            info["bottleneck_speed"] = avg_bottleneck_speed
            info["total_ramp_queue"] = total_queue

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            reward = 0.0

        return float(reward), info

    # -----------------------------
    # 指标记录和输出
    # -----------------------------

    def _record_metrics(self, reward: float, rates: List[float], reward_info: Dict[str, float]):
        """记录详细指标（包括 reward 各项）"""
        try:
            self.episode_metrics['reward'].append(float(reward))
            self.episode_metrics['speed_term'].append(float(reward_info.get("speed_term", 0.0)))
            self.episode_metrics['queue_term'].append(float(reward_info.get("queue_term", 0.0)))
            self.episode_metrics['bottleneck_speed'].append(float(reward_info.get("bottleneck_speed", 0.0)))
            self.episode_metrics['total_ramp_queue'].append(float(reward_info.get("total_ramp_queue", 0.0)))

            # 主线平均速度（全段）
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e)
                for e in self.MAINLINE_EDGES
            ])
            self.episode_metrics['mainline_speed'].append(float(mainline_speed))

            # 各ramp排队
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                queue = sum(traci.edge.getLastStepHaltingNumber(e)
                            for e in ramp_edges)
                self.episode_metrics['ramp_queues'][i].append(float(queue))

            # 吞吐量（仍然记录，但不进 reward）
            throughput = sum(
                traci.edge.getLastStepMeanSpeed(e) *
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            self.episode_metrics['throughput'].append(float(throughput))

            # 放行率
            for i, rate in enumerate(rates):
                self.episode_metrics['metering_rates'][i].append(float(rate))

            # 总延误（仅记录）
            total_delay = 0.0
            for edge_list in self.RAMP_EDGES + [self.MAINLINE_EDGES]:
                edges = edge_list if isinstance(edge_list, list) else [edge_list]
                for e in edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(e):
                        total_delay += traci.vehicle.getAccumulatedWaitingTime(veh_id)
            self.episode_metrics['total_delay'].append(float(total_delay))

        except Exception as e:
            print(f"Warning: Error recording metrics: {e}")

    def _get_episode_summary(self) -> Dict[str, Any]:
        """生成episode摘要 + 保存逐步曲线"""
        summary = {}

        # 基础指标
        summary['total_reward'] = float(np.sum(self.episode_metrics['reward'])) if self.episode_metrics['reward'] else 0.0
        summary['avg_reward'] = float(np.mean(self.episode_metrics['reward'])) if self.episode_metrics['reward'] else 0.0
        summary['avg_mainline_speed'] = float(np.mean(self.episode_metrics['mainline_speed'])) if self.episode_metrics['mainline_speed'] else 0.0
        summary['avg_throughput'] = float(np.mean(self.episode_metrics['throughput'])) if self.episode_metrics['throughput'] else 0.0

        # 各ramp排队
        for i in range(3):
            q_list = self.episode_metrics['ramp_queues'][i]
            r_list = self.episode_metrics['metering_rates'][i]
            summary[f'avg_ramp{i + 1}_queue'] = float(np.mean(q_list)) if q_list else 0.0
            summary[f'avg_ramp{i + 1}_rate'] = float(np.mean(r_list)) if r_list else 0.0

        # 总延误
        if len(self.episode_metrics['total_delay']) > 0:
            summary['total_delay'] = float(np.sum(self.episode_metrics['total_delay']))

        summary['episode_steps'] = self.current_step
        summary['episode_number'] = self.episode_count

        # 保存 summary + series
        if self.save_metrics:
            os.makedirs(self.metrics_dir, exist_ok=True)
            # 1) 摘要
            filename = os.path.join(
                self.metrics_dir,
                f"episode_{self.episode_count:04d}.json"
            )
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)

            # 2) 逐步曲线（reward 各项 & 关键指标）
            series = {
                'reward': self.episode_metrics['reward'],
                'speed_term': self.episode_metrics['speed_term'],
                'queue_term': self.episode_metrics['queue_term'],
                'bottleneck_speed': self.episode_metrics['bottleneck_speed'],
                'total_ramp_queue': self.episode_metrics['total_ramp_queue'],
                'mainline_speed': self.episode_metrics['mainline_speed'],
                'ramp_queues': self.episode_metrics['ramp_queues'],
                'throughput': self.episode_metrics['throughput'],
                'metering_rates': self.episode_metrics['metering_rates'],
                'total_delay': self.episode_metrics['total_delay'],
            }
            series_file = os.path.join(
                self.metrics_dir,
                f"episode_{self.episode_count:04d}_series.json"
            )
            with open(series_file, 'w') as f:
                json.dump(series, f, indent=2)

        return summary

    def _print_episode_summary(self, info: Dict):
        """打印episode摘要"""
        print(f"\n{'=' * 70}")
        print(f"Episode {info.get('episode_number', self.episode_count)} Summary")
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

    def __init__(self, save_freq=10000, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            avg_reward = float(np.mean(recent_rewards))

            if self.verbose:
                print(f"\n[Step {self.n_calls:,}] Avg Reward (last episodes): {avg_reward:.3f}")

        return True


# =====================================================
# 训练函数（PPO 超参数已调整）
# =====================================================

def train_ppo(
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        total_timesteps: int = 100_000,
        n_envs: int = 4,
        model_save_path: str = "./models/ppo_multi_ramp",
        log_path: str = "./logs"
):
    """训练PPO模型"""

    print(f"\n{'=' * 70}")
    print("Training PPO Multi-Ramp Metering (Stable Reward)")
    print(f"{'=' * 70}")
    print(f"Config:         {sumo_cfg}")
    print(f"Timesteps:      {total_timesteps:,}")
    print(f"Envs:           {n_envs}")
    print(f"Control Freq:   10 seconds")
    print(f"Episode Length: 3600 seconds (360 steps)")
    print(f"{'=' * 70}\n")

    def make_env(rank):
        def _init():
            env = MultiRampEnv(
                sumo_cfg=sumo_cfg,
                gui=False,
                seed=1000 + rank,
                save_metrics=False,
                max_steps=360,
                delta_time=10
            )
            return Monitor(env)

        return _init

    # 创建并行环境
    if n_envs > 1 and os.name != "nt":
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # 创建PPO模型（参数适配 SUMO+交通控制）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,          # 每次 rollout 的步数
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
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
        save_freq=10_000,
        save_path=log_path,
        verbose=1
    )

    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False
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

    # 创建环境（这里开启 save_metrics 便于画曲线）
    env = MultiRampEnv(
        sumo_cfg=sumo_cfg,
        gui=gui,
        seed=2000,
        save_metrics=True,
        metrics_dir=f"./eval_{policy_type}",
        max_steps=360,
        delta_time=10
    )

    all_summaries = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...")
        obs, info = env.reset()
        done = False

        while not done:
            if policy_type == "ppo":
                action, _ = model.predict(obs, deterministic=True)
            elif policy_type == "fixed":
                fixed_rate = (600 - env.MIN_RATE) / (env.MAX_RATE - env.MIN_RATE)
                action = np.array([fixed_rate, fixed_rate, fixed_rate], dtype=np.float32)
            elif policy_type == "no_control":
                action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
        ('total_delay', 'Total Delay (veh·s)'),
    ]

    results = {}
    for key, name in metrics:
        values = [s.get(key, 0) for s in all_summaries]
        mean_val = float(np.mean(values)) if values else 0.0
        std_val = float(np.std(values)) if values else 0.0
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
# 画 reward 各项 & 关键指标曲线的辅助函数
# =====================================================

def plot_episode_series(
        metrics_dir: str = "./eval_ppo",
        episode: int = None,
        output_path: str = None
):
    """
    从 metrics_dir 中读取某个 episode 的 *_series.json，
    画出 reward / speed_term / queue_term 等随步数变化的曲线。

    使用方法（例如在训练+评估后）：
        python multi_ramp_ppo_clean.py --plot --metrics-dir ./eval_ppo --episode 1
    或在交互式环境中：
        from multi_ramp_ppo_clean import plot_episode_series
        plot_episode_series("./eval_ppo", episode=1)
    """
    import glob
    import matplotlib.pyplot as plt

    pattern = os.path.join(metrics_dir, "episode_*_series.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No series files found in {metrics_dir}")
        return

    # 如果未指定 episode，就用最后一个
    if episode is None:
        series_file = files[-1]
    else:
        series_file = os.path.join(metrics_dir, f"episode_{episode:04d}_series.json")
        if not os.path.exists(series_file):
            print(f"{series_file} not found, fallback to last episode.")
            series_file = files[-1]

    print(f"Loading series from: {series_file}")
    with open(series_file, 'r') as f:
        series = json.load(f)

    steps = list(range(len(series['reward'])))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, series['reward'], label='reward')
    plt.plot(steps, series['speed_term'], label='speed_term')
    plt.plot(steps, series['queue_term'], label='queue_term')
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Reward and Components over Steps")
    plt.legend()
    plt.grid(True)

    if output_path is None:
        output_path = os.path.join(metrics_dir, "reward_components.png")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Reward component curves saved to {output_path}")
    plt.close()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Ramp PPO Control (Stable Reward)")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Evaluate model")
    parser.add_argument("--compare", action="store_true", help="Compare all policies")
    parser.add_argument("--plot", action="store_true", help="Plot reward component curves from metrics")
    parser.add_argument("--gui", action="store_true", help="Use GUI for SUMO")
    parser.add_argument("--timesteps", type=int, default=8000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel envs")
    parser.add_argument("--n-episodes", type=int, default=5, help="Eval episodes")
    parser.add_argument("--policy", type=str, default="ppo",
                        choices=["ppo", "fixed", "no_control"])
    parser.add_argument("--metrics-dir", type=str, default="./eval_ppo", help="Metrics directory for plotting")
    parser.add_argument("--episode", type=int, default=None, help="Which episode to plot (default: last)")

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
    elif args.plot:
        plot_episode_series(metrics_dir=args.metrics_dir, episode=args.episode)
    else:
        print("Please specify --train, --eval, --compare or --plot")
        parser.print_help()
