#!/usr/bin/env python3
"""
PPO Ramp Metering - 改进稳定版

主要改动（相对你之前的版本）：
1. 奖励函数重新设计：
   - 同时考虑：瓶颈速度、主线速度、吞吐量、匝道排队、拥堵惩罚、动作平滑性
   - 避免 PPO 学到“几乎不放车但主线很快”的极端策略

2. 安全约束：
   - 不再在 _safety_filter 里强行篡改 PPO 动作
   - 改为在奖励中对严重排队/拥堵进行“软惩罚”，保留动作-反馈因果

3. 动作变化惩罚修复：
   - 原版由于 last_rate 更新位置问题，rate_change 永远为 0
   - 现在在 step() 中将 prev_rate 显式传给 _calculate_reward

4. 训练接口保持不变：
   - train_ppo / evaluate_policy / compare_policies 完整保留
   - 你可以直接用命令行参数 --train / --eval / --compare
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
# 环境类
# =====================================================

class RampMeteringEnv(gym.Env):
    """
    改进的 Ramp Metering 环境（稳定版）

    观测空间（12维）：
    0-2: 主线3段速度 / 30
    3-5: 主线3段占有率
    6: 匝道排队长度 / MAX_RAMP_QUEUE
    7: 匝道平均等待时间 / 120
    8: 瓶颈车辆数 / 40
    9: 匝道当前车辆数 / 历史平均（负载比）
    10: 上一步放行速率（归一化）
    11: 主线总车辆数 / 80

    动作空间：
    - 归一化匝道放行速率 [0, 1]

    奖励函数（核心思想）：
    - + 瓶颈速度（权重最高）
    - + 主线平均速度
    - + 吞吐量（所有主线边上的 speed * veh 总和）
    - - 匝道排队
    - - 严重拥堵惩罚（低速 + 高占有率）
    - - 过大速率变化（鼓励平滑控制）
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            sumo_cfg: str,
            max_steps: int = 720,      # 默认 1 小时仿真（取决于 delta_time）
            delta_time: int = 5,       # 每个 RL step 的 SUMO 步数（秒）
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

        # 网络配置
        self.MAINLINE_EDGES = ['main_1', 'main_2', 'main_3']
        self.RAMP_EDGES = ['ramp_0', 'ramp_1']
        self.MERGE_EDGE = 'main_2'
        self.BOTTLENECK_EDGE = 'main_3'
        self.METER_ID = 'meter'

        # 容量参数
        self.MAINLINE_CAPACITY = 4000
        self.BOTTLENECK_CAPACITY = 4000
        self.MAX_RAMP_QUEUE = 30

        # 放行速率范围 (veh/h)
        self.MIN_RATE = 400.0
        self.MAX_RATE = 1200.0

        # 观测空间（12维）
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # 动作空间
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 状态变量
        self.current_step = 0
        self.episode_count = 0
        self.sumo_running = False
        self.last_rate = (self.MIN_RATE + self.MAX_RATE) / 2.0

        # 历史信息追踪
        self.ramp_veh_history = deque(maxlen=3)  # 追踪匝道车辆数变化

        # 指标记录
        self.episode_metrics = {
            'reward': [],
            'mainline_speed': [],
            'merge_speed': [],
            'bottleneck_speed': [],
            'ramp_queue': [],
            'throughput': [],
            'occupancy': [],
            'metering_rate': [],
            'bottleneck_flow': [],  # 瓶颈流量（车辆数）
            'total_delay': []       # 总延误
        }

    # -----------------------------
    # Gym 接口
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
            "--seed", str(self.seed_value + self.episode_count)
        ]

        traci.start(sumo_cmd)
        self.sumo_running = True

        self.current_step = 0
        self.episode_count += 1
        self.last_rate = (self.MIN_RATE + self.MAX_RATE) / 2.0
        self.ramp_veh_history.clear()

        for key in self.episode_metrics:
            self.episode_metrics[key] = []

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """执行一步"""
        # 1) 动作解析（PPO 输出为 [0,1]）
        rate_norm = float(np.clip(action[0], 0.0, 1.0))
        raw_rate = self.MIN_RATE + rate_norm * (self.MAX_RATE - self.MIN_RATE)

        # 2) 只做基本裁剪，不再做复杂“安全逻辑”
        safe_rate = self._safety_filter(raw_rate)

        # 记录旧的放行速率，用于奖励中的平滑性项
        prev_rate = self.last_rate

        # 3) 应用控制（这里会更新 self.last_rate）
        self._apply_metering(safe_rate)

        # 4) 仿真前进 delta_time 秒
        for _ in range(self.delta_time):
            traci.simulationStep()

        # 5) 观测 & 奖励
        obs = self._get_observation()
        reward = self._calculate_reward(current_rate=safe_rate, prev_rate=prev_rate)

        # 6) 终止判断
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 7) 记录指标
        self._record_metrics(reward, safe_rate)

        # 8) Info
        info = {}
        if terminated:
            info = self._get_episode_summary()
            if self.episode_count % 5 == 0:  # 每5个episode打印
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
    # 核心功能：观测、控制、奖励
    # -----------------------------

    def _get_observation(self) -> np.ndarray:
        """
        12维观测空间
        """
        obs = np.zeros(12, dtype=np.float32)

        try:
            # [0-2] 主线3段速度
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                obs[i] = speed / 30.0

            # [3-5] 主线3段占有率
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                occ = traci.edge.getLastStepOccupancy(edge_id)
                obs[3 + i] = occ

            # [6] 匝道排队长度
            ramp_queue = sum(
                traci.edge.getLastStepHaltingNumber(e)
                for e in self.RAMP_EDGES
            )
            obs[6] = min(ramp_queue / self.MAX_RAMP_QUEUE, 1.0)

            # [7] 匝道平均等待时间
            wait_times = []
            for edge_id in self.RAMP_EDGES:
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    wait_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id))
            avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
            obs[7] = min(avg_wait / 120.0, 1.0)

            # [8] 瓶颈车辆数
            bottle_vehs = traci.edge.getLastStepVehicleNumber(self.BOTTLENECK_EDGE)
            obs[8] = min(bottle_vehs / 40.0, 1.0)

            # [9] 匝道当前车辆数 / 历史平均
            current_ramp_vehs = sum(
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.RAMP_EDGES
            )
            self.ramp_veh_history.append(current_ramp_vehs)
            avg_ramp_vehs = np.mean(self.ramp_veh_history) if self.ramp_veh_history else current_ramp_vehs
            obs[9] = min(current_ramp_vehs / max(20.0, avg_ramp_vehs), 1.0)

            # [10] 上一步放行速率（归一化）
            obs[10] = (self.last_rate - self.MIN_RATE) / (self.MAX_RATE - self.MIN_RATE)

            # [11] 主线总车辆数
            mainline_total = sum(
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            obs[11] = min(mainline_total / 80.0, 1.0)

        except Exception as e:
            print(f"Warning: Error getting observation: {e}")
            obs = np.zeros(12, dtype=np.float32)

        return obs

    def _safety_filter(self, rate: float) -> float:
        """
        安全裁剪版：仅做数值裁剪，不做策略篡改

        复杂的“如果排队大就强制提速、如果占有率高就强制减速”的逻辑
        会破坏 PPO 学习动作与结果的映射，这里全部去掉。
        """
        rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))
        return rate

    def _apply_metering(self, rate: float):
        """应用匝道控制：根据放行速率设置红绿灯周期"""
        try:
            rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))
            headway = 3600.0 / rate  # 秒/车

            green_time = 1
            red_time = max(int(round(headway - green_time)), 1)

            logic = traci.trafficlight.Logic(
                programID="rl_control",
                type=0,
                currentPhaseIndex=0,
                phases=[
                    traci.trafficlight.Phase(green_time, "G"),
                    traci.trafficlight.Phase(red_time, "r")
                ]
            )

            traci.trafficlight.setProgramLogic(self.METER_ID, logic)
            traci.trafficlight.setProgram(self.METER_ID, "rl_control")

            # 这里更新 last_rate，用于下一步的平滑性奖励
            self.last_rate = rate

        except Exception as e:
            print(f"Warning: Error applying metering: {e}")

    # def _calculate_reward(self, current_rate: float, prev_rate: float) -> float:
    #     """
    #     改进的奖励函数（平衡速度、吞吐量、排队与拥堵）
    #
    #     组成部分：
    #     1. 瓶颈速度奖励（权重最高）
    #     2. 主线平均速度奖励
    #     3. 吞吐量奖励（所有主线边上的 speed * veh 总和）
    #     4. 匝道排队惩罚
    #     5. 严重拥堵惩罚（低速 + 高占有率）
    #     6. 动作平滑性惩罚（避免频繁大幅调整放行率）
    #     7. 安全软惩罚：排队接近上限或占有率过高时额外惩罚
    #     """
    #     reward = 0.0
    #
    #     try:
    #         # ===== 1. 速度奖励 =====
    #         bottle_speed = traci.edge.getLastStepMeanSpeed(self.BOTTLENECK_EDGE)
    #         bottle_speed_norm = bottle_speed / 30.0
    #         bottle_speed_norm = np.clip(bottle_speed_norm, 0.0, 2.0)
    #
    #         main_speeds = [
    #             traci.edge.getLastStepMeanSpeed(e)
    #             for e in self.MAINLINE_EDGES
    #         ]
    #         mainline_speed = float(np.mean(main_speeds))
    #         mainline_speed_norm = mainline_speed / 30.0
    #         mainline_speed_norm = np.clip(mainline_speed_norm, 0.0, 2.0)
    #
    #         # 瓶颈速度权重更大
    #         speed_reward = 2.5 * bottle_speed_norm + 1.0 * mainline_speed_norm
    #
    #         # ===== 2. 吞吐量奖励 =====
    #         throughput = 0.0
    #         for e in self.MAINLINE_EDGES:
    #             speed = traci.edge.getLastStepMeanSpeed(e)
    #             vehs = traci.edge.getLastStepVehicleNumber(e)
    #             throughput += speed * vehs
    #
    #         # 大致归一化到 [0, 3] 左右
    #         throughput_reward = throughput / 1000.0
    #
    #         # ===== 3. 匝道排队惩罚 =====
    #         ramp_queue = sum(
    #             traci.edge.getLastStepHaltingNumber(e)
    #             for e in self.RAMP_EDGES
    #         )
    #         queue_ratio = ramp_queue / self.MAX_RAMP_QUEUE
    #         queue_ratio = np.clip(queue_ratio, 0.0, 2.0)
    #         queue_penalty = 2.0 * queue_ratio  # 权重 2
    #
    #         # ===== 4. 拥堵惩罚 =====
    #         bottle_occ = traci.edge.getLastStepOccupancy(self.BOTTLENECK_EDGE)
    #         congestion_penalty = 0.0
    #         if bottle_speed < 6.0 and bottle_occ > 0.35:
    #             congestion_penalty = 2.0
    #         elif bottle_speed < 10.0 and bottle_occ > 0.30:
    #             congestion_penalty = 1.0
    #
    #         # ===== 5. 动作平滑性惩罚 =====
    #         rate_change = abs(current_rate - prev_rate) / self.MAX_RATE
    #         smooth_penalty = 0.5 * max(0.0, rate_change - 0.2)  # 超过 20% 的变化才惩罚
    #
    #         # ===== 6. 安全软惩罚（不改动作，只加惩罚）=====
    #         soft_safety_penalty = 0.0
    #         if ramp_queue > 0.9 * self.MAX_RAMP_QUEUE:
    #             soft_safety_penalty += 2.0
    #         if bottle_occ > 0.5:
    #             soft_safety_penalty += 2.0
    #
    #         # ===== 7. 组合 =====
    #         reward = (
    #             speed_reward +
    #             throughput_reward -
    #             queue_penalty -
    #             congestion_penalty -
    #             smooth_penalty -
    #             soft_safety_penalty
    #         )
    #
    #     except Exception as e:
    #         print(f"Warning: Error calculating reward: {e}")
    #         reward = 0.0
    #
    #     return float(reward)

    # -----------------------------
    # 指标记录
    # -----------------------------
    def _calculate_reward(self, current_rate: float, prev_rate: float) -> float:
        """极简reward - 只优化TTS"""
        reward = 0.0

        try:
            # 计算所有车辆的travel time
            total_tts = 0.0

            # 主线车辆
            for edge_id in self.MAINLINE_EDGES:
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    # 等待时间 + 行驶时间代理
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    speed = traci.vehicle.getSpeed(veh_id)
                    # 低速行驶也算延误
                    if speed < 20.0:  # 正常速度阈值
                        total_tts += wait_time + (20.0 - speed) * 0.1
                    else:
                        total_tts += wait_time

            # 匝道车辆 (权重更高,避免PPO故意堵匝道)
            for edge_id in self.RAMP_EDGES:
                for veh_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    total_tts += wait_time * 1.5  # 匝道延误权重1.5倍

            # Reward = 负TTS (越小越好)
            reward = -total_tts / 100.0  # 归一化

            # 可选: 轻微的平滑性惩罚
            rate_change = abs(current_rate - prev_rate) / self.MAX_RATE
            if rate_change > 0.3:
                reward -= 0.1

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            reward = 0.0

        return float(reward)

    def _record_metrics(self, reward: float, rate: float):
        """记录详细指标"""
        try:
            self.episode_metrics['reward'].append(reward)
            self.episode_metrics['metering_rate'].append(rate)

            # 速度指标
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e)
                for e in self.MAINLINE_EDGES
            ])
            self.episode_metrics['mainline_speed'].append(mainline_speed)

            merge_speed = traci.edge.getLastStepMeanSpeed(self.MERGE_EDGE)
            self.episode_metrics['merge_speed'].append(merge_speed)

            bottle_speed = traci.edge.getLastStepMeanSpeed(self.BOTTLENECK_EDGE)
            self.episode_metrics['bottleneck_speed'].append(bottle_speed)

            # 排队
            ramp_queue = sum(
                traci.edge.getLastStepHaltingNumber(e)
                for e in self.RAMP_EDGES
            )
            self.episode_metrics['ramp_queue'].append(ramp_queue)

            # 吞吐量
            throughput = sum(
                traci.edge.getLastStepMeanSpeed(e) *
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            self.episode_metrics['throughput'].append(throughput)

            # 占有率
            occ = np.mean([
                traci.edge.getLastStepOccupancy(e)
                for e in self.MAINLINE_EDGES
            ])
            self.episode_metrics['occupancy'].append(occ)

            # 瓶颈流量（车辆数）
            bottle_flow = traci.edge.getLastStepVehicleNumber(self.BOTTLENECK_EDGE)
            self.episode_metrics['bottleneck_flow'].append(bottle_flow)

            # 总延误（所有等待车辆的等待时间总和）
            total_delay = 0.0
            for e in self.MAINLINE_EDGES + self.RAMP_EDGES:
                for veh_id in traci.edge.getLastStepVehicleIDs(e):
                    total_delay += traci.vehicle.getAccumulatedWaitingTime(veh_id)
            self.episode_metrics['total_delay'].append(total_delay)

        except Exception as e:
            print(f"Warning: Error recording metrics: {e}")

    def _get_episode_summary(self) -> Dict[str, Any]:
        """生成episode摘要"""
        summary = {}

        for key, values in self.episode_metrics.items():
            if len(values) > 0:
                summary[f'avg_{key}'] = float(np.mean(values))
                summary[f'std_{key}'] = float(np.std(values))
                summary[f'max_{key}'] = float(np.max(values))
                summary[f'min_{key}'] = float(np.min(values))

        summary['total_reward'] = float(np.sum(self.episode_metrics['reward']))
        summary['episode_steps'] = self.current_step
        summary['episode_number'] = self.episode_count

        # 计算总TTS（veh·s）
        if len(self.episode_metrics['total_delay']) > 0:
            summary['total_tts'] = float(np.sum(self.episode_metrics['total_delay']))

        if self.save_metrics:
            filename = os.path.join(
                self.metrics_dir,
                f"episode_{self.episode_count:04d}.json"
            )
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)

        return summary

    def _print_episode_summary(self, info: Dict):
        """打印摘要"""
        print(f"\n{'=' * 70}")
        print(f"Episode {self.episode_count} Summary")
        print(f"{'=' * 70}")
        print(f"  Avg Reward:         {info.get('avg_reward', 0):8.3f} (Total: {info.get('total_reward', 0):.2f})")
        print(
            f"  Bottleneck Speed:   {info.get('avg_bottleneck_speed', 0):8.2f} m/s (±{info.get('std_bottleneck_speed', 0):.2f})")
        print(f"  Mainline Speed:     {info.get('avg_mainline_speed', 0):8.2f} m/s")
        print(f"  Merge Speed:        {info.get('avg_merge_speed', 0):8.2f} m/s")
        print(f"  Ramp Queue:         {info.get('avg_ramp_queue', 0):8.2f} veh")
        print(f"  Throughput:         {info.get('avg_throughput', 0):8.0f}")
        print(f"  Metering Rate:      {info.get('avg_metering_rate', 0):8.1f} veh/h")
        if 'total_tts' in info:
            print(f"  Total TTS:          {info.get('total_tts', 0):8.1f} veh·s")
        print(f"{'=' * 70}\n")


# =====================================================
# 回调函数
# =====================================================

class DetailedMetricsCallback(BaseCallback):
    """增强的训练监控回调"""

    def __init__(self, save_freq=500, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.metrics = {
            'steps': [],
            'avg_reward': [],
            'avg_ep_length': [],
            'success_rate': []
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and len(self.model.ep_info_buffer) > 0:

            # 提取最近episodes的信息
            recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            recent_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

            avg_reward = float(np.mean(recent_rewards))
            avg_length = float(np.mean(recent_lengths))

            # 成功率：定义为平均奖励 > 0 的比例
            success_rate = float(np.mean([1 if r > 0 else 0 for r in recent_rewards]))

            self.metrics['steps'].append(self.n_calls)
            self.metrics['avg_reward'].append(avg_reward)
            self.metrics['avg_ep_length'].append(avg_length)
            self.metrics['success_rate'].append(success_rate)

            # 保存
            with open(f'{self.save_path}/training_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)

            if self.verbose:
                print(f"\n[Step {self.n_calls:,}]")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Success Rate: {success_rate:.1%}")

        return True


# =====================================================
# 训练函数
# =====================================================

def train_ppo(
        sumo_cfg: str = "congestion.sumocfg",
        total_timesteps: int = 200_000,
        n_envs: int = 4,
        model_save_path: str = "./models/ppo_ramp_metering_v2",
        log_path: str = "./logs"
):
    """训练 PPO Ramp Metering Agent"""

    print(f"\n{'=' * 70}")
    print("Training Improved PPO Ramp Metering Agent (稳定版)")
    print(f"{'=' * 70}")
    print(f"SUMO Config:       {sumo_cfg}")
    print(f"Total Timesteps:   {total_timesteps:,}")
    print(f"Parallel Envs:     {n_envs}")
    print(f"Model Save Path:   {model_save_path}")
    print(f"Control Frequency: 5 seconds (delta_time)")
    print(f"Episode Length:    {720} steps")
    print(f"{'=' * 70}\n")

    def make_env(rank):
        def _init():
            env = RampMeteringEnv(
                sumo_cfg=sumo_cfg,
                gui=False,
                seed=1000 + rank,
                save_metrics=False,
                max_steps=720,  # 3600s / 5s
                delta_time=5
            )
            return Monitor(env)

        return _init

    # Windows下用DummyVecEnv
    if n_envs > 1 and os.name != "nt":
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # PPO 参数
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
        clip_range_vf=None,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callback = DetailedMetricsCallback(
        save_freq=500,
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


# =====================================================
# 评估函数 - 支持多种策略对比
# =====================================================

def evaluate_policy(
        model_path: str = None,
        sumo_cfg: str = "congestion.sumocfg",
        n_episodes: int = 10,
        gui: bool = False,
        policy_type: str = "ppo"  # "ppo", "fixed", "no_control"
):
    """
    评估策略

    policy_type:
    - "ppo": 使用训练好的PPO模型
    - "fixed": 固定放行率（600 veh/h）
    - "no_control": 不控制（永久绿灯）
    """

    print(f"\n{'=' * 70}")
    print(f"Evaluating Policy: {policy_type.upper()}")
    print(f"{'=' * 70}")
    print(f"Episodes:  {n_episodes}")
    print(f"GUI:       {gui}")
    print(f"{'=' * 70}\n")

    # 加载PPO模型（如果需要）
    model = None
    if policy_type == "ppo":
        if model_path is None:
            model_path = "./models/ppo_ramp_metering_v2"
        model = PPO.load(model_path)
        print(f"Loaded model: {model_path}.zip\n")

    # 创建环境
    env = RampMeteringEnv(
        sumo_cfg=sumo_cfg,
        gui=gui,
        seed=2000,
        save_metrics=True,
        metrics_dir=f"./evaluation_metrics_{policy_type}",
        max_steps=720,
        delta_time=5
    )

    all_summaries = []

    for episode in range(n_episodes):
        print(f"Running episode {episode + 1}/{n_episodes}...")
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            # 根据策略类型选择动作
            if policy_type == "ppo":
                action, _states = model.predict(obs, deterministic=True)
            elif policy_type == "fixed":
                # 固定600 veh/h
                action = np.array([(600.0 - env.MIN_RATE) / (env.MAX_RATE - env.MIN_RATE)])
            elif policy_type == "no_control":
                # 最大放行率（相当于不控制）
                action = np.array([1.0])
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        if info:
            all_summaries.append(info)
            print(f"  → Reward: {info.get('total_reward', 0):.2f}, "
                  f"Bottleneck Speed: {info.get('avg_bottleneck_speed', 0):.2f} m/s")

    env.close()

    # 统计结果
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS - {policy_type.upper()}")
    print(f"{'=' * 70}")

    metrics = [
        ('avg_bottleneck_speed', 'Bottleneck Speed (m/s)'),
        ('avg_mainline_speed', 'Mainline Speed (m/s)'),
        ('avg_ramp_queue', 'Ramp Queue (veh)'),
        ('avg_throughput', 'Throughput'),
        ('total_reward', 'Total Reward'),
        ('total_tts', 'Total TTS (veh·s)')
    ]

    results = {}
    for metric_key, metric_name in metrics:
        values = [s.get(metric_key, 0) for s in all_summaries]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        results[metric_key] = {'mean': mean_val, 'std': std_val}
        print(f"  {metric_name:30s}: {mean_val:8.2f} ± {std_val:6.2f}")

    print(f"{'=' * 70}\n")

    # 保存结果
    output_file = f"evaluation_results_{policy_type}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'policy_type': policy_type,
            'n_episodes': n_episodes,
            'summaries': all_summaries,
            'statistics': results
        }, f, indent=2)

    print(f"✅ Results saved to {output_file}\n")

    return results


def compare_policies(
        sumo_cfg: str = "congestion.sumocfg",
        n_episodes: int = 10,
        model_path: str = "./models/ppo_ramp_metering_v2"
):
    """对比不同策略的性能"""

    print("\n" + "=" * 70)
    print("COMPARING RAMP METERING POLICIES")
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

    # 对比表格
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'No Control':>15} {'Fixed (600)':>15} {'PPO':>15}")
    print("-" * 70)

    metrics = [
        'avg_bottleneck_speed',
        'avg_mainline_speed',
        'avg_ramp_queue',
        'total_reward',
        'total_tts'
    ]

    for metric in metrics:
        row = f"{metric:<30}"
        for policy in policies:
            val = all_results[policy][metric]['mean']
            row += f"{val:>15.2f}"
        print(row)

    print("=" * 70 + "\n")

    # 保存对比结果
    with open("policy_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✅ Comparison saved to policy_comparison.json\n")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Improved PPO Ramp Metering (稳定版)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--compare", action="store_true", help="Compare all policies")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--n-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--policy", type=str, default="ppo",
                        choices=["ppo", "fixed", "no_control"],
                        help="Policy type for evaluation")

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
        compare_policies(
            n_episodes=args.n_episodes
        )
    else:
        print("Please specify --train, --eval, or --compare")
        parser.print_help()
