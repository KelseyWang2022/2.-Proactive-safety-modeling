#!/usr/bin/env python3
"""
多Ramp DDPG控制 - 稳定版

特点：
1. 使用 DDPG 控制三个匝道信号（meter1、meter2、meter3）
2. 奖励函数采用简化稳定版本：瓶颈速度奖励 + 轻微 ramp 排队惩罚
3. 控制周期 10s，episode 时长 3600s（360 步）
4. 提供：
   - train_ddpg() 训练函数
   - evaluate_policy() 评估 DDPG / fixed / no_control
   - compare_policies() 对比三种策略
   - plot_episode_series() 画 reward 各项曲线
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List

import gymnasium as gym
from gymnasium import spaces

# ============ SUMO / TRACI ============
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"  # 根据你本机情况修改

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ============ RL 库 ============
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor


# =====================================================
# 多Ramp环境类（与 PPO 稳定版保持一致）
# =====================================================

class MultiRampEnv(gym.Env):
    """
    多Ramp Metering环境 (3个ramp)

    观测空间（26维）:
    [0-6]:   主线7段速度 / FREE_FLOW_SPEED  (main_0 ~ main_6)
    [7-13]:  主线7段占有率 (main_0 ~ main_6)
    [14-17]: Ramp1状态 [排队/30, 等待/300, 上一步rate, 流入/1500]
    [18-21]: Ramp2状态 [排队/30, 等待/300, 上一步rate, 流入/1500]
    [22-25]: Ramp3状态 [排队/30, 等待/300, 上一步rate, 流入/1500]

    动作空间（3维）:
    - [rate1, rate2, rate3]: 每个ramp的放行率 [0,1] -> [MIN_RATE, MAX_RATE]
    """

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            sumo_cfg: str,
            max_steps: int = 360,   # 3600秒 / 10秒 = 360步
            delta_time: int = 10,   # 控制间隔10秒
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

        # ===== SUMO 网络配置 =====
        self.MAINLINE_EDGES = ['main_0', 'main_1', 'main_2', 'main_3',
                               'main_4', 'main_5', 'main_6']
        self.RAMP_EDGES = [
            ['ramp1_0', 'ramp1_1'],  # Ramp1
            ['ramp2_0', 'ramp2_1'],  # Ramp2
            ['ramp3_0', 'ramp3_1']   # Ramp3
        ]
        self.METER_IDS = ['meter1', 'meter2', 'meter3']

        # 关键路段：合流区 & 瓶颈区
        self.MERGE_EDGES = ['main_1', 'main_3', 'main_5']
        self.BOTTLENECK_EDGES = ['main_2', 'main_4', 'main_6']

        # 物理与容量参数
        self.FREE_FLOW_SPEED = 30.0   # m/s, 约108 km/h
        self.MAX_RAMP_QUEUE = 30      # veh
        self.MAX_RAMP_WAIT = 300.0    # s
        self.MIN_RATE = 200.0         # veh/h
        self.MAX_RATE = 1200.0        # veh/h
        self.MAX_RAMP_INFLOW = 1500.0 # veh/h

        # ===== 观测 & 动作空间 =====
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
        self.last_rates = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # 上一步放行率（归一化）

        # 历史信息
        self.ramp_inflows = [0.0, 0.0, 0.0]

        # 指标记录（包括 reward 分解）
        self.episode_metrics = {
            'reward': [],
            'speed_term': [],
            'queue_term': [],
            'bottleneck_speed': [],
            'total_ramp_queue': [],
            'mainline_speed': [],
            'ramp_queues': [[], [], []],
            'throughput': [],
            'metering_rates': [[], [], []],
            'total_delay': []
        }

    # -----------------------------
    # Gym接口
    # -----------------------------

    def reset(self, seed=None, options=None):
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
        self.last_rates = np.array([0.5, 0.5, 0.5], dtype=np.float32)
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
        # 1) 动作裁剪
        action = np.clip(action, 0.0, 1.0)

        # 2) 连续动作 -> 真实匝道放行率
        rates = [self.MIN_RATE + float(a) * (self.MAX_RATE - self.MIN_RATE)
                 for a in action]

        # 3) 温和安全约束
        safe_rates = [self._safety_filter(rates[i], i) for i in range(3)]

        # 4) 应用到 meter
        for i, rate in enumerate(safe_rates):
            self._apply_metering(rate, i)

        # 5) SUMO 仿真前进 delta_time 秒
        for _ in range(self.delta_time):
            traci.simulationStep()

        # 6) 更新 ramp inflow
        self._update_inflows()

        # 7) 新观测
        obs = self._get_observation()

        # 8) 奖励
        reward, rinfo = self._calculate_reward(safe_rates)

        # 9) 结束判断
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 10) 记录指标
        self._record_metrics(reward, safe_rates, rinfo)

        # 11) 更新 last_rates
        self.last_rates = np.clip(action.copy(), 0.0, 1.0)

        info = {}
        if terminated:
            info = self._get_episode_summary()
            # 这里不 print，避免 DDPG 太啰嗦；可以按需打开

        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
            self.sumo_running = False

    # -----------------------------
    # 观测相关
    # -----------------------------

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(26, dtype=np.float32)

        try:
            # 主线速度
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                speed = traci.edge.getLastStepMeanSpeed(edge_id)
                v_norm = speed / self.FREE_FLOW_SPEED if self.FREE_FLOW_SPEED > 0 else 0.0
                obs[i] = float(np.clip(v_norm, 0.0, 1.0))

            # 主线占有率
            for i, edge_id in enumerate(self.MAINLINE_EDGES):
                occ = traci.edge.getLastStepOccupancy(edge_id) / 100.0
                obs[7 + i] = float(np.clip(occ, 0.0, 1.0))

            # 各 ramp 状态（排队、等待、上一步rate、流入）
            for ramp_idx in range(3):
                base = 14 + 4 * ramp_idx
                obs[base: base + 4] = self._get_ramp_state(ramp_idx)

        except Exception as e:
            print(f"Warning: Error getting observation: {e}")

        return obs

    def _get_ramp_state(self, ramp_idx: int) -> np.ndarray:
        state = np.zeros(4, dtype=np.float32)

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]

            # 排队长度
            queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)
            state[0] = float(np.clip(queue / self.MAX_RAMP_QUEUE, 0.0, 1.0))

            # 平均等待时间
            wait_times = []
            for e in ramp_edges:
                for veh_id in traci.edge.getLastStepVehicleIDs(e):
                    wait_times.append(traci.vehicle.getAccumulatedWaitingTime(veh_id))
            avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
            state[1] = float(np.clip(avg_wait / self.MAX_RAMP_WAIT, 0.0, 1.0))

            # 上一步放行率（0-1）
            state[2] = float(self.last_rates[ramp_idx])

            # ramp 流入
            inflow = self.ramp_inflows[ramp_idx]
            state[3] = float(np.clip(inflow / self.MAX_RAMP_INFLOW, 0.0, 1.0))

        except Exception as e:
            print(f"Warning: Error getting ramp state: {e}")

        return state

    # -----------------------------
    # 控制相关
    # -----------------------------

    def _update_inflows(self):
        try:
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                vehs = sum(traci.edge.getLastStepVehicleNumber(e) for e in ramp_edges)
                self.ramp_inflows[i] = (vehs / self.delta_time) * 3600
        except Exception as e:
            print(f"Warning: Error updating inflows: {e}")

    def _safety_filter(self, rate: float, ramp_idx: int) -> float:
        """
        简化安全约束：只在队列 >90% 上限时，往 MAX_RATE 稍微推一点。
        不根据下游占有率硬剪，尽量不破坏 RL 学习。
        """
        rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))

        try:
            ramp_edges = self.RAMP_EDGES[ramp_idx]
            queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)

            if queue > 0.9 * self.MAX_RAMP_QUEUE:
                alpha = min((queue - 0.9 * self.MAX_RAMP_QUEUE) / (0.1 * self.MAX_RAMP_QUEUE), 1.0)
                rate = (1 - 0.3 * alpha) * rate + 0.3 * alpha * self.MAX_RATE

        except Exception as e:
            print(f"Warning: Error in safety_filter: {e}")

        return float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))

    def _apply_metering(self, rate: float, ramp_idx: int):
        try:
            rate = float(np.clip(rate, self.MIN_RATE, self.MAX_RATE))
            meter_id = self.METER_IDS[ramp_idx]

            headway = 3600.0 / rate  # s / veh
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
            print(f"Warning: Error applying metering: {e}")

    # -----------------------------
    # 奖励函数（稳定可学习）⭐
    # -----------------------------

    def _calculate_reward(self, rates: List[float]) -> (float, Dict[str, float]):
        """
        reward = speed_term - 0.1 * queue_term

        speed_term = avg_bottleneck_speed / FREE_FLOW_SPEED  (截断到[0,1.5])
        queue_term = total_ramp_queue / (3 * MAX_RAMP_QUEUE) (截断到[0,2])
        """
        reward = 0.0
        info = {
            "speed_term": 0.0,
            "queue_term": 0.0,
            "bottleneck_speed": 0.0,
            "total_ramp_queue": 0.0,
        }

        try:
            # 瓶颈速度
            bottleneck_speeds = [
                traci.edge.getLastStepMeanSpeed(e) for e in self.BOTTLENECK_EDGES
            ]
            avg_bottleneck_speed = float(np.mean(bottleneck_speeds)) if bottleneck_speeds else 0.0
            speed_term = avg_bottleneck_speed / self.FREE_FLOW_SPEED if self.FREE_FLOW_SPEED > 0 else 0.0
            speed_term = float(np.clip(speed_term, 0.0, 1.5))

            # 总 ramp 排队
            total_queue = 0.0
            for ramp_edges in self.RAMP_EDGES:
                q = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)
                total_queue += q
            queue_term = total_queue / (3.0 * self.MAX_RAMP_QUEUE) if self.MAX_RAMP_QUEUE > 0 else 0.0
            queue_term = float(np.clip(queue_term, 0.0, 2.0))

            reward = speed_term - 0.1 * queue_term

            info["speed_term"] = speed_term
            info["queue_term"] = queue_term
            info["bottleneck_speed"] = avg_bottleneck_speed
            info["total_ramp_queue"] = total_queue

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            reward = 0.0

        return float(reward), info

    # -----------------------------
    # 指标记录
    # -----------------------------

    def _record_metrics(self, reward: float, rates: List[float], rinfo: Dict[str, float]):
        try:
            self.episode_metrics['reward'].append(float(reward))
            self.episode_metrics['speed_term'].append(float(rinfo.get("speed_term", 0.0)))
            self.episode_metrics['queue_term'].append(float(rinfo.get("queue_term", 0.0)))
            self.episode_metrics['bottleneck_speed'].append(float(rinfo.get("bottleneck_speed", 0.0)))
            self.episode_metrics['total_ramp_queue'].append(float(rinfo.get("total_ramp_queue", 0.0)))

            # 主线平均速度
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e) for e in self.MAINLINE_EDGES
            ])
            self.episode_metrics['mainline_speed'].append(float(mainline_speed))

            # 各 ramp 排队
            for i, ramp_edges in enumerate(self.RAMP_EDGES):
                q = sum(traci.edge.getLastStepHaltingNumber(e) for e in ramp_edges)
                self.episode_metrics['ramp_queues'][i].append(float(q))

            # 吞吐量，记录但不进 reward
            throughput = sum(
                traci.edge.getLastStepMeanSpeed(e) *
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            self.episode_metrics['throughput'].append(float(throughput))

            # 放行率
            for i, rate in enumerate(rates):
                self.episode_metrics['metering_rates'][i].append(float(rate))

            # 总延误
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
        summary = {}

        summary['total_reward'] = float(np.sum(self.episode_metrics['reward'])) if self.episode_metrics['reward'] else 0.0
        summary['avg_reward'] = float(np.mean(self.episode_metrics['reward'])) if self.episode_metrics['reward'] else 0.0
        summary['avg_mainline_speed'] = float(np.mean(self.episode_metrics['mainline_speed'])) if self.episode_metrics['mainline_speed'] else 0.0
        summary['avg_throughput'] = float(np.mean(self.episode_metrics['throughput'])) if self.episode_metrics['throughput'] else 0.0

        for i in range(3):
            q_list = self.episode_metrics['ramp_queues'][i]
            r_list = self.episode_metrics['metering_rates'][i]
            summary[f'avg_ramp{i + 1}_queue'] = float(np.mean(q_list)) if q_list else 0.0
            summary[f'avg_ramp{i + 1}_rate'] = float(np.mean(r_list)) if r_list else 0.0

        if len(self.episode_metrics['total_delay']) > 0:
            summary['total_delay'] = float(np.sum(self.episode_metrics['total_delay']))

        summary['episode_steps'] = self.current_step
        summary['episode_number'] = self.episode_count

        # 保存 summary + series
        if self.save_metrics:
            os.makedirs(self.metrics_dir, exist_ok=True)

            summary_file = os.path.join(self.metrics_dir, f"episode_{self.episode_count:04d}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

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
            series_file = os.path.join(self.metrics_dir, f"episode_{self.episode_count:04d}_series.json")
            with open(series_file, 'w') as f:
                json.dump(series, f, indent=2)

        return summary


# =====================================================
# DDPG 训练
# =====================================================

def train_ddpg(
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        total_timesteps: int = 3000,
        model_save_path: str = "./models/ddpg_multi_ramp",
        log_path: str = "./logs_ddpg"
):
    """
    训练 DDPG 模型
    """
    print("\n" + "=" * 70)
    print("Training DDPG Multi-Ramp Metering")
    print("=" * 70)
    print(f"Config:         {sumo_cfg}")
    print(f"Timesteps:      {total_timesteps:,}")
    print(f"Control Freq:   10 seconds")
    print(f"Episode Length: 3600 seconds (360 steps)")
    print("=" * 70 + "\n")

    env = MultiRampEnv(
        sumo_cfg=sumo_cfg,
        gui=False,
        seed=1234,
        save_metrics=False,
        max_steps=360,
        delta_time=10
    )
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    # 正态噪声，探索用
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.15 * np.ones(n_actions)
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model.learn(total_timesteps=total_timesteps, log_interval=50)

    model.save(model_save_path)
    print(f"\n✅ DDPG Model saved to {model_save_path}.zip\n")

    env.close()
    return model


# =====================================================
# 评估 + 对比
# =====================================================

def evaluate_policy(
        model_path: str = None,
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        n_episodes: int = 5,
        gui: bool = False,
        policy_type: str = "ddpg"
):
    """
    policy_type:
      - "ddpg": 训练好的 DDPG 策略
      - "fixed": 固定放行率(600 veh/h)
      - "no_control": 最大放行
    """

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {policy_type.upper()}")
    print(f"{'=' * 70}\n")

    model = None
    if policy_type == "ddpg":
        if model_path is None:
            model_path = "./models/ddpg_multi_ramp"
        model = DDPG.load(model_path)
        print(f"Loaded model: {model_path}.zip\n")

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

    for ep in range(n_episodes):
        print(f"Episode {ep + 1}/{n_episodes}...")
        obs, info = env.reset()
        done = False

        while not done:
            if policy_type == "ddpg":
                action, _ = model.predict(obs, deterministic=True)
            elif policy_type == "fixed":
                # 固定600 veh/h
                fixed_rate = (600 - env.MIN_RATE) / (env.MAX_RATE - env.MIN_RATE)
                action = np.array([fixed_rate, fixed_rate, fixed_rate], dtype=np.float32)
            elif policy_type == "no_control":
                # 最大放行
                action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown policy: {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info:
            all_summaries.append(info)
            print(f"  Total Reward: {info.get('total_reward', 0):.2f}, "
                  f"Avg Speed: {info.get('avg_mainline_speed', 0):.2f} m/s\n")

    env.close()

    # 聚合结果
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
        ('total_delay', 'Total Delay (veh·s)')
    ]

    results = {}
    for key, name in metrics:
        vals = [s.get(key, 0) for s in all_summaries]
        mean_val = float(np.mean(vals)) if vals else 0.0
        std_val = float(np.std(vals)) if vals else 0.0
        results[key] = {"mean": mean_val, "std": std_val}
        print(f"  {name:25s}: {mean_val:8.2f} ± {std_val:6.2f}")

    print(f"{'=' * 70}\n")

    out_file = f"eval_results_{policy_type}.json"
    with open(out_file, "w") as f:
        json.dump({
            "policy": policy_type,
            "n_episodes": n_episodes,
            "results": results,
            "summaries": all_summaries
        }, f, indent=2)

    print(f"✅ Results saved to {out_file}\n")
    return results


def compare_policies(
        sumo_cfg: str = "multi_ramp_scenario.sumocfg",
        n_episodes: int = 5,
        model_path: str = "./models/ddpg_multi_ramp"
):
    """
    对比：no_control, fixed, ddpg
    """

    print("\n" + "=" * 70)
    print("COMPARING POLICIES")
    print("=" * 70 + "\n")

    policies = ["no_control", "fixed", "ddpg"]
    all_results = {}

    for p in policies:
        res = evaluate_policy(
            model_path=model_path,
            sumo_cfg=sumo_cfg,
            n_episodes=n_episodes,
            gui=False,
            policy_type=p
        )
        all_results[p] = res

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} {'No Control':>15} {'Fixed':>15} {'DDPG':>15}")
    print("-" * 70)

    metrics = [
        "avg_mainline_speed",
        "avg_throughput",
        "total_reward",
        "avg_ramp1_queue",
    ]

    for m in metrics:
        row = f"{m:<25}"
        for p in policies:
            val = all_results[p][m]["mean"]
            row += f"{val:>15.2f}"
        print(row)

    print("=" * 70 + "\n")

    with open("comparison_ddpg.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("✅ Comparison saved to comparison_ddpg.json\n")


# =====================================================
# 画 reward 各项曲线
# =====================================================

def plot_episode_series(
        metrics_dir: str = "./eval_ddpg",
        episode: int = None,
        output_path: str = None
):
    """
    从 metrics_dir 中读取 episode_XXXX_series.json，画
    reward / speed_term / queue_term 随步数变化曲线。
    """
    import glob
    import matplotlib.pyplot as plt

    pattern = os.path.join(metrics_dir, "episode_*_series.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No series files found in {metrics_dir}")
        return

    if episode is None:
        series_file = files[-1]
    else:
        series_file = os.path.join(metrics_dir, f"episode_{episode:04d}_series.json")
        if not os.path.exists(series_file):
            print(f"{series_file} not found, fallback to last episode.")
            series_file = files[-1]

    print(f"Loading series from: {series_file}")
    with open(series_file, "r") as f:
        series = json.load(f)

    steps = list(range(len(series["reward"])))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, series["reward"], label="reward")
    plt.plot(steps, series["speed_term"], label="speed_term")
    plt.plot(steps, series["queue_term"], label="queue_term")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Reward and Components over Steps")
    plt.legend()
    plt.grid(True)

    if output_path is None:
        output_path = os.path.join(metrics_dir, "reward_components_ddpg.png")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Reward component curves saved to {output_path}")
    plt.close()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Ramp DDPG Control")
    parser.add_argument("--train", action="store_true", help="Train DDPG model")
    parser.add_argument("--eval", action="store_true", help="Evaluate policy")
    parser.add_argument("--compare", action="store_true", help="Compare ddpg/fixed/no_control")
    parser.add_argument("--plot", action="store_true", help="Plot reward components")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI during eval")
    parser.add_argument("--timesteps", type=int, default=3000, help="Training timesteps")
    parser.add_argument("--n-episodes", type=int, default=5, help="Eval episodes")
    parser.add_argument("--policy", type=str, default="ddpg",
                        choices=["ddpg", "fixed", "no_control"])
    parser.add_argument("--metrics-dir", type=str, default="./eval_ddpg", help="Metrics dir for plotting")
    parser.add_argument("--episode", type=int, default=None, help="Which episode to plot")

    args = parser.parse_args()

    if args.train:
        train_ddpg(
            sumo_cfg="multi_ramp_scenario.sumocfg",
            total_timesteps=args.timesteps
        )
    elif args.eval:
        evaluate_policy(
            model_path="./models/ddpg_multi_ramp",
            sumo_cfg="multi_ramp_scenario.sumocfg",
            n_episodes=args.n_episodes,
            gui=args.gui,
            policy_type=args.policy
        )
    elif args.compare:
        compare_policies(
            sumo_cfg="multi_ramp_scenario.sumocfg",
            n_episodes=args.n_episodes,
            model_path="./models/ddpg_multi_ramp"
        )
    elif args.plot:
        plot_episode_series(
            metrics_dir=args.metrics_dir,
            episode=args.episode
        )
    else:
        print("Please specify --train, --eval, --compare or --plot")
        parser.print_help()
