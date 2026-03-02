import os
import sys
import time
import numpy as np
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# SUMO
def _check_sumo():
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "找不到环境变量 SUMO_HOME。请先设置 SUMO_HOME 指向你的 SUMO 安装目录。"
        )
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

_check_sumo()
import traci
import sumolib


@dataclass
class EnvConfig:
    sumocfg_path: str = "Ramp-metering-Project-main/config/highway.sumocfg"
    use_gui: bool = False

    # 控制周期（秒）：PPO 每一步决策控制一个周期
    control_interval: int = 10

    # episode 最大仿真时长（秒）
    max_sim_time: int = 3600

    # reward 权重
    w_speed: float = 1.0       # 主线速度奖励
    w_queue: float = 0.4       # 匝道排队惩罚
    w_throughput: float = 0.05 # 通行量奖励（可选）

    # 你这个网络的关键对象（从 net.xml 读出来的 id）
    tls_id: str = "meter"
    main_edge: str = "main_2"   # merge 后主线关键路段
    ramp_in_edge: str = "ramp_0"
    ramp_out_edge: str = "ramp_1"


class SumoRampMeteringEnv(gym.Env):
    """
    PPO 用的 SUMO ramp metering 环境（on-policy）。

    动作（连续）:
        a in [0,1] -> 本控制周期内绿灯比例（green ratio）
    观测（连续向量）:
        [main_2: veh_count, mean_speed, halting_count,
         ramp_0: halting_count,
         ramp_1: veh_count]
        然后做简单归一化
    奖励:
        + w_speed * main_speed_norm
        - w_queue * ramp_queue_norm
        + w_throughput * throughput_norm
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        # 动作：绿灯比例 [0,1]
        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32),
                                       dtype=np.float32)

        # 观测：5维连续
        # 我们先用宽松的 Box，实际训练里会归一化到 [0,1] 左右
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(5,), dtype=np.float32)

        self.sim_time = 0
        self._sumo_running = False

        # 用于 throughput 统计
        self._prev_main_veh_total = 0

    def _start_sumo(self):
        if self._sumo_running:
            return

        sumo_binary = "sumo-gui" if self.cfg.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.cfg.sumocfg_path,
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
        ]
        traci.start(sumo_cmd)
        self._sumo_running = True
        self.sim_time = 0

        # 初始化：把匝道灯先设为红（phase 0）
        try:
            traci.trafficlight.setPhase(self.cfg.tls_id, 0)
        except traci.TraCIException as e:
            raise RuntimeError(f"找不到 tls_id={self.cfg.tls_id}。请检查 net.xml。原始错误: {e}")

        self._prev_main_veh_total = traci.edge.getLastStepVehicleNumber(self.cfg.main_edge)

    def _close_sumo(self):
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 每个 episode 都重启 SUMO（最稳妥）
        self._close_sumo()
        self._start_sumo()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # action: green ratio in [0,1]
        green_ratio = float(np.clip(action[0], 0.0, 1.0))
        interval = self.cfg.control_interval

        green_steps = int(round(green_ratio * interval))
        green_steps = max(0, min(interval, green_steps))
        red_steps = interval - green_steps

        # 在一个控制周期内执行：先绿后红（你也可以反过来）
        self._apply_metering(green_steps=green_steps, red_steps=red_steps)

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = False
        truncated = self.sim_time >= self.cfg.max_sim_time

        info = {
            "sim_time": self.sim_time,
            "green_ratio": green_ratio,
            "green_steps": green_steps,
            "red_steps": red_steps,
        }
        return obs, reward, terminated, truncated, info

    def _apply_metering(self, green_steps: int, red_steps: int):
        # phase 1: G, phase 0: r（你的 net.xml 里就是这么写的）
        # 绿灯放行
        if green_steps > 0:
            traci.trafficlight.setPhase(self.cfg.tls_id, 1)
            for _ in range(green_steps):
                traci.simulationStep()
                self.sim_time += 1

        # 红灯截流
        if red_steps > 0:
            traci.trafficlight.setPhase(self.cfg.tls_id, 0)
            for _ in range(red_steps):
                traci.simulationStep()
                self.sim_time += 1

    def _get_obs(self):
        # 主线关键 edge：main_2
        main_veh = traci.edge.getLastStepVehicleNumber(self.cfg.main_edge)
        main_speed = traci.edge.getLastStepMeanSpeed(self.cfg.main_edge)  # m/s
        main_halt = traci.edge.getLastStepHaltingNumber(self.cfg.main_edge)

        # 匝道排队：用 halting number 作为队列 proxy（速度<0.1m/s 的车辆数）
        ramp0_halt = traci.edge.getLastStepHaltingNumber(self.cfg.ramp_in_edge)
        ramp1_veh = traci.edge.getLastStepVehicleNumber(self.cfg.ramp_out_edge)

        # 简单归一化（避免数值尺度太乱）
        # 速度上限：主线 type speed 30.56m/s（约110km/h）
        speed_norm = np.clip(main_speed / 30.56, 0.0, 1.5)

        # 车辆数量/排队数量归一化：用一个宽松常数（按你路网规模可以再调）
        main_veh_norm = np.clip(main_veh / 50.0, 0.0, 2.0)
        main_halt_norm = np.clip(main_halt / 30.0, 0.0, 2.0)
        ramp0_halt_norm = np.clip(ramp0_halt / 30.0, 0.0, 3.0)
        ramp1_veh_norm = np.clip(ramp1_veh / 30.0, 0.0, 3.0)

        obs = np.array([
            main_veh_norm,
            speed_norm,
            main_halt_norm,
            ramp0_halt_norm,
            ramp1_veh_norm
        ], dtype=np.float32)
        return obs

    def _get_reward(self, obs: np.ndarray) -> float:
        # obs:
        # [main_veh_norm, speed_norm, main_halt_norm, ramp0_halt_norm, ramp1_veh_norm]
        speed_norm = float(obs[1])
        ramp_queue_norm = float(obs[3])  # ramp_0 排队

        # throughput：用 main_edge 车辆数变化做个 proxy（简化）
        main_veh = traci.edge.getLastStepVehicleNumber(self.cfg.main_edge)
        throughput = max(0, main_veh - self._prev_main_veh_total)
        self._prev_main_veh_total = main_veh
        throughput_norm = np.clip(throughput / 5.0, 0.0, 1.0)

        reward = (
            self.cfg.w_speed * speed_norm
            - self.cfg.w_queue * ramp_queue_norm
            + self.cfg.w_throughput * throughput_norm
        )
        return float(reward)

    def close(self):
        self._close_sumo()
        super().close()


def make_env(sumocfg="Ramp-metering-Project-main/config/highway.sumocfg", use_gui=False):
    cfg = EnvConfig(sumocfg_path=sumocfg, use_gui=use_gui)
    env = SumoRampMeteringEnv(cfg)
    env = Monitor(env)  # 记录 episode reward 等
    return env


def train():
    env = DummyVecEnv([lambda: make_env("Ramp-metering-Project-main/config/highway.sumocfg", use_gui=False)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,          # rollout 长度（PPO关键超参）
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./tb_ppo_ramp/",
        device="auto"
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_ramp_metering_sb3")
    env.close()
    print("训练完成，模型已保存：ppo_ramp_metering_sb3.zip")


def evaluate(episodes=3, use_gui=False):
    env = make_env("Ramp-metering-Project-main/config/highway.sumocfg", use_gui=use_gui)
    model = PPO.load("ppo_ramp_metering_sb3", device="auto")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if use_gui:
                time.sleep(0.01)  # GUI 慢一点更好看
            if terminated or truncated:
                break

        print(f"Episode {ep+1}: total_reward={total_reward:.3f}, sim_time={info['sim_time']}")

    env.close()


if __name__ == "__main__":
    # 第一次建议先 train，再 evaluate
    train()
    # evaluate(episodes=3, use_gui=False)
    # evaluate(episodes=1, use_gui=True)  # 想看可视化就开 GUI
