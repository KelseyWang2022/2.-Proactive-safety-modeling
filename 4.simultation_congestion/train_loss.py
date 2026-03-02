# # #!/usr/bin/env python3
# # """
# # 绘制训练曲线
# # """
# # import json
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # def plot_training_curves(json_path='./logs/training_metrics.json'):
# #     """绘制训练曲线"""
# #
# #     # 读取数据
# #     with open(json_path, 'r') as f:
# #         data = json.load(f)
# #
# #     steps = data['steps']
# #     avg_rewards = data['avg_reward']
# #     avg_lengths = data['avg_ep_length']
# #     success_rates = data['success_rate']
# #
# #     # 创建图表
# #     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# #     fig.suptitle('PPO Ramp Metering Training Curves', fontsize=16)
# #
# #     # 1. 平均奖励
# #     axes[0, 0].plot(steps, avg_rewards, linewidth=2, color='blue', alpha=0.7)
# #     axes[0, 0].set_xlabel('Training Steps')
# #     axes[0, 0].set_ylabel('Average Reward')
# #     axes[0, 0].set_title('Average Episode Reward')
# #     axes[0, 0].grid(True, alpha=0.3)
# #     axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero baseline')
# #     axes[0, 0].legend()
# #
# #     # 2. Episode长度
# #     axes[0, 1].plot(steps, avg_lengths, linewidth=2, color='green', alpha=0.7)
# #     axes[0, 1].set_xlabel('Training Steps')
# #     axes[0, 1].set_ylabel('Average Episode Length')
# #     axes[0, 1].set_title('Average Episode Length')
# #     axes[0, 1].grid(True, alpha=0.3)
# #
# #     # 3. 成功率
# #     axes[1, 0].plot(steps, success_rates, linewidth=2, color='orange', alpha=0.7)
# #     axes[1, 0].set_xlabel('Training Steps')
# #     axes[1, 0].set_ylabel('Success Rate')
# #     axes[1, 0].set_title('Success Rate (Reward > 0)')
# #     axes[1, 0].grid(True, alpha=0.3)
# #     axes[1, 0].set_ylim([0, 1])
# #
# #     # 4. 奖励平滑曲线（移动平均）
# #     window = min(10, len(avg_rewards) // 10)
# #     if window > 1:
# #         smoothed_rewards = np.convolve(avg_rewards, np.ones(window) / window, mode='valid')
# #         smoothed_steps = steps[window - 1:]
# #         axes[1, 1].plot(steps, avg_rewards, linewidth=1, color='blue', alpha=0.3, label='Raw')
# #         axes[1, 1].plot(smoothed_steps, smoothed_rewards, linewidth=2, color='blue',
# #                         label=f'Smoothed (window={window})')
# #     else:
# #         axes[1, 1].plot(steps, avg_rewards, linewidth=2, color='blue', label='Reward')
# #
# #     axes[1, 1].set_xlabel('Training Steps')
# #     axes[1, 1].set_ylabel('Average Reward')
# #     axes[1, 1].set_title('Smoothed Reward Curve')
# #     axes[1, 1].grid(True, alpha=0.3)
# #     axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
# #     axes[1, 1].legend()
# #
# #     plt.tight_layout()
# #     plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
# #     print("✅ Training curves saved to: training_curves.png")
# #     plt.show()
# #
# #
# # if __name__ == "__main__":
# #     plot_training_curves()
# import json
# import numpy as np
#
# with open('./logs/training_metrics.json', 'r') as f:
#     data = json.load(f)
#
# rewards = data['avg_reward']
#
# # 分析不同阶段的方差
# early = rewards[:10]    # 前10个记录点
# mid = rewards[10:30]    # 中期
# late = rewards[30:]     # 后期
#
# print(f"训练早期方差: {np.std(early):.2f}")
# print(f"训练中期方差: {np.std(mid):.2f}")
# print(f"训练后期方差: {np.std(late):.2f}")
#
# # 如果后期方差接近0，可能是过拟合
# if np.std(late) < 5:
#     print("⚠️  警告: 后期方差过小，可能过拟合!")
# !/usr/bin/env python3
"""
鲁棒性测试 - 验证PPO是否过拟合
"""

import os
import sys
import numpy as np
import json
from stable_baselines3 import PPO

# 假设您的环境类已经导入
from ppo_ramp_metering_complete_metrics import RampMeteringEnv


def test_robustness(
        model_path="./models/ppo_ramp_metering_v2",
        sumo_cfg="congestion.sumocfg",
        n_seeds=10
):
    """
    测试1: 不同随机种子下的性能
    """
    print(f"\n{'=' * 70}")
    print("鲁棒性测试 1: 不同随机种子")
    print(f"{'=' * 70}\n")

    model = PPO.load(model_path)
    results = []

    for seed in range(n_seeds):
        print(f"测试种子 {seed + 1}/{n_seeds}...", end=" ")

        env = RampMeteringEnv(
            sumo_cfg=sumo_cfg,
            gui=False,
            seed=seed,  # 不同的种子
            save_metrics=False,
            max_steps=720,
            delta_time=5
        )

        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        results.append({
            'seed': seed,
            'total_reward': info.get('total_reward', episode_reward),
            'tts': info.get('total_tts', 0),
            'avg_bottleneck_speed': info.get('avg_bottleneck_speed', 0),
            'avg_ramp_queue': info.get('avg_ramp_queue', 0)
        })

        print(f"TTS: {info.get('total_tts', 0):.0f}, Reward: {episode_reward:.2f}")
        env.close()

    # 统计分析
    print(f"\n{'=' * 70}")
    print("统计结果:")
    print(f"{'=' * 70}")

    metrics = ['total_reward', 'tts', 'avg_bottleneck_speed', 'avg_ramp_queue']
    metric_names = ['总奖励', 'TTS', '瓶颈速度', '匝道排队']

    for metric, name in zip(metrics, metric_names):
        values = [r[metric] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0  # 变异系数

        print(f"{name:12s}: {mean_val:8.2f} ± {std_val:6.2f} (CV: {cv:5.2f}%)")

    # 判断
    tts_values = [r['tts'] for r in results]
    cv_tts = np.std(tts_values) / np.mean(tts_values) * 100

    print(f"\n{'=' * 70}")
    if cv_tts < 5:
        print("✅ 优秀: TTS变异系数 < 5%，性能非常稳定")
    elif cv_tts < 10:
        print("✅ 良好: TTS变异系数 < 10%，性能稳定")
    elif cv_tts < 20:
        print("⚠️  一般: TTS变异系数 < 20%，有一定波动")
    else:
        print("❌ 警告: TTS变异系数 > 20%，性能不稳定，可能过拟合!")
    print(f"{'=' * 70}\n")

    # 保存结果
    with open('robustness_test_seeds.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def test_demand_levels(
        model_path="./models/ppo_ramp_metering_v2",
        base_cfg="congestion.sumocfg"
):
    """
    测试2: 不同需求水平下的性能
    (需要您手动创建不同需求的配置文件)
    """
    print(f"\n{'=' * 70}")
    print("鲁棒性测试 2: 不同需求水平")
    print(f"{'=' * 70}\n")

    # 这里需要您准备不同需求水平的配置文件
    scenarios = [
        ("低峰", "congestion_low.sumocfg"),
        ("正常", "congestion.sumocfg"),
        ("高峰", "congestion_high.sumocfg"),
    ]

    model = PPO.load(model_path)
    results = {}

    for name, cfg in scenarios:
        # 检查文件是否存在
        if not os.path.exists(cfg):
            print(f"⚠️  跳过 {name}: 配置文件 {cfg} 不存在")
            continue

        print(f"测试场景: {name}...", end=" ")

        env = RampMeteringEnv(
            sumo_cfg=cfg,
            gui=False,
            seed=999,
            save_metrics=False,
            max_steps=720,
            delta_time=5
        )

        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        results[name] = {
            'total_reward': info.get('total_reward', episode_reward),
            'tts': info.get('total_tts', 0),
            'avg_bottleneck_speed': info.get('avg_bottleneck_speed', 0)
        }

        print(f"TTS: {info.get('total_tts', 0):.0f}")
        env.close()

    # 显示结果
    if results:
        print(f"\n{'=' * 70}")
        print(f"{'场景':<10} {'TTS':>12} {'瓶颈速度':>12} {'总奖励':>12}")
        print(f"{'-' * 70}")
        for name, metrics in results.items():
            print(
                f"{name:<10} {metrics['tts']:>12.0f} {metrics['avg_bottleneck_speed']:>12.2f} {metrics['total_reward']:>12.2f}")
        print(f"{'=' * 70}\n")

    return results


def test_action_diversity(
        model_path="./models/ppo_ramp_metering_v2",
        sumo_cfg="congestion.sumocfg",
        n_episodes=5
):
    """
    测试3: 动作多样性
    检查模型是否总是输出相同的动作
    """
    print(f"\n{'=' * 70}")
    print("鲁棒性测试 3: 动作多样性")
    print(f"{'=' * 70}\n")

    model = PPO.load(model_path)
    all_actions = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...", end=" ")

        env = RampMeteringEnv(
            sumo_cfg=sumo_cfg,
            gui=False,
            seed=episode,
            save_metrics=False,
            max_steps=720,
            delta_time=5
        )

        obs, _ = env.reset()
        done = False
        episode_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        all_actions.extend(episode_actions)
        print(f"动作范围: [{min(episode_actions):.3f}, {max(episode_actions):.3f}]")
        env.close()

    # 分析动作分布
    all_actions = np.array(all_actions)

    print(f"\n{'=' * 70}")
    print("动作统计:")
    print(f"{'=' * 70}")
    print(f"平均动作:     {np.mean(all_actions):.4f}")
    print(f"动作标准差:   {np.std(all_actions):.4f}")
    print(f"动作范围:     [{np.min(all_actions):.4f}, {np.max(all_actions):.4f}]")
    print(f"唯一值数量:   {len(np.unique(np.round(all_actions, 3)))}")

    # 判断
    action_std = np.std(all_actions)
    print(f"\n{'=' * 70}")
    if action_std < 0.01:
        print("❌ 警告: 动作几乎不变（标准差 < 0.01），严重过拟合!")
        print("   模型可能学会了一个固定策略，缺乏适应性")
    elif action_std < 0.05:
        print("⚠️  注意: 动作变化较小（标准差 < 0.05），可能过拟合")
    else:
        print("✅ 良好: 动作有足够的多样性")
    print(f"{'=' * 70}\n")

    return all_actions


def run_all_tests(model_path="./models/ppo_ramp_metering_v2"):
    """运行所有鲁棒性测试"""

    print("\n" + "=" * 70)
    print(" " * 20 + "PPO 鲁棒性测试套件")
    print("=" * 70)

    # 测试1: 不同种子
    seed_results = test_robustness(model_path=model_path, n_seeds=10)

    # 测试2: 不同需求（如果配置文件存在）
    # demand_results = test_demand_levels(model_path=model_path)

    # 测试3: 动作多样性
    actions = test_action_diversity(model_path=model_path, n_episodes=5)

    # 综合评估
    print("\n" + "=" * 70)
    print("综合评估:")
    print("=" * 70)

    # 计算TTS变异系数
    tts_values = [r['tts'] for r in seed_results]
    cv_tts = np.std(tts_values) / np.mean(tts_values) * 100

    # 计算动作标准差
    action_std = np.std(actions)

    score = 0
    max_score = 3

    print("\n评分:")
    if cv_tts < 10:
        print("  [✅] 性能稳定性: 良好")
        score += 1
    else:
        print("  [⚠️ ] 性能稳定性: 需改进")

    if action_std > 0.05:
        print("  [✅] 动作多样性: 良好")
        score += 1
    else:
        print("  [⚠️ ] 动作多样性: 需改进")

    # 简化的TTS绝对值评估
    avg_tts = np.mean(tts_values)
    if avg_tts < 25000:
        print("  [✅] 绝对性能: 优秀")
        score += 1
    else:
        print("  [⚠️ ] 绝对性能: 一般")

    print(f"\n总分: {score}/{max_score}")

    if score == max_score:
        print("\n✅ 模型鲁棒性优秀，可以放心使用!")
    elif score >= 2:
        print("\n⚠️  模型性能良好，但建议增加训练多样性")
    else:
        print("\n❌ 模型可能存在过拟合，建议重新训练:")
        print("   - 增加环境随机性")
        print("   - 使用更多不同场景训练")
        print("   - 调整奖励函数")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()