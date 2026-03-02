#!/usr/bin/env python3
"""
Baseline策略对比脚本
对比4种策略:
1. No Control (无控制 - 匝道始终放行)
2. Fixed Control (固定控制 - 50%绿灯)
3. ALINEA (经典反馈控制)
4. PPO (训练好的强化学习策略)

#30%：匝道强限制，造成匝道排队严重 → 偏向保护主线

70%：匝道弱限制，可能主线拥堵 → 偏向保护匝道

50% 最中性，不偏主线也不偏匝道(关于fixed 方案)

"""

import os
import sys
import numpy as np
import json
from typing import Dict, List

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci


class BaselineController:
    """基础控制器类"""
    
    def __init__(self, sumo_cfg: str, strategy: str = "no_control"):
        self.sumo_cfg = sumo_cfg
        self.strategy = strategy
        
        # 网络配置
        self.MAINLINE_EDGES = ['main_1', 'main_2', 'main_3']
        self.RAMP_EDGES = ['ramp_0', 'ramp_1']
        self.BOTTLENECK_EDGE = 'main_3'
        self.METER_ID = 'meter'
        
        # ALINEA参数
        self.kr = 70.0  # 调节参数
        self.target_occupancy = 0.25  # 目标占有率
        self.last_metering_rate = 0.5
        
    def run_episode(self, max_steps: int = 3600, control_interval: int = 30):
        """运行一个episode"""
        
        # 启动SUMO
        sumo_cmd = [
            "sumo",
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end", 
            "--no-warnings",
            "--time-to-teleport", "-1"  # 禁用teleport
        ]
        
        traci.start(sumo_cmd)
        
        # 指标记录
        metrics = {
            'mainline_speed': [],
            'bottleneck_speed': [],
            'ramp_queue': [],
            'throughput': [],
            'occupancy': [],
            'metering_rate': [],
            'travel_time': [],
            'total_vehicles': 0,
            'congestion_count': 0
        }
        
        step = 0
        last_control = 0
        
        try:
            while step < max_steps:
                traci.simulationStep()
                step += 1
                
                # 控制逻辑
                if step - last_control >= control_interval:
                    state = self._get_state()
                    
                    # 根据策略选择动作
                    if self.strategy == "no_control":
                        green_ratio = 1.0  # 几乎总是放行
                    elif self.strategy == "fixed":
                        green_ratio = 0.5   # 固定50%
                    elif self.strategy == "alinea":
                        green_ratio = self._alinea_control(state)
                    else:
                        green_ratio = 0.5
                    
                    # 应用控制
                    self._apply_metering(green_ratio, control_interval)
                    
                    # 记录指标
                    metrics['mainline_speed'].append(state['mainline_speed'])
                    metrics['bottleneck_speed'].append(state['bottleneck_speed'])
                    metrics['ramp_queue'].append(state['ramp_queue'])
                    metrics['throughput'].append(state['throughput'])
                    metrics['occupancy'].append(state['occupancy'])
                    metrics['metering_rate'].append(green_ratio * 3600)
                    
                    # 拥堵判定
                    if state['bottleneck_speed'] < 15:
                        metrics['congestion_count'] += 1
                    
                    last_control = step
                
                # 记录通过车辆数
                departed = traci.simulation.getDepartedNumber()
                metrics['total_vehicles'] += departed
                
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            traci.close()
        
        return self._compute_summary(metrics, max_steps)
    
    def _get_state(self) -> Dict:
        """获取交通状态"""
        try:
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e) 
                for e in self.MAINLINE_EDGES
            ])
            
            bottleneck_speed = traci.edge.getLastStepMeanSpeed(self.BOTTLENECK_EDGE)
            bottleneck_occ = traci.edge.getLastStepOccupancy(self.BOTTLENECK_EDGE)
            
            ramp_queue = sum(
                traci.edge.getLastStepHaltingNumber(e) 
                for e in self.RAMP_EDGES
            )
            
            throughput = sum(
                traci.edge.getLastStepMeanSpeed(e) * 
                traci.edge.getLastStepVehicleNumber(e)
                for e in self.MAINLINE_EDGES
            )
            
            return {
                'mainline_speed': mainline_speed,
                'bottleneck_speed': bottleneck_speed,
                'occupancy': bottleneck_occ,
                'ramp_queue': ramp_queue,
                'throughput': throughput
            }
        except:
            return {
                'mainline_speed': 0, 'bottleneck_speed': 0,
                'occupancy': 0, 'ramp_queue': 0, 'throughput': 0
            }
    
    def _alinea_control(self, state: Dict) -> float:
        """ALINEA控制算法"""
        # 当前占有率
        current_occ = state['occupancy']
        
        # ALINEA公式: r(k) = r(k-1) + KR * (O_target - O_current)
        # 转换为绿灯比例
        error = self.target_occupancy - current_occ
        adjustment = self.kr * error
        
        # 更新放行率 (归一化到[0.2, 0.9])
        new_rate = self.last_metering_rate + adjustment / 3600.0
        new_rate = np.clip(new_rate, 0.2, 0.9)
        
        self.last_metering_rate = new_rate
        return new_rate
    
    def _apply_metering(self, green_ratio: float, cycle_time: int):
        """应用匝道控制"""
        try:
            green_time = int(green_ratio * cycle_time)
            red_time = cycle_time - green_time
            
            logic = traci.trafficlight.Logic(
                programID="control",
                type=0,
                currentPhaseIndex=0,
                phases=[
                    traci.trafficlight.Phase(green_time, "G"),
                    traci.trafficlight.Phase(red_time, "r")
                ]
            )
            
            traci.trafficlight.setProgramLogic(self.METER_ID, logic)
            traci.trafficlight.setProgram(self.METER_ID, "control")
        except:
            pass
    
    def _compute_summary(self, metrics: Dict, max_steps: int) -> Dict:
        """计算汇总统计"""
        summary = {}
        
        for key in ['mainline_speed', 'bottleneck_speed', 'ramp_queue', 
                    'throughput', 'occupancy', 'metering_rate']:
            if metrics[key]:
                summary[f'avg_{key}'] = float(np.mean(metrics[key]))
                summary[f'std_{key}'] = float(np.std(metrics[key]))
                summary[f'max_{key}'] = float(np.max(metrics[key]))
                summary[f'min_{key}'] = float(np.min(metrics[key]))
        
        summary['total_vehicles'] = metrics['total_vehicles']
        summary['congestion_percentage'] = (
            metrics['congestion_count'] / len(metrics['bottleneck_speed']) * 100
            if metrics['bottleneck_speed'] else 0
        )
        summary['simulation_time'] = max_steps
        
        return summary


def compare_all_strategies(
    sumo_cfg: str,
    n_episodes: int = 3,
    max_steps: int = 3600
):
    """对比所有策略"""
    
    strategies = {
        'no_control': 'No Control (无控制)',
        'fixed': 'Fixed 50% (固定控制)',
        'alinea': 'ALINEA (经典控制)',
    }
    
    print(f"\n{'='*70}")
    print("Baseline Strategies Comparison")
    print(f"{'='*70}")
    print(f"Episodes per strategy: {n_episodes}")
    print(f"Simulation time: {max_steps}s\n")
    
    all_results = {}
    
    for strategy_key, strategy_name in strategies.items():
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*70}\n")
        
        controller = BaselineController(sumo_cfg, strategy_key)
        episode_results = []
        
        for episode in range(n_episodes):
            print(f"  Episode {episode + 1}/{n_episodes}...", end=" ")
            result = controller.run_episode(max_steps)
            episode_results.append(result)
            print(f"✓ (Avg Speed: {result['avg_bottleneck_speed']:.1f} m/s, "
                  f"Queue: {result['avg_ramp_queue']:.1f} veh)")
        
        # 计算平均结果
        avg_results = {}
        for key in episode_results[0].keys():
            if isinstance(episode_results[0][key], (int, float)):
                values = [r[key] for r in episode_results]
                avg_results[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        all_results[strategy_key] = {
            'name': strategy_name,
            'episodes': episode_results,
            'average': avg_results
        }
    
    # 打印对比结果
    print_comparison_table(all_results)
    
    # 保存结果
    with open('baseline_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to baseline_comparison.json\n")
    
    return all_results


def print_comparison_table(results: Dict):
    """打印对比表格"""
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")
    
    # 表头
    strategies = list(results.keys())
    print(f"{'Metric':<25} ", end="")
    for key in strategies:
        print(f"{results[key]['name']:<20} ", end="")
    print()
    print("-" * 70)
    
    # 关键指标
    metrics = [
        ('avg_bottleneck_speed', 'Bottleneck Speed (m/s)'),
        ('avg_mainline_speed', 'Mainline Speed (m/s)'),
        ('avg_ramp_queue', 'Ramp Queue (veh)'),
        ('max_ramp_queue', 'Max Queue (veh)'),
        ('avg_throughput', 'Throughput'),
        ('congestion_percentage', 'Congestion (%)'),
        ('total_vehicles', 'Total Vehicles'),
    ]
    
    for metric_key, metric_name in metrics:
        print(f"{metric_name:<25} ", end="")
        
        for strategy_key in strategies:
            avg = results[strategy_key]['average']
            if metric_key in avg:
                mean = avg[metric_key]['mean']
                std = avg[metric_key]['std']
                print(f"{mean:6.2f} ± {std:5.2f}      ", end="")
            else:
                print(f"{'N/A':<20} ", end="")
        print()
    
    print("=" * 70)
    
    # 找出最佳策略
    print("\n📊 Performance Ranking:")
    
    # 按瓶颈速度排序（越高越好）
    speed_ranking = sorted(
        strategies,
        key=lambda k: results[k]['average']['avg_bottleneck_speed']['mean'],
        reverse=True
    )
    
    print("\n  By Bottleneck Speed (higher is better):")
    for i, key in enumerate(speed_ranking, 1):
        name = results[key]['name']
        speed = results[key]['average']['avg_bottleneck_speed']['mean']
        print(f"    {i}. {name}: {speed:.2f} m/s")
    
    # 按排队长度排序（越低越好）
    queue_ranking = sorted(
        strategies,
        key=lambda k: results[k]['average']['avg_ramp_queue']['mean']
    )
    
    print("\n  By Ramp Queue (lower is better):")
    for i, key in enumerate(queue_ranking, 1):
        name = results[key]['name']
        queue = results[key]['average']['avg_ramp_queue']['mean']
        print(f"    {i}. {name}: {queue:.2f} veh")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Comparison")
    parser.add_argument("--config", type=str,
                       default="congestion.sumocfg")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3600)
    
    args = parser.parse_args()
    
    compare_all_strategies(
        sumo_cfg=args.config,
        n_episodes=args.episodes,
        max_steps=args.steps
    )
