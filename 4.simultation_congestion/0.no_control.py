#!/usr/bin/env python3
"""
0_no_control.py

Baseline: 没有任何匝道控制（No Ramp Metering）
- 不设置 / 不修改 ramp 信号灯，相当于匝道完全自由进入主线
- 指标统计方式与 1_fixed.py / 2_ALINEA.py 完全一致，方便对比
"""

import os
import sys
import numpy as np
import json
from typing import Dict

# ---------- SUMO / TraCI 设置 ----------
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


class NoControlController:
    """
    无匝道控制：
    - 不更改 ramp meter 的信号逻辑
    - 如果你在 network.net.xml 中已经删除 tlLogic / 把 junction 设为 priority，
      那就是完全“无信号灯”的匝道。
    - 如果还保留了 meter 信号灯，也不会在这里动态改变它的 program
    """

    def __init__(self, sumo_cfg: str):
        self.sumo_cfg = sumo_cfg

        # 这些要和你的路网一致（和 1_fixed / 2_ALINEA 保持相同）
        self.MAINLINE_EDGES = ['main_1', 'main_2', 'main_3']
        self.RAMP_EDGES = ['ramp_0', 'ramp_1']
        self.BOTTLENECK_EDGE = 'main_3'
        self.METER_ID = 'meter'   # 如果你已经删除了信号灯，这个 ID 就不会被用到

        # 记录每辆车的出发时间，用于 travel time 计算
        self.depart_times: Dict[str, float] = {}

    def run_episode(self, max_steps: int = 3600):
        """
        运行一个 episode
        这里完全不做控制，只做统计
        """
        sumo_cmd = [
            "sumo-gui",  # 想看画面可以改为 "sumo-gui"
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--time-to-teleport", "-1"
        ]
        traci.start(sumo_cmd)

        metrics = {
            'mainline_speed': [],
            'bottleneck_speed': [],
            'ramp_queue': [],
            'throughput': [],
            'occupancy': [],
            'metering_rate': [],   # 这里可以理解为“自然状态”，我们用 NaN 占位
            'travel_times': [],
            'total_vehicles': 0,
            'congestion_count': 0
        }

        step = 0

        try:
            while step < max_steps:
                traci.simulationStep()
                step += 1
                now = traci.simulation.getTime()

                # ---- 状态统计（每个 step 都记一次）----
                state = self._get_state()
                metrics['mainline_speed'].append(state['mainline_speed'])
                metrics['bottleneck_speed'].append(state['bottleneck_speed'])
                metrics['ramp_queue'].append(state['ramp_queue'])
                metrics['throughput'].append(state['throughput'])
                metrics['occupancy'].append(state['occupancy'])
                metrics['metering_rate'].append(np.nan)  # 无控制，用 NaN 占位

                if state['bottleneck_speed'] < 15:
                    metrics['congestion_count'] += 1

                # ---- 统计出发车辆（需求）----
                departed_ids = traci.simulation.getDepartedIDList()
                metrics['total_vehicles'] += len(departed_ids)
                for vid in departed_ids:
                    self.depart_times[vid] = now

                # ---- 统计到达车辆 travel time（吞吐 + 出行时间）----
                arrived_ids = traci.simulation.getArrivedIDList()
                for vid in arrived_ids:
                    dep_t = self.depart_times.pop(vid, now)
                    tt = now - dep_t
                    if tt >= 0:
                        metrics['travel_times'].append(tt)

        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            traci.close()

        summary = self._compute_summary(metrics, max_steps)
        return summary

    # ---------- 状态获取 ----------

    def _get_state(self) -> Dict:
        """获取当前交通状态（与 fixed / ALINEA 保持一致）"""
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
        except Exception:
            return {
                'mainline_speed': 0, 'bottleneck_speed': 0,
                'occupancy': 0, 'ramp_queue': 0, 'throughput': 0
            }

    # ---------- 结果汇总 ----------

    def _compute_summary(self, metrics: Dict, max_steps: int) -> Dict:
        summary = {}
        for key in ['mainline_speed', 'bottleneck_speed', 'ramp_queue',
                    'throughput', 'occupancy']:
            if metrics[key]:
                summary[f'avg_{key}'] = float(np.mean(metrics[key]))
                summary[f'std_{key}'] = float(np.std(metrics[key]))
                summary[f'max_{key}'] = float(np.max(metrics[key]))
                summary[f'min_{key}'] = float(np.min(metrics[key]))

        # travel time 统计
        if metrics['travel_times']:
            summary['avg_travel_time'] = float(np.mean(metrics['travel_times']))
            summary['std_travel_time'] = float(np.std(metrics['travel_times']))
            summary['max_travel_time'] = float(np.max(metrics['travel_times']))
            summary['min_travel_time'] = float(np.min(metrics['travel_times']))
        else:
            summary['avg_travel_time'] = 0.0
            summary['std_travel_time'] = 0.0
            summary['max_travel_time'] = 0.0
            summary['min_travel_time'] = 0.0

        summary['total_vehicles'] = metrics['total_vehicles']
        summary['congestion_percentage'] = (
            metrics['congestion_count'] / len(metrics['bottleneck_speed']) * 100
            if metrics['bottleneck_speed'] else 0
        )
        summary['simulation_time'] = max_steps
        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="No-Control Ramp Baseline")
    parser.add_argument("--config", type=str, default="congestion.sumocfg")
    parser.add_argument("--steps", type=int, default=3600)
    args = parser.parse_args()

    controller = NoControlController(args.config)
    result = controller.run_episode(max_steps=args.steps)

    print("\n=== No-Control Ramp Results ===")
    print(f"Avg bottleneck speed: {result.get('avg_bottleneck_speed', 0):.2f} m/s")
    print(f"Avg mainline speed:   {result.get('avg_mainline_speed', 0):.2f} m/s")
    print(f"Avg ramp queue:       {result.get('avg_ramp_queue', 0):.2f} veh")
    print(f"Avg travel time:      {result.get('avg_travel_time', 0):.2f} s")
    print(f"Total vehicles:       {result.get('total_vehicles', 0)}")
    print(f"Congestion %:         {result.get('congestion_percentage', 0):.2f}%\n")

    with open("no_control_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("✅ Saved to no_control_results.json")


if __name__ == "__main__":
    main()
