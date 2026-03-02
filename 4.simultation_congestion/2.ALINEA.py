#!/usr/bin/env python3
"""
ALINEA Ramp Metering
r(k) = r(k-1) + Kr * (O_target - O_current)
With TTS calculation following paper definition: TTS = T × ∑N(k)
"""

import os
import sys
import numpy as np
import json
from typing import Dict, Set

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


class AlineaController:
    def __init__(self, sumo_cfg: str):
        self.sumo_cfg = sumo_cfg

        self.MAINLINE_EDGES = ['main_1', 'main_2', 'main_3']
        self.RAMP_EDGES = ['ramp_0', 'ramp_1']
        self.BOTTLENECK_EDGE = 'main_3'
        self.METER_ID = 'meter'

        # ALINEA 参数
        self.kr = 70.0
        self.target_occupancy = 0.25  # 25%
        self.last_metering_rate = 0.5  # 上一周期绿灯比例

        # 记录出发时间和车辆类型
        self.depart_times: Dict[str, float] = {}
        self.mainline_vehicles: Set[str] = set()
        self.ramp_vehicles: Set[str] = set()

    def run_episode(self, max_steps: int = 3600, control_interval: int = 60):
        sumo_cmd = [
            "sumo",  # 或 "sumo-gui"
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--time-to-teleport", "-1",
            "--step-length", "1.0"  # 确保步长为 1 秒
        ]
        traci.start(sumo_cmd)

        metrics = {
            'mainline_speed': [],
            'bottleneck_speed': [],
            'ramp_queue': [],
            'throughput': [],
            'occupancy': [],
            'metering_rate': [],
            'travel_times': [],
            'total_vehicles': 0,
            'congestion_count': 0,
            # TTS 相关指标 - 按论文定义
            'total_tts': 0.0,  # Total Travel Spend = T × ∑N(k)
            'vehicle_count_per_step': []  # 记录每步的车辆数
        }

        step = 0
        last_control = 0

        try:
            while step < max_steps:
                traci.simulationStep()
                step += 1
                now = traci.simulation.getTime()

                # --- 计算当前网络中的车辆总数 (关键!) ---
                current_vehicle_count = self._get_network_vehicle_count()
                metrics['vehicle_count_per_step'].append(current_vehicle_count)
                # TTS 累加: 每一秒网络中的车辆数
                metrics['total_tts'] += current_vehicle_count * 1.0  # 1.0 秒步长

                if step - last_control >= control_interval:
                    state = self._get_state()

                    green_ratio = self._alinea_control(state)
                    self._apply_metering(green_ratio, control_interval)

                    metrics['mainline_speed'].append(state['mainline_speed'])
                    metrics['bottleneck_speed'].append(state['bottleneck_speed'])
                    metrics['ramp_queue'].append(state['ramp_queue'])
                    metrics['throughput'].append(state['throughput'])
                    metrics['occupancy'].append(state['occupancy'])
                    metrics['metering_rate'].append(green_ratio * 3600)

                    if state['bottleneck_speed'] < 15:
                        metrics['congestion_count'] += 1

                    last_control = step

                # 出发车辆
                departed_ids = traci.simulation.getDepartedIDList()
                metrics['total_vehicles'] += len(departed_ids)
                for vid in departed_ids:
                    self.depart_times[vid] = now
                    # 判断车辆来源
                    try:
                        route = traci.vehicle.getRoute(vid)
                        if route and route[0] in self.RAMP_EDGES:
                            self.ramp_vehicles.add(vid)
                        else:
                            self.mainline_vehicles.add(vid)
                    except:
                        self.mainline_vehicles.add(vid)

                # 到达车辆 travel time
                arrived_ids = traci.simulation.getArrivedIDList()
                for vid in arrived_ids:
                    dep_t = self.depart_times.pop(vid, now)
                    tt = now - dep_t
                    if tt >= 0:
                        metrics['travel_times'].append(tt)

                    # 清理车辆类型记录
                    self.mainline_vehicles.discard(vid)
                    self.ramp_vehicles.discard(vid)

        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            traci.close()

        summary = self._compute_summary(metrics, max_steps)
        return summary

    # ---------- TTS 计算 ----------

    def _get_network_vehicle_count(self) -> int:
        """
        获取当前网络中的总车辆数
        这是计算 TTS 的核心: TTS = T × ∑N(k)
        """
        try:
            # 方法1: 获取所有边上的车辆数
            total_count = 0

            # 主线车辆
            for edge in self.MAINLINE_EDGES:
                total_count += traci.edge.getLastStepVehicleNumber(edge)

            # 匝道车辆
            for edge in self.RAMP_EDGES:
                total_count += traci.edge.getLastStepVehicleNumber(edge)

            return total_count

            # 方法2 (备选): 直接获取所有车辆
            # return len(traci.vehicle.getIDList())

        except Exception as e:
            print(f"Error getting vehicle count: {e}")
            return 0

    # ---------- 状态 & ALINEA ----------

    def _get_state(self) -> Dict:
        try:
            mainline_speed = np.mean([
                traci.edge.getLastStepMeanSpeed(e)
                for e in self.MAINLINE_EDGES
            ])

            bottleneck_speed = traci.edge.getLastStepMeanSpeed(self.BOTTLENECK_EDGE)
            bottleneck_occ = traci.edge.getLastStepOccupancy(self.BOTTLENECK_EDGE) / 100.0

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

    def _alinea_control(self, state: Dict) -> float:
        current_occ = state['occupancy']  # 0~1
        error = self.target_occupancy - current_occ
        adjustment = self.kr * error

        new_rate = self.last_metering_rate + adjustment / 3600.0
        new_rate = float(np.clip(new_rate, 0.2, 0.9))

        self.last_metering_rate = new_rate
        return new_rate

    def _apply_metering(self, green_ratio: float, cycle_time: int):
        try:
            green_time = max(int(green_ratio * cycle_time), 1)
            red_time = max(cycle_time - green_time, 1)

            logic = traci.trafficlight.Logic(
                programID="alinea",
                type=0,
                currentPhaseIndex=0,
                phases=[
                    traci.trafficlight.Phase(green_time, "G"),
                    traci.trafficlight.Phase(red_time, "r")
                ]
            )
            traci.trafficlight.setProgramLogic(self.METER_ID, logic)
            traci.trafficlight.setProgram(self.METER_ID, "alinea")
        except Exception as e:
            print(f"Error applying metering: {e}")

    # ---------- 结果汇总 ----------

    def _compute_summary(self, metrics: Dict, max_steps: int) -> Dict:
        summary = {}
        for key in ['mainline_speed', 'bottleneck_speed', 'ramp_queue',
                    'throughput', 'occupancy', 'metering_rate']:
            if metrics[key]:
                summary[f'avg_{key}'] = float(np.mean(metrics[key]))
                summary[f'std_{key}'] = float(np.std(metrics[key]))
                summary[f'max_{key}'] = float(np.max(metrics[key]))
                summary[f'min_{key}'] = float(np.min(metrics[key]))

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

        # TTS 统计 (按论文定义)
        summary['total_tts'] = float(metrics['total_tts'])  # 单位: 车辆·秒 (veh·s)
        summary['total_tts_hours'] = float(metrics['total_tts'] / 3600.0)  # 转换为小时

        # 平均每辆车的时间消耗
        summary['avg_time_per_vehicle'] = (
            float(metrics['total_tts'] / metrics['total_vehicles'])
            if metrics['total_vehicles'] > 0 else 0.0
        )

        # 平均网络车辆数
        if metrics['vehicle_count_per_step']:
            summary['avg_vehicles_in_network'] = float(np.mean(metrics['vehicle_count_per_step']))
            summary['max_vehicles_in_network'] = float(np.max(metrics['vehicle_count_per_step']))

        summary['total_vehicles'] = metrics['total_vehicles']
        summary['congestion_percentage'] = (
            metrics['congestion_count'] / len(metrics['bottleneck_speed']) * 100
            if metrics['bottleneck_speed'] else 0
        )
        summary['simulation_time'] = max_steps
        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ALINEA Ramp Metering")
    parser.add_argument("--config", type=str, default="congestion.sumocfg")
    parser.add_argument("--steps", type=int, default=3600)
    args = parser.parse_args()

    controller = AlineaController(args.config)
    result = controller.run_episode(max_steps=args.steps, control_interval=60)

    print("\n=== ALINEA Results ===")
    print(f"Avg bottleneck speed:     {result.get('avg_bottleneck_speed', 0):.2f} m/s")
    print(f"Avg mainline speed:       {result.get('avg_mainline_speed', 0):.2f} m/s")
    print(f"Avg ramp queue:           {result.get('avg_ramp_queue', 0):.2f} veh")
    print(f"Avg travel time:          {result.get('avg_travel_time', 0):.2f} s")
    print(f"\n--- TTS (Total Travel Spend) ---")
    print(f"Total TTS:                {result.get('total_tts', 0):.2f} veh·s")
    print(f"Total TTS:                {result.get('total_tts_hours', 0):.2f} veh·h")
    print(f"Avg time per vehicle:     {result.get('avg_time_per_vehicle', 0):.2f} s")
    print(f"Avg vehicles in network:  {result.get('avg_vehicles_in_network', 0):.2f} veh")
    print(f"Max vehicles in network:  {result.get('max_vehicles_in_network', 0):.0f} veh")
    print(f"\nTotal vehicles:           {result.get('total_vehicles', 0)}")
    print(f"Congestion %:             {result.get('congestion_percentage', 0):.2f}%\n")

    with open("alinea_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("✅ Saved to alinea_results.json")


if __name__ == "__main__":
    main()