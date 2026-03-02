"""
基于纯交通需求驱动的拥堵场景控制器 - 最终版
基于真实路网结构：
- main_1: 2车道主线上游
- main_2: 3车道合流区（包含1条辅助加速车道）
- main_3: 2车道主线下游（瓶颈）
- ramp_0 → meter → ramp_1: 匝道（带信号灯）
"""

import os
import sys
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')


class RealisticCongestionController:
    """真实交通需求驱动的拥堵场景控制器"""

    def __init__(self, sumo_cfg_file, gui=True):
        self.sumo_cfg = sumo_cfg_file
        self.use_gui = gui

        # ============ 路网结构（基于实际路网） ============
        self.edges = {
            'mainline': ['main_1', 'main_2', 'main_3'],
            'ramp': ['ramp_0', 'ramp_1'],
            'bottleneck': 'main_3',
            'merge_area': 'main_2'  # 3车道合流区
        }

        # ============ 容量计算 ============
        # main_1: 2车道
        # main_2: 3车道（但lane_0是辅助车道，不算主线容量）
        # main_3: 2车道（瓶颈）
        # 单车道容量 ≈ 2000 veh/h
        self.capacity = {
            'main_1': 2 * 2000,  # 4000 veh/h
            'main_2': 2 * 2000,  # 4000 veh/h（辅助车道不算）
            'main_3': 2 * 2000,  # 4000 veh/h（瓶颈）
            'ramp': 1000  # 1000 veh/h（单车道匝道）
        }

        # ============ 时段划分（调整后） ============
        self.time_phases = {
            'warmup': {
                'duration': (0, 600),
                'mainline_flow': 2500,  # 62.5%容量
                'ramp_flow': 400
            },
            'congestion_buildup': {
                'duration': (600, 1200),
                'mainline_flow': 4200,  # 105%容量
                'ramp_flow': 900
            },
            'peak': {
                'duration': (1200, 2400),
                'mainline_flow': 5000,  # 125%容量 - 超容量！
                'ramp_flow': 1100
            },
            'sustained_congestion': {
                'duration': (2400, 3000),
                'mainline_flow': 4800,  # 120%容量
                'ramp_flow': 1000
            },
            'recovery': {
                'duration': (3000, 3600),
                'mainline_flow': 3500,  # 87.5%容量
                'ramp_flow': 600
            }
        }

        # ============ 车辆类型（只用小客车） ============
        self.vehicle_types = {
            'passenger': {'probability': 1.0}
        }

        # ============ 路由定义 ============
        self.routes = {
            'mainline_through': ['main_1', 'main_2', 'main_3'],
            'ramp_merge': ['ramp_0', 'ramp_1', 'main_2', 'main_3']
        }

        # ============ 数据收集 ============
        self.metrics = defaultdict(list)
        self.vehicle_count = 0

        print("\n" + "=" * 70)
        print("Realistic Congestion Controller - Final Version")
        print("=" * 70)
        print(f"Network structure:")
        print(f"  main_1: 2 lanes (upstream)")
        print(f"  main_2: 3 lanes (2 main + 1 auxiliary for merging)")
        print(f"  main_3: 2 lanes (downstream bottleneck)")
        print(f"  ramp: ramp_0 → [meter signal] → ramp_1")
        print(f"\nCapacity:")
        print(f"  Bottleneck: {self.capacity['main_3']} veh/h")
        print(
            f"  Peak demand: {self.time_phases['peak']['mainline_flow'] + self.time_phases['peak']['ramp_flow']} veh/h")
        print(
            f"  Oversaturation: {((self.time_phases['peak']['mainline_flow'] + self.time_phases['peak']['ramp_flow']) / self.capacity['main_3'] - 1) * 100:.1f}%")
        print("=" * 70 + "\n")

    def start_simulation(self):
        """启动SUMO仿真"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--step-length", "1",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "300",
            "--no-warnings", "false",
            "--duration-log.statistics", "true",
            "--device.rerouting.probability", "0",
            "--random", "true"
        ]

        traci.start(sumo_cmd)
        print("[OK] SUMO simulation started")
        print(f"[OK] Available vehicle types: {traci.vehicletype.getIDList()}\n")

    def run_simulation(self):
        """运行完整仿真"""
        self.start_simulation()

        step = 0
        print("Starting demand-driven congestion simulation...")
        print("No manual speed control - vehicles follow SUMO's IDM model\n")

        try:
            while step < 3600:
                current_time = traci.simulation.getTime()

                # 生成车辆（唯一的控制点）
                self._generate_vehicles(current_time)

                # 收集数据
                self._collect_metrics(current_time)

                # SUMO处理交通流
                traci.simulationStep()

                step += 1

                # 每分钟报告
                if step % 60 == 0:
                    self._print_status(current_time)

        except KeyboardInterrupt:
            print("\n[WARNING] Simulation interrupted by user")

        finally:
            traci.close()
            print("\n[OK] Simulation finished")
            self._analyze_and_visualize()

    def _generate_vehicles(self, current_time):
        """根据时段生成车辆"""
        current_phase = None
        for phase_name, phase_data in self.time_phases.items():
            start, end = phase_data['duration']
            if start <= current_time < end:
                current_phase = phase_data
                break

        if current_phase is None:
            return

        # 每秒生成概率
        mainline_prob = current_phase['mainline_flow'] / 3600.0
        ramp_prob = current_phase['ramp_flow'] / 3600.0

        # 主线车辆（从main_1起点生成）
        if np.random.random() < mainline_prob:
            self._insert_vehicle('mainline_through', 'main_1')

        # 匝道车辆（从ramp_0起点生成）
        if np.random.random() < ramp_prob:
            self._insert_vehicle('ramp_merge', 'ramp_0')

    def _insert_vehicle(self, route_type, start_edge):
        """插入车辆到SUMO"""
        veh_id = f"{route_type}_{self.vehicle_count}"
        self.vehicle_count += 1

        try:
            traci.vehicle.add(
                veh_id,
                routeID="",
                typeID='passenger',
                departSpeed="max",
                departLane="best"
            )
            traci.vehicle.setRoute(veh_id, self.routes[route_type])
        except traci.exceptions.TraCIException:
            # 道路太满，无法插入（说明拥堵开始形成）
            pass

    def _collect_metrics(self, current_time):
        """收集交通流指标"""
        all_vehicles = traci.vehicle.getIDList()
        if len(all_vehicles) == 0:
            return

        speeds = []
        stopped_count = 0
        waiting_times = []
        edge_vehicles = defaultdict(int)
        edge_speeds = defaultdict(list)

        for veh_id in all_vehicles:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                edge = traci.vehicle.getRoadID(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)

                speeds.append(speed)
                waiting_times.append(waiting_time)

                if speed < 0.1:
                    stopped_count += 1

                if edge.startswith('main'):
                    edge_vehicles[edge] += 1
                    edge_speeds[edge].append(speed)
            except:
                continue

        # 测量流量（修正版）
        bottleneck_flow = self._measure_flow('main_3')
        upstream_flow = self._measure_flow('main_2')

        # 记录数据
        self.metrics['time'].append(current_time)
        self.metrics['total_vehicles'].append(len(all_vehicles))
        self.metrics['avg_speed'].append(np.mean(speeds) if speeds else 0)
        self.metrics['stopped_count'].append(stopped_count)
        self.metrics['stopped_ratio'].append(stopped_count / len(all_vehicles) if len(all_vehicles) > 0 else 0)
        self.metrics['avg_waiting_time'].append(np.mean(waiting_times) if waiting_times else 0)
        self.metrics['bottleneck_flow'].append(bottleneck_flow)
        self.metrics['upstream_flow'].append(upstream_flow)

        # 瓶颈速度
        if 'main_3' in edge_speeds and edge_speeds['main_3']:
            self.metrics['bottleneck_speed'].append(np.mean(edge_speeds['main_3']))
        else:
            self.metrics['bottleneck_speed'].append(0)

        # 瓶颈密度
        try:
            bottleneck_length = traci.lane.getLength('main_3_0')
            bottleneck_density = edge_vehicles.get('main_3', 0) / (bottleneck_length / 1000)
            self.metrics['bottleneck_density'].append(bottleneck_density)
        except:
            self.metrics['bottleneck_density'].append(0)

    def _measure_flow(self, edge_id):
        """
        测量通过指定路段的流量（修正版）
        基于车辆离开路段的计数
        """
        if not hasattr(self, 'flow_trackers'):
            self.flow_trackers = {}

        if edge_id not in self.flow_trackers:
            self.flow_trackers[edge_id] = {
                'vehicle_set': set(),
                'departed_count': 0,
                'start_time': traci.simulation.getTime(),
                'last_flow': 0
            }

        tracker = self.flow_trackers[edge_id]
        current_time = traci.simulation.getTime()

        # 获取当前在该路段的所有车辆
        try:
            current_vehicles = set([
                veh for veh in traci.vehicle.getIDList()
                if traci.vehicle.getRoadID(veh) == edge_id
            ])
        except:
            current_vehicles = set()

        # 计算离开该路段的车辆数（之前在，现在不在）
        departed = tracker['vehicle_set'] - current_vehicles
        tracker['departed_count'] += len(departed)

        # 更新车辆集合
        tracker['vehicle_set'] = current_vehicles

        # 每60秒计算一次流量
        time_elapsed = current_time - tracker['start_time']

        if time_elapsed >= 60:
            # 计算平均流量
            if time_elapsed > 0:
                flow = (tracker['departed_count'] / time_elapsed) * 3600
            else:
                flow = 0

            # 保存并重置
            tracker['last_flow'] = flow
            tracker['departed_count'] = 0
            tracker['start_time'] = current_time

            return flow

        # 时间不足60秒，返回上次的流量
        return tracker['last_flow']

    def _print_status(self, current_time):
        """打印当前状态"""
        if not self.metrics['time']:
            return

        avg_speed = self.metrics['avg_speed'][-1]
        stopped_ratio = self.metrics['stopped_ratio'][-1]
        total_vehs = self.metrics['total_vehicles'][-1]
        bottleneck_flow = self.metrics['bottleneck_flow'][-1]

        # 判断拥堵状态
        if avg_speed < 5.0:
            status = "[!!] SEVERE CONGESTION"
        elif avg_speed < 10.0:
            status = "[!] CONGESTED"
        else:
            status = "[OK] FLOWING"

        # 容量比例
        capacity_ratio = bottleneck_flow / self.capacity['main_3'] if self.capacity['main_3'] > 0 else 0

        print(
            f"[{current_time / 60:5.1f} min] {status:25s} | "
            f"Speed: {avg_speed:5.2f} m/s ({avg_speed * 3.6:5.1f} km/h) | "
            f"Vehicles: {total_vehs:4d} | "
            f"Stopped: {stopped_ratio * 100:5.1f}% | "
            f"Flow: {bottleneck_flow:5.0f} veh/h ({capacity_ratio * 100:5.1f}%)"
        )

    def _analyze_and_visualize(self):
        """分析结果并可视化"""
        df = pd.DataFrame(self.metrics)
        df['time_minutes'] = df['time'] / 60
        df['avg_speed_kmh'] = df['avg_speed'] * 3.6
        df['bottleneck_speed_kmh'] = df['bottleneck_speed'] * 3.6

        # 保存数据
        output_file = 'realistic_congestion_metrics.csv'
        df.to_csv(output_file, index=False)
        print(f"\n[SAVE] Metrics saved to: {output_file}")

        # 创建图表
        self._create_plots(df)

        # 分析容量下降
        self._analyze_capacity_drop(df)

        # 打印摘要
        self._print_summary(df)

    def _create_plots(self, df):
        """创建可视化图表"""
        fig = plt.figure(figsize=(16, 12))

        # 图1: 累积曲线
        ax1 = plt.subplot(3, 2, 1)
        upstream_cum = np.cumsum(df['upstream_flow'].values / 3600)
        bottleneck_cum = np.cumsum(df['bottleneck_flow'].values / 3600)
        ax1.plot(df['time_minutes'], upstream_cum, label='Upstream (main_2)', linewidth=2, color='#2E86AB')
        ax1.plot(df['time_minutes'], bottleneck_cum, label='Bottleneck (main_3)', linewidth=2, color='#A23B72')
        ax1.axvspan(10, 40, alpha=0.2, color='red', label='Peak period')
        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Cumulative vehicles', fontsize=11)
        ax1.set_title('Cumulative Arrival Curves', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 速度
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(df['time_minutes'], df['avg_speed_kmh'], linewidth=2, color='#2E86AB', label='Network avg')
        ax2.plot(df['time_minutes'], df['bottleneck_speed_kmh'], linewidth=2, color='#F18F01', label='Bottleneck')
        ax2.axhline(y=18, color='red', linestyle='--', label='Congestion threshold', alpha=0.7)
        ax2.axvspan(10, 40, alpha=0.2, color='red')
        ax2.set_xlabel('Time (min)', fontsize=11)
        ax2.set_ylabel('Speed (km/h)', fontsize=11)
        ax2.set_title('Speed Evolution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: 流量
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(df['time_minutes'], df['bottleneck_flow'], linewidth=2, color='#C73E1D', label='Bottleneck flow')
        ax3.axhline(y=self.capacity['main_3'], color='green', linestyle='--',
                    label=f"Capacity ({self.capacity['main_3']} veh/h)")
        ax3.axvspan(10, 40, alpha=0.2, color='red')
        ax3.set_xlabel('Time (min)', fontsize=11)
        ax3.set_ylabel('Flow (veh/h)', fontsize=11)
        ax3.set_title('Bottleneck Flow (Capacity Drop Analysis)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图4: 密度
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(df['time_minutes'], df['bottleneck_density'], linewidth=2, color='#A23B72', label='Bottleneck density')
        ax4.axvspan(10, 40, alpha=0.2, color='red')
        ax4.set_xlabel('Time (min)', fontsize=11)
        ax4.set_ylabel('Density (veh/km)', fontsize=11)
        ax4.set_title('Traffic Density', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 图5: 停车比例
        ax5 = plt.subplot(3, 2, 5)
        ax5.fill_between(df['time_minutes'], df['stopped_ratio'] * 100, alpha=0.6, color='#F18F01')
        ax5.axvspan(10, 40, alpha=0.2, color='red')
        ax5.set_xlabel('Time (min)', fontsize=11)
        ax5.set_ylabel('Stopped ratio (%)', fontsize=11)
        ax5.set_title('Congestion Severity', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 图6: 基本图
        ax6 = plt.subplot(3, 2, 6)
        valid = df[(df['bottleneck_density'] > 0) & (df['bottleneck_speed'] > 0)]
        if len(valid) > 0:
            scatter = ax6.scatter(valid['bottleneck_density'], valid['bottleneck_speed_kmh'],
                                  c=valid['time_minutes'], cmap='coolwarm', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax6, label='Time (min)')
        ax6.set_xlabel('Density (veh/km)', fontsize=11)
        ax6.set_ylabel('Speed (km/h)', fontsize=11)
        ax6.set_title('Fundamental Diagram', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output_img = 'realistic_congestion_analysis.png'
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Visualization saved to: {output_img}")
        plt.close()

    def _analyze_capacity_drop(self, df):
        """分析容量下降"""
        print("\n" + "=" * 70)
        print("CAPACITY DROP ANALYSIS")
        print("=" * 70)

        before_peak = df[(df['time'] >= 600) & (df['time'] < 1200)]
        during_peak = df[(df['time'] >= 1200) & (df['time'] < 2400)]

        if len(before_peak) > 0 and len(during_peak) > 0:
            flow_before = before_peak['bottleneck_flow'].mean()
            flow_during = during_peak['bottleneck_flow'].mean()

            if flow_before > 0:
                capacity_drop = (flow_before - flow_during) / flow_before * 100

                print(f"\nBottleneck flow before congestion: {flow_before:.0f} veh/h")
                print(f"Bottleneck flow during congestion: {flow_during:.0f} veh/h")
                print(f"Capacity drop: {capacity_drop:.1f}%")

                if capacity_drop > 5:
                    print("\n✅ Capacity drop observed - Congestion is REALISTIC")
                    print("   (Paper reports 2-30% capacity drop)")
                else:
                    print("\n⚠️  Capacity drop not significant")

        print("=" * 70 + "\n")

    def _print_summary(self, df):
        """打印统计摘要"""
        print("\n" + "=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)

        phases = [
            ('Warmup', 0, 600),
            ('Buildup', 600, 1200),
            ('Peak', 1200, 2400),
            ('Sustained', 2400, 3000),
            ('Recovery', 3000, 3600)
        ]

        for phase_name, start, end in phases:
            phase_data = df[(df['time'] >= start) & (df['time'] < end)]
            if len(phase_data) > 0:
                print(f"\n{phase_name} phase ({start / 60:.0f}-{end / 60:.0f} min):")
                print(f"  Avg speed: {phase_data['avg_speed_kmh'].mean():.2f} km/h")
                print(f"  Min speed: {phase_data['avg_speed_kmh'].min():.2f} km/h")
                print(f"  Stopped ratio: {phase_data['stopped_ratio'].mean() * 100:.1f}%")
                print(f"  Max vehicles: {phase_data['total_vehicles'].max():.0f}")

        print(f"\nOverall:")
        print(f"  Total vehicles generated: {self.vehicle_count}")
        print(f"  Avg speed: {df['avg_speed_kmh'].mean():.2f} km/h")
        print(f"  Max vehicles on network: {df['total_vehicles'].max():.0f}")

        severe = df[df['avg_speed_kmh'] < 18]
        if len(severe) > 0:
            print(f"  Severe congestion duration: {len(severe) / 60:.1f} min")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Realistic Congestion Controller - Final Version')
    parser.add_argument('--config', type=str, default='congestion.sumocfg',
                        help='SUMO configuration file')
    parser.add_argument('--gui', action='store_true', default=False,
                        help='Use SUMO GUI')

    args = parser.parse_args()

    controller = RealisticCongestionController(
        sumo_cfg_file=args.config,
        gui=args.gui
    )

    controller.run_simulation()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED!")
    print("Check outputs:")
    print("  - realistic_congestion_metrics.csv")
    print("  - realistic_congestion_analysis.png")
    print("=" * 70)