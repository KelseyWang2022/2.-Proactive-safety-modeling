#!/usr/bin/env python3
"""
匝道汇入行为对比测试
对比3种场景：
1. 默认SUMO行为（不真实 - 总能插入）
2. 真实汇入行为（保守参数）
3. 有ramp metering控制
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci


def test_merging_behavior(route_file: str, test_name: str, with_control: bool = False):
    """测试特定配置下的匝道汇入行为"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}\n")
    
    # 创建临时配置
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="/mnt/user-data/uploads/network_net.xml"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>
</configuration>"""
    
    temp_config = f"/tmp/test_{test_name.replace(' ', '_')}.sumocfg"
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    # 启动SUMO
    sumo_cmd = [
        "sumo",
        "-c", temp_config,
        "--no-warnings"
    ]
    
    traci.start(sumo_cmd)
    
    # 数据收集
    data = {
        'time': [],
        'ramp_queue': [],
        'ramp_waiting': [],
        'merge_attempts': [],
        'merge_success': [],
        'mainline_speed': [],
        'merge_speed': [],
        'ramp_vehicles': []
    }
    
    step = 0
    last_ramp_count = 0
    
    try:
        while step < 3600:
            traci.simulationStep()
            step += 1
            
            # 每10秒记录一次
            if step % 10 == 0:
                # 匝道排队
                ramp_queue = sum(
                    traci.edge.getLastStepHaltingNumber(e)
                    for e in ['ramp_0', 'ramp_1']
                )
                
                # 匝道车辆总数
                ramp_vehs = sum(
                    traci.edge.getLastStepVehicleNumber(e)
                    for e in ['ramp_0', 'ramp_1']
                )
                
                # 平均等待时间
                wait_times = []
                for edge in ['ramp_0', 'ramp_1']:
                    for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                        try:
                            wait_times.append(
                                traci.vehicle.getAccumulatedWaitingTime(veh_id)
                            )
                        except:
                            pass
                avg_wait = np.mean(wait_times) if wait_times else 0
                
                # 主线和合流区速度
                mainline_speed = traci.edge.getLastStepMeanSpeed('main_1')
                merge_speed = traci.edge.getLastStepMeanSpeed('main_2')
                
                # 记录
                data['time'].append(step)
                data['ramp_queue'].append(ramp_queue)
                data['ramp_waiting'].append(avg_wait)
                data['mainline_speed'].append(mainline_speed)
                data['merge_speed'].append(merge_speed)
                data['ramp_vehicles'].append(ramp_vehs)
                
                # 打印关键时刻
                if step in [600, 1200, 1800, 2400, 3000]:
                    print(f"Time {step}s: "
                          f"Queue={ramp_queue} veh, "
                          f"Wait={avg_wait:.1f}s, "
                          f"Merge Speed={merge_speed:.1f} m/s")
            
            # 简单控制（如果启用）
            if with_control and step % 15 == 0:
                bottle_speed = traci.edge.getLastStepMeanSpeed('main_3')
                if bottle_speed < 20:
                    green_ratio = 0.3
                else:
                    green_ratio = 0.7
                
                try:
                    green_time = int(green_ratio * 15)
                    logic = traci.trafficlight.Logic(
                        programID="control",
                        type=0,
                        currentPhaseIndex=0,
                        phases=[
                            traci.trafficlight.Phase(green_time, "G"),
                            traci.trafficlight.Phase(15 - green_time, "r")
                        ]
                    )
                    traci.trafficlight.setProgramLogic('meter', logic)
                    traci.trafficlight.setProgram('meter', "control")
                except:
                    pass
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        traci.close()
    
    # 统计
    print(f"\nStatistics:")
    print(f"  Max Queue: {max(data['ramp_queue'])} veh")
    print(f"  Avg Queue: {np.mean(data['ramp_queue']):.2f} veh")
    print(f"  Max Wait: {max(data['ramp_waiting']):.1f} s")
    print(f"  Avg Wait: {np.mean(data['ramp_waiting']):.1f} s")
    print(f"  Avg Merge Speed: {np.mean(data['merge_speed']):.2f} m/s")
    
    return data


def compare_scenarios():
    """对比不同场景"""
    
    print("\n" + "="*60)
    print("Ramp Merging Behavior Comparison")
    print("="*60)
    
    # 准备3个场景的route文件
    scenarios = {
        'Default (Unrealistic)': create_default_routes(),
        'Realistic': create_realistic_routes(),
        'With Control': create_realistic_routes()
    }
    
    results = {}
    
    # 场景1: 默认SUMO
    print("\n>>> Scenario 1: Default SUMO (总能插入)")
    results['default'] = test_merging_behavior(
        scenarios['Default (Unrealistic)'],
        'Default',
        with_control=False
    )
    
    # 场景2: 真实参数
    print("\n>>> Scenario 2: Realistic Parameters (保守汇入)")
    results['realistic'] = test_merging_behavior(
        scenarios['Realistic'],
        'Realistic',
        with_control=False
    )
    
    # 场景3: 有控制
    print("\n>>> Scenario 3: With Ramp Metering (有控制)")
    results['controlled'] = test_merging_behavior(
        scenarios['With Control'],
        'Controlled',
        with_control=True
    )
    
    # 绘图对比
    plot_comparison(results)
    
    print("\n✅ Comparison complete. Plots saved.\n")


def create_default_routes():
    """创建默认场景的route文件"""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="default_car" accel="2.6" decel="4.5" sigma="0.5" 
           length="5.0" minGap="2.5" maxSpeed="33.33" tau="1.0">
        <param key="lcStrategic" value="1.0"/>
        <param key="lcCooperative" value="1.0"/>
        <param key="lcAssertive" value="1.0"/>
    </vType>
    <route id="main_route" edges="main_1 main_2 main_3"/>
    <route id="ramp_route" edges="ramp_0 ramp_1 main_2 main_3"/>
    <flow id="main_flow" route="main_route" begin="0" end="3600" 
          vehsPerHour="3600" type="default_car" departLane="best" departSpeed="max"/>
    <flow id="ramp_flow" route="ramp_route" begin="0" end="3600" 
          vehsPerHour="1200" type="default_car" departSpeed="max"/>
</routes>"""
    
    filename = "/tmp/default_routes.rou.xml"
    with open(filename, 'w') as f:
        f.write(content)
    return filename


def create_realistic_routes():
    """创建真实场景的route文件"""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="realistic_car" accel="2.6" decel="4.5" sigma="0.5" 
           length="5.0" minGap="3.5" maxSpeed="33.33" tau="1.5">
        <param key="lcStrategic" value="0.5"/>
        <param key="lcCooperative" value="0.0"/>
        <param key="lcAssertive" value="0.3"/>
        <param key="lcSpeedGain" value="0.5"/>
    </vType>
    <route id="main_route" edges="main_1 main_2 main_3"/>
    <route id="ramp_route" edges="ramp_0 ramp_1 main_2 main_3"/>
    <flow id="main_flow" route="main_route" begin="0" end="3600" 
          vehsPerHour="3600" type="realistic_car" departLane="best" departSpeed="max"/>
    <flow id="ramp_flow" route="ramp_route" begin="0" end="3600" 
          vehsPerHour="1200" type="realistic_car" departSpeed="max"/>
</routes>"""
    
    filename = "/tmp/realistic_routes.rou.xml"
    with open(filename, 'w') as f:
        f.write(content)
    return filename


def plot_comparison(results):
    """绘制对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ramp Merging Behavior Comparison', fontsize=16, fontweight='bold')
    
    scenarios = list(results.keys())
    colors = ['blue', 'orange', 'green']
    labels = ['Default (Unrealistic)', 'Realistic', 'With Control']
    
    # 1. 匝道排队对比
    ax = axes[0, 0]
    for i, (key, data) in enumerate(results.items()):
        time_min = np.array(data['time']) / 60
        ax.plot(time_min, data['ramp_queue'], 
                label=labels[i], color=colors[i], linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Ramp Queue (veh)')
    ax.set_title('Ramp Queue Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 等待时间对比
    ax = axes[0, 1]
    for i, (key, data) in enumerate(results.items()):
        time_min = np.array(data['time']) / 60
        ax.plot(time_min, data['ramp_waiting'], 
                label=labels[i], color=colors[i], linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Waiting Time (s)')
    ax.set_title('Average Waiting Time on Ramp')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 合流区速度对比
    ax = axes[1, 0]
    for i, (key, data) in enumerate(results.items()):
        time_min = np.array(data['time']) / 60
        ax.plot(time_min, data['merge_speed'], 
                label=labels[i], color=colors[i], linewidth=2)
    ax.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Congestion Threshold')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Merge Area Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 统计对比柱状图
    ax = axes[1, 1]
    metrics = ['Max Queue', 'Avg Queue', 'Max Wait (s)']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (key, data) in enumerate(results.items()):
        values = [
            max(data['ramp_queue']),
            np.mean(data['ramp_queue']),
            max(data['ramp_waiting'])
        ]
        ax.bar(x + i * width, values, width, 
               label=labels[i], color=colors[i])
    
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/merging_comparison.png', 
                dpi=150, bbox_inches='tight')
    
    print("\n📊 Plot saved: /mnt/user-data/outputs/merging_comparison.png")


if __name__ == "__main__":
    compare_scenarios()
