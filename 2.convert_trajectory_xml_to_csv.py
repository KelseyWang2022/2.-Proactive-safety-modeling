import xml.etree.ElementTree as ET
import pandas as pd

# 读取 vehicle_trajectories.xml
tree = ET.parse("1.simulation/fcd_output.xml")
root = tree.getroot()

# 存储提取的数据
records = []

# 遍历每个 timestep 和其下的车辆
for timestep in root.findall("timestep"):
    time = float(timestep.attrib["time"])
    for vehicle in timestep.findall("vehicle"):
        record = {
            "time": time,
            "id": vehicle.attrib.get("id"),
            "x": float(vehicle.attrib.get("x", 0)),
            "y": float(vehicle.attrib.get("y", 0)),
            "angle": float(vehicle.attrib.get("angle", 0)),
            "speed": float(vehicle.attrib.get("speed", 0)),
            "lane": vehicle.attrib.get("lane"),
            "type": vehicle.attrib.get("type")
        }
        records.append(record)

# 转换为 DataFrame
df = pd.DataFrame(records)

# 保存为 CSV 文件
output_path = "trajectories_with_angle(june).csv"
df.to_csv(output_path, index=False)

