import pandas as pd
import numpy as np

# ---------- 参数设置 ----------
input_path = "trajectories_with_angle(june).csv"  # 替换为你的输入文件路径
output_path = "ttc_lttb_final_split_by_type_yaw(june).csv"
a_max = 6.0  # 最大减速度 (m/s²)
yaw_threshold_min = 0.0
yaw_threshold_max = 5.0

# ---------- 数据加载与预处理 ----------
df = pd.read_csv(input_path)

# 构造 yaw_rate = Δangle / Δt（按车辆）
df = df.sort_values(by=["id", "time"])
df["yaw_rate"] = df.groupby("id")["angle"].diff() / df.groupby("id")["time"].diff()
df["yaw_rate"] = df["yaw_rate"].fillna(0)

# 提取 lane_index（如 "769108790_0" → 0）
df["lane_index"] = df["lane"].apply(lambda x: int(x.split("_")[-1]) if pd.notnull(x) else -1)

# ---------- 冲突检测 ----------
results = []

for time, frame in df.groupby("time"):
    frame = frame.copy()

    for ego_lane in sorted(frame["lane_index"].unique()):
        if ego_lane < 0:
            continue

        for lane_offset in [-1, 0, 1]:
            lane = ego_lane + lane_offset
            sub = frame[frame["lane_index"] == lane].sort_values("y")
            if len(sub) < 2:
                continue

            ego = sub.iloc[:-1]
            front = sub.iloc[1:]

            # 弧度转换
            ego_angle_rad = np.radians(ego["angle"].values)
            front_angle_rad = np.radians(front["angle"].values)

            # Y方向速度分量
            v_ego_y = ego["speed"].values * np.cos(ego_angle_rad)
            v_front_y = front["speed"].values * np.cos(front_angle_rad)

            # 相对速度 & 距离
            rel_speed_y = v_ego_y - v_front_y
            distance_y = front["y"].values - ego["y"].values

            # 偏航率和角度差
            yaw_rate_abs = np.abs(ego["yaw_rate"].values)
            angle_diff = np.abs(ego["angle"].values - front["angle"].values)

            # ---------- same_lane ----------
            if lane_offset == 0:
                valid = (rel_speed_y > 0.19) & (distance_y > 0) & (distance_y < 100)
                conflict_type = "same_lane"

            # ---------- lane_change_risk ----------
            elif lane_offset in [-1, 1]:
                valid = (
                    (rel_speed_y > 0.19) &
                    (distance_y > 0) & (distance_y < 100) &
                    (yaw_rate_abs > yaw_threshold_min) & (yaw_rate_abs < yaw_threshold_max)
                )
                conflict_type = "lane_change_risk"

            else:
                continue

            if valid.any():
                ttc = distance_y[valid] / rel_speed_y[valid]
                lttb = ((distance_y[valid] - (rel_speed_y[valid] ** 2) / (2 * a_max)) / rel_speed_y[valid]) + 1

                result = pd.DataFrame({
                    "time": time,
                    "ego_id": ego["id"].values[valid],
                    "front_id": front["id"].values[valid],
                    "lane_ego": ego["lane_index"].values[valid],
                    "lane_front": front["lane_index"].values[valid],
                    "distance_y": distance_y[valid],
                    "rel_speed_y": rel_speed_y[valid],
                    "angle_diff": angle_diff[valid],
                    "yaw_rate": yaw_rate_abs[valid],
                    "TTC": ttc,
                    "LTTB": lttb,
                    "type": conflict_type
                })

                results.append(result)

# 合并所有冲突
conflict_df = pd.concat(results, ignore_index=True)

# # # 移除极值（可选）
# conflict_df = conflict_df[(conflict_df["TTC"] < 100) & (conflict_df["LTTB"] < 100)]

# 保存为 CSV
conflict_df.to_csv(output_path, index=False)
