# 重新执行前面的代码块以生成数据和图表

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# 构造时间序列（一天24小时，每半小时一个点）
time_points = [datetime(2024, 1, 1, 0, 0) + timedelta(minutes=30*i) for i in range(48)]

# 生成模拟客流数据：早高峰（7:00-9:00）和晚高峰（17:00-19:00）
flow = []
for t in time_points:
    hour = t.hour + t.minute / 60
    # 高峰期模拟
    if 7 <= hour <= 9:
        f = np.random.normal(loc=300, scale=20)  # 早高峰
    elif 17 <= hour <= 19:
        f = np.random.normal(loc=320, scale=25)  # 晚高峰
    elif 11 <= hour <= 13:
        f = np.random.normal(loc=180, scale=15)  # 午间适中
    else:
        f = np.random.normal(loc=80, scale=10)  # 非高峰期
    flow.append(max(0, f))  # 确保非负

# 构建 DataFrame
df = pd.DataFrame({
    'Time': time_points,
    'Traffic Flow': flow
})

# 绘图
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Time', y='Traffic Flow', marker='o')
plt.title('Traffic Flow Over Time at a Station', fontsize=14)
plt.xlabel('Time of Day')
plt.ylabel('Traffic Flow')
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.tight_layout()

plt.show()
