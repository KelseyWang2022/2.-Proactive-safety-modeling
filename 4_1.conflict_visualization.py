import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

ttc_df = pd.read_csv(r"ttc_lttb_final_split_by_type_yaw(june).csv")
#统计LTTB值小于5的数据
lttb_values_count = len(ttc_df[ttc_df["LTTB"] < 5])
print(f"Number of LTTB values less than 5: {lttb_values_count}")
# Filter the data to only include TTC and LTTB values less than 100
ttc_df = ttc_df[(ttc_df["TTC"] < 5) & (ttc_df["LTTB"] < 5)]


plt.figure(figsize=(10, 5))
sns.histplot(ttc_df["TTC"], bins=50, kde=True, color='skyblue')
plt.title("Distribution of TTC values (Filtered)")
plt.xlabel("Time To Collision (s)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(ttc_df["LTTB"], bins=50, kde=True, color='orange')
plt.title("Distribution of LTTB values (Filtered)")
plt.xlabel("Latest Time To Brake (s)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


