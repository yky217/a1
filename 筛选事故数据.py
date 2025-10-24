import pandas as pd

# === 文件路径 ===
file_path = r"F:\大三上\大数据\期初作业\US_Accidents_March23.csv"
output_path = r"F:\大三上\大数据\期初作业\US_Accidents_CA_2021_2022.csv"

# === 读取数据 ===
print("正在加载数据，请稍候...")
df = pd.read_csv(file_path, parse_dates=["Start_Time"], low_memory=False)

# === 筛选加州数据 ===
df = df[df["State"] == "CA"]

# === 筛选 2021 和 2022 年数据 ===
mask = (df["Start_Time"] >= "2021-01-01") & (df["Start_Time"] < "2023-01-01")
df = df[mask]

# === 输出基本信息 ===
print(f"筛选后记录数：{len(df):,}")
print(f"字段总数：{len(df.columns)}")
print("\n数据样例：")
print(df.head(3))

# === 保存结果 ===
df.to_csv(output_path, index=False)
print(f"\n✅ 清洗完成！文件已保存到：{output_path}")
