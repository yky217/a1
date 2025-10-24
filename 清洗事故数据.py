import pandas as pd

# 加载清洗后的加州数据
df = pd.read_csv(r"F:\大三上\大数据\期初作业\US_Accidents_CA_2021_2022.csv")

# 选择需要的字段
keep_cols = [
    'ID', 'Severity',
    'Start_Time', 'End_Time', 'Weather_Timestamp',
    'Start_Lat', 'Start_Lng', 'City', 'County',
    'Distance(mi)', 'Crossing', 'Junction', 'Traffic_Signal', 'Bump',
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition'
]

df_filtered = df[keep_cols]

# 保存清洗后的文件
output_path = r"C:\Users\20531\Desktop\US_Accidents_CA_2021_2022_filtered.csv"
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"筛选完成，共保留字段 {len(keep_cols)} 个，输出文件：{output_path}")
print(df_filtered.head(3))
