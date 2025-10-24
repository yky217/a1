import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# ---- 配置 ----
accident_file = r"F:\大三上\大数据\期初作业\US_Accidents_CA_2021_2022_filtered.csv"
weather_file = r"F:\大三上\大数据\期初作业\NOAA_CA_2021_2022_clean.csv"
output_file = r"F:\大三上\大数据\期初作业\Accidents_with_Weather_final_kdtree.csv"
chunk_size = 50000
max_dist_m = 20000  # 最大匹配距离：20km

# ---- 1. 加载气象数据 ----
print("加载气象数据...")
weather = pd.read_csv(weather_file)
cols = ['STATION', 'LATITUDE', 'LONGITUDE', 'DATE', 'PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNOW', 'SNWD', 'AWND']
weather = weather[[c for c in cols if c in weather.columns]]
weather['DATE'] = pd.to_datetime(weather['DATE'], errors='coerce')
weather.rename(columns={'LATITUDE': 'Weather_Lat', 'LONGITUDE': 'Weather_Lng'}, inplace=True)
print(f"气象数据记录数: {len(weather)}")

# ---- 经纬度 → 笛卡尔坐标函数 ----
def latlng_to_cartesian(lat, lng):
    R = 6371000  # 地球半径
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    x = R * np.cos(lat_rad) * np.cos(lng_rad)
    y = R * np.cos(lat_rad) * np.sin(lng_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))

# ---- 2. 分块处理事故数据 ----
reader = pd.read_csv(accident_file, chunksize=chunk_size)
processed_chunks = []

for i, chunk in enumerate(reader, 1):
    print(f"开始处理第 {i} 个数据块...")

    # --- 时间字段处理 ---
    chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'], errors='coerce')
    chunk['Start_Date'] = chunk['Start_Time'].dt.date
    chunk['Hour'] = chunk['Start_Time'].dt.hour
    chunk['Weekday'] = chunk['Start_Time'].dt.weekday
    chunk['Severity_flag'] = (chunk['Severity'] >= 3).astype(int)

    # --- ✅ 布尔字段 TRUE/FALSE 转换为 1/0 ---
    bool_cols = ['Crossing', 'Junction', 'Traffic_Signal', 'Bump']
    for col in bool_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    # --- 按日期匹配气象数据 ---
    chunk_result_list = []
    for date, sub_chunk in chunk.groupby('Start_Date'):
        weather_day = weather[weather['DATE'].dt.date == date]
        if weather_day.empty:
            sub_chunk[['PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNOW', 'SNWD', 'AWND',
                       'Weather_Distance', 'Weather_STATION']] = np.nan
            chunk_result_list.append(sub_chunk)
            continue

        acc_xyz = latlng_to_cartesian(sub_chunk['Start_Lat'].values, sub_chunk['Start_Lng'].values)
        wea_xyz = latlng_to_cartesian(weather_day['Weather_Lat'].values, weather_day['Weather_Lng'].values)
        tree = cKDTree(wea_xyz)
        dist, idx = tree.query(acc_xyz, k=1)
        mask = dist <= max_dist_m

        matched_weather = weather_day.iloc[idx].reset_index(drop=True)
        for col in ['PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNOW', 'SNWD', 'AWND', 'STATION']:
            sub_chunk[col] = np.where(mask, matched_weather[col], np.nan)
        sub_chunk['Weather_Distance'] = np.where(mask, dist, np.nan)
        sub_chunk['Weather_STATION'] = np.where(mask, matched_weather['STATION'], np.nan)
        chunk_result_list.append(sub_chunk)

    merged_chunk = pd.concat(chunk_result_list, ignore_index=True)
    processed_chunks.append(merged_chunk)
    print(f"块 {i} 已处理完成，共 {len(merged_chunk)} 条记录")

# ---- 3. 合并所有块 ----
print("正在合并所有数据块...")
result = pd.concat(processed_chunks, ignore_index=True)

# ---- ✅ 对连续变量进行 Z-score 标准化 ----
continuous_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                   'Wind_Speed(mph)', 'Precipitation(in)', 'PRCP', 'TAVG', 'TMAX', 'TMIN',
                   'SNOW', 'SNWD', 'AWND']

existing_cont_cols = [c for c in continuous_cols if c in result.columns]
scaler = StandardScaler()

# 仅对非缺失值部分标准化
result[existing_cont_cols] = scaler.fit_transform(result[existing_cont_cols].fillna(result[existing_cont_cols].mean()))

print("已完成布尔转换与连续变量标准化。")

# ---- 4. 保存结果 ----
result.to_csv(output_file, index=False)
print("✅ 数据清洗与融合全部完成，文件已保存至：", output_file)
