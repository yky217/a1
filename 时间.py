import pandas as pd
df=pd.read_csv('USAccidents.csv')
print(f"数据集形状: {df.shape}")
print(f"数据列名: {df.columns.tolist()}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# 2. 数据预处理
# 转换时间列，错误时设为NaN
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')

# 检查转换失败的行数
start_time_failed = df['Start_Time'].isna().sum()
end_time_failed = df['End_Time'].isna().sum()
weather_time_failed = df['Weather_Timestamp'].isna().sum()

print(f"Start_Time转换失败: {start_time_failed} 行")
print(f"End_Time转换失败: {end_time_failed} 行")
print(f"Weather_Timestamp转换失败: {weather_time_failed} 行")

# 剔除任何时间列转换失败的行
df_clean = df.dropna(subset=['Start_Time', 'End_Time', 'Weather_Timestamp']).copy()

# 如果数据量太少，停止分析
if len(df_clean) == 0:
    print("错误：所有数据都被剔除了，请检查数据格式")
    exit()

# 3. 提取时间特征 - 添加错误检查
print("\n正在提取时间特征...")
print(f"Start_Time数据类型: {df_clean['Start_Time'].dtype}")

# 确保Start_Time是datetime类型
if df_clean['Start_Time'].dtype == 'object':
    print("Start_Time仍然是object类型，尝试重新转换...")
    df_clean['Start_Time'] = pd.to_datetime(df_clean['Start_Time'], errors='coerce')
    # 再次删除转换失败的行
    df_clean = df_clean.dropna(subset=['Start_Time'])

# 现在提取时间特征
try:
    df_clean['Hour'] = df_clean['Start_Time'].dt.hour
    df_clean['DayOfWeek'] = df_clean['Start_Time'].dt.dayofweek
    df_clean['Month'] = df_clean['Start_Time'].dt.month
    df_clean['Year'] = df_clean['Start_Time'].dt.year
    
    print("时间特征提取成功!")
    print(f"Hour列前5个值: {df_clean['Hour'].head().tolist()}")
    print(f"数据列名: {df_clean.columns.tolist()}")
    
except Exception as e:
    print(f"提取时间特征时出错: {e}")
    print("检查Start_Time列的前几个值:")
    print(df_clean['Start_Time'].head())
    exit()

# 创建严重事故标志 (假设Severity=1为严重事故)
df_clean['Severity_flag'] = (df_clean['Severity'] == 1).astype(int)

print("数据预处理完成!")

# 4. 时间维度分析 - 带数据标注的图表
print("\n正在进行时间维度分析...")

# 4.1 按小时统计事故数量
hourly_accidents = df_clean.groupby('Hour').size()
hourly_severe = df_clean[df_clean['Severity_flag'] == 1].groupby('Hour').size()
hourly_non_severe = df_clean[df_clean['Severity_flag'] == 0].groupby('Hour').size()

print(f"小时统计完成，共{len(hourly_accidents)}个小时段")

# 4.2 按星期统计事故数量
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
daily_accidents = df_clean.groupby('DayOfWeek').size()
daily_accidents.index = weekday_names

# 4.3 按月份统计事故数量
month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
monthly_accidents = df_clean.groupby('Month').size()
# 确保月份索引在有效范围内
monthly_accidents = monthly_accidents.reindex(range(1, 13), fill_value=0)
monthly_accidents.index = month_names

# 绘制时间分布图 - 所有图表都添加数据标注
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. 小时分布折线图 - 添加数据点标注
if len(hourly_accidents) > 0:
    axes[0, 0].plot(hourly_accidents.index, hourly_accidents.values, marker='o', linewidth=2, 
                    label='总事故', color='blue', markersize=6)
    
    # 为每个数据点添加数值标注
    for hour, count in hourly_accidents.items():
        axes[0, 0].annotate(f'{count}', (hour, count), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, color='blue')
    
    axes[0, 0].set_xlabel('小时')
    axes[0, 0].set_ylabel('事故数量')
    axes[0, 0].set_title('事故数量按小时分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 2))
else:
    axes[0, 0].text(0.5, 0.5, '无小时数据', ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('事故数量按小时分布（无数据）')

# 2. 星期分布柱状图 - 在柱子顶部添加数值
if len(daily_accidents) > 0:
    bars1 = axes[0, 1].bar(daily_accidents.index, daily_accidents.values, 
                          color='skyblue', alpha=0.7, edgecolor='black')
    
    # 在柱子顶部添加数值标注
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(daily_accidents.values)*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    axes[0, 1].set_xlabel('星期')
    axes[0, 1].set_ylabel('事故数量')
    axes[0, 1].set_title('事故数量按星期分布')
    axes[0, 1].tick_params(axis='x', rotation=45)
else:
    axes[0, 1].text(0.5, 0.5, '无星期数据', ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('事故数量按星期分布（无数据）')

# 3. 月份分布柱状图 - 在柱子顶部添加数值
if len(monthly_accidents) > 0:
    bars2 = axes[1, 0].bar(monthly_accidents.index, monthly_accidents.values, 
                          color='lightcoral', alpha=0.7, edgecolor='black')
    
    # 在柱子顶部添加数值标注
    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(monthly_accidents.values)*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 0].set_xlabel('月份')
    axes[1, 0].set_ylabel('事故数量')
    axes[1, 0].set_title('事故数量按月份分布')
    axes[1, 0].tick_params(axis='x', rotation=45)
else:
    axes[1, 0].text(0.5, 0.5, '无月份数据', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('事故数量按月份分布（无数据）')

# 4. 小时-星期热力图 - 在每个格子中添加数值
try:
    hour_day_data = df_clean.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
    hour_day_data.index = weekday_names[:len(hour_day_data)]
    
    im = axes[1, 1].imshow(hour_day_data.values, cmap='YlOrRd', aspect='auto')
    
    # 在每个热力图格子上添加数值
    for i in range(len(hour_day_data.index)):
        for j in range(len(hour_day_data.columns)):
            text = axes[1, 1].text(j, i, f'{hour_day_data.iloc[i, j]}',
                                  ha="center", va="center", color="black", fontsize=8,
                                  fontweight='bold')
    
    axes[1, 1].set_xlabel('小时')
    axes[1, 1].set_ylabel('星期')
    axes[1, 1].set_title('事故数量小时-星期热力图（数值表示事故数量）')
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].set_yticks(range(len(hour_day_data.index)))
    axes[1, 1].set_yticklabels(hour_day_data.index)
    plt.colorbar(im, ax=axes[1, 1], label='事故数量')
except Exception as e:
    axes[1, 1].text(0.5, 0.5, f'热力图生成失败: {e}', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('事故数量小时-星期热力图（生成失败）')

plt.tight_layout()
plt.show()

# 7. 生成分析报告
print("\n" + "="*60)
print("分析报告摘要")
print("="*60)

print(f"总事故数量: {len(df_clean):,}")
print(f"严重事故数量: {df_clean['Severity_flag'].sum():,}")
print(f"严重事故比例: {df_clean['Severity_flag'].mean():.2%}")

# 时间分析结果
if len(hourly_accidents) > 0:
    max_hour = hourly_accidents.idxmax()
    print(f"\n事故最高发时段: {max_hour}:00-{max_hour+1}:00 (数量: {hourly_accidents[max_hour]:,})")

if len(daily_accidents) > 0:
    max_day = daily_accidents.idxmax()
    print(f"事故最高发星期: {max_day} (数量: {daily_accidents[max_day]:,})")

if len(monthly_accidents) > 0:
    max_month = monthly_accidents.idxmax()
    print(f"事故最高发月份: {max_month} (数量: {monthly_accidents[max_month]:,})")

print("\n所有图表已生成完成！")