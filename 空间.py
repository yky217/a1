import pandas as pd
df=pd.read_csv('USAccidents.csv')
print(f"数据集形状: {df.shape}")
print(f"数据列名: {df.columns.tolist()}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv('USAccidents.csv')
    original_shape = df.shape
    print(f"原始数据集形状: {original_shape}")
    
    # 转换时间列
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # 剔除时间格式不一致的行
    df_clean = df.dropna(subset=['Start_Time', 'End_Time']).copy()
    print(f"有效数据行数: {len(df_clean)}")
    
    # 创建严重事故标志
    df_clean['Severity_flag'] = (df_clean['Severity'] == 1).astype(int)
    
    return df_clean

def create_spatial_bar_charts(df):
    """创建空间分布的柱状图"""
    print("\n正在生成空间分布柱状图...")
    
    # 1. 城市分布柱状图
    if 'City' in df.columns:
        city_accidents = df['City'].value_counts().head(20)
        
        plt.figure(figsize=(16, 10))
        bars_city = plt.bar(range(len(city_accidents)), city_accidents.values, 
                           color='lightgreen', alpha=0.7, edgecolor='black')
        
        # 在柱子顶部添加数值标注
        for i, bar in enumerate(bars_city):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(city_accidents.values)*0.005,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('城市')
        plt.ylabel('事故数量')
        plt.title('事故数量前20城市分布（柱顶数字为事故数量）')
        plt.xticks(range(len(city_accidents)), city_accidents.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        print("城市分布柱状图生成完成")
    
    # 2. 县分布柱状图
    if 'County' in df.columns:
        county_accidents = df['County'].value_counts().head(15)
        
        plt.figure(figsize=(14, 8))
        bars_county = plt.bar(range(len(county_accidents)), county_accidents.values, 
                             color='lightblue', alpha=0.7, edgecolor='black')
        
        # 在柱子顶部添加数值标注
        for i, bar in enumerate(bars_county):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(county_accidents.values)*0.005,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('县')
        plt.ylabel('事故数量')
        plt.title('事故数量前15县分布（柱顶数字为事故数量）')
        plt.xticks(range(len(county_accidents)), county_accidents.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        print("县分布柱状图生成完成")

def create_heatmap(df):
    """创建地理热力图"""
    print("\n正在生成地理热力图...")
    
    # 检查必要的经纬度列
    if 'Start_Lat' not in df.columns or 'Start_Lng' not in df.columns:
        print("错误：缺少经纬度数据")
        return
    
    # 过滤掉无效的经纬度
    spatial_df = df.dropna(subset=['Start_Lat', 'Start_Lng']).copy()
    spatial_df = spatial_df[
        (spatial_df['Start_Lat'].between(-90, 90)) & 
        (spatial_df['Start_Lng'].between(-180, 180))
    ]
    
    print(f"有效经纬度数据点: {len(spatial_df)}")
    
    if len(spatial_df) == 0:
        print("错误：没有有效的经纬度数据")
        return
    
    # 计算中心点
    center_lat = spatial_df['Start_Lat'].mean()
    center_lng = spatial_df['Start_Lng'].mean()
    
    # 创建基础地图
    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # 准备热力图数据
    heat_data = [[row['Start_Lat'], row['Start_Lng']] for _, row in spatial_df.iterrows()]
    
    # 添加热力图
    HeatMap(
        heat_data,
        radius=15,
        blur=10,
        max_zoom=13,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)
    
    # 保存热力图
    m.save('accidents_heatmap.html')
    print("地理热力图已保存为 'accidents_heatmap.html'")
    
    return m

def create_cluster_map(df):
    """创建聚类标记地图"""
    print("\n正在生成聚类标记地图...")
    
    if 'Start_Lat' not in df.columns or 'Start_Lng' not in df.columns:
        print("错误：缺少经纬度数据")
        return
    
    # 过滤掉无效的经纬度
    spatial_df = df.dropna(subset=['Start_Lat', 'Start_Lng']).copy()
    spatial_df = spatial_df[
        (spatial_df['Start_Lat'].between(-90, 90)) & 
        (spatial_df['Start_Lng'].between(-180, 180))
    ]
    
    # 采样以避免过多的点（如果需要）
    if len(spatial_df) > 10000:
        spatial_df = spatial_df.sample(10000, random_state=42)
        print(f"采样到10000个点进行聚类显示")
    
    # 计算中心点
    center_lat = spatial_df['Start_Lat'].mean()
    center_lng = spatial_df['Start_Lng'].mean()
    
    # 创建聚类地图
    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=6
    )
    
    # 添加聚类标记
    marker_cluster = MarkerCluster().add_to(m)
    
    # 添加标记点
    for _, row in spatial_df.iterrows():
        popup_text = f"""
        城市: {row.get('City', 'N/A')}<br>
        严重程度: {row.get('Severity', 'N/A')}<br>
        时间: {row.get('Start_Time', 'N/A')}<br>
        天气: {row.get('Weather_Condition', 'N/A')}
        """
        
        folium.Marker(
            location=[row['Start_Lat'], row['Start_Lng']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(
                color='red' if row.get('Severity_flag') == 1 else 'orange',
                icon='info-sign'
            )
        ).add_to(marker_cluster)
    
    # 保存聚类地图
    m.save('accidents_cluster_map.html')
    print("聚类标记地图已保存为 'accidents_cluster_map.html'")
    
    return m

def create_severity_spatial_analysis(df):
    """创建严重程度空间分析"""
    print("\n正在生成严重程度空间分析...")
    
    # 1. 各城市严重事故比例
    if 'City' in df.columns:
        city_severity = df.groupby('City').agg({
            'ID': 'count',
            'Severity_flag': 'mean'
        }).rename(columns={'ID': '总事故数', 'Severity_flag': '严重事故比例'})
        
        # 选取事故数量前15的城市
        top_cities = city_severity.nlargest(15, '总事故数')
        
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # 左轴：总事故数（柱状图）
        bars = ax1.bar(range(len(top_cities)), top_cities['总事故数'], 
                      color='lightblue', alpha=0.7, label='总事故数')
        ax1.set_xlabel('城市')
        ax1.set_ylabel('总事故数量', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 在柱子上添加总事故数
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(top_cities['总事故数'])*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9, color='blue')
        
        # 右轴：严重事故比例（折线图）
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(top_cities)), top_cities['严重事故比例'], 
                       marker='o', color='red', linewidth=2, markersize=6, label='严重事故比例')
        ax2.set_ylabel('严重事故比例', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        # 在折线点上添加比例
        for i, ratio in enumerate(top_cities['严重事故比例']):
            ax2.text(i, ratio + 0.02, f'{ratio:.1%}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, color='red')
        
        plt.title('前15城市事故数量与严重事故比例分析')
        plt.xticks(range(len(top_cities)), top_cities.index, rotation=45, ha='right')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
def generate_spatial_report(df):
    """生成空间分析报告"""
    print("\n" + "="*60)
    print("空间分析报告")
    print("="*60)
    
    print(f"总事故数量: {len(df):,}")
    
    # 城市统计
    if 'City' in df.columns:
        city_stats = df['City'].value_counts()
        print(f"\n涉及城市数量: {len(city_stats)}")
        print(f"事故最多城市: {city_stats.index[0]} ({city_stats.iloc[0]:,}起)")
        print(f"前5城市事故占比: {city_stats.head(5).sum()/len(df):.1%}")
    
    # 县统计
    if 'County' in df.columns:
        county_stats = df['County'].value_counts()
        print(f"\n涉及县数量: {len(county_stats)}")
        print(f"事故最多县: {county_stats.index[0]} ({county_stats.iloc[0]:,}起)")
    
    # 经纬度统计
    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
        valid_coords = df.dropna(subset=['Start_Lat', 'Start_Lng'])
        print(f"\n有效经纬度数据点: {len(valid_coords):,}")
        print(f"经纬度覆盖率: {len(valid_coords)/len(df):.1%}")
        
        # 地理范围
        print(f"纬度范围: {valid_coords['Start_Lat'].min():.4f} - {valid_coords['Start_Lat'].max():.4f}")
        print(f"经度范围: {valid_coords['Start_Lng'].min():.4f} - {valid_coords['Start_Lng'].max():.4f}")
    

def main():
    """主函数"""
    print("开始空间分析...")
    
    # 1. 加载数据
    df = load_and_preprocess_data()
    
    # 2. 生成空间分布柱状图
    create_spatial_bar_charts(df)
    
    # 3. 生成地理热力图
    create_heatmap(df)
    
    # 4. 生成聚类标记地图
    create_cluster_map(df)
    
    # 5. 生成严重程度空间分析
    create_severity_spatial_analysis(df)
    
    # 6. 生成分析报告
    generate_spatial_report(df)
    
    print("\n空间分析完成！")
    print("生成的文件:")
    print("  - accidents_heatmap.html (热力图)")
    print("  - accidents_cluster_map.html (聚类地图)")

if __name__ == "__main__":
    main()