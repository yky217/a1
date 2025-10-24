import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    try:
        file_path="AccidentswithWeather.csv"
        df = pd.read_csv(file_path)
        print(f"数据加载成功！数据形状: {df.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 数据预处理
    time_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp', 'Start_Date']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 提取日期信息
    df['Start_Date_only'] = df['Start_Time'].dt.date
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Month'] = df['Start_Time'].dt.month
    
    return df

def create_weather_severity_table(df):
    """创建天气类型与事故严重性的统计表格"""
    print("\n" + "="*60)
    print("天气类型与事故严重性统计表格")
    print("="*60)
    
    if 'Weather_Condition' in df.columns and 'Severity_flag' in df.columns:
        # 统计各天气类型的事故数量和严重性比例
        weather_stats = df.groupby('Weather_Condition').agg({
            'ID': 'count',
            'Severity_flag': ['mean', 'sum']
        }).round(4)
        
        # 重命名列
        weather_stats.columns = ['Accident_Count', 'Severity_Flag_Ratio', 'Severity_Flag_Count']
        weather_stats = weather_stats.sort_values('Accident_Count', ascending=False)
        
        # 计算百分比
        total_accidents = len(df)
        weather_stats['Percentage'] = (weather_stats['Accident_Count'] / total_accidents * 100).round(2)
        
        print("\n各天气类型事故统计表:")
        print(weather_stats)
        
        return weather_stats
    else:
        print("缺少 Weather_Condition 或 Severity_flag 列")
        return None

def plot_weather_severity_dual_axis(df):
    """绘制天气类型与事故数量和严重性比例的双Y轴图"""
    print("\n" + "="*60)
    print("天气类型与事故严重性双Y轴分析")
    print("="*60)
    
    if 'Weather_Condition' in df.columns and 'Severity_flag' in df.columns:
        # 统计各天气类型的事故数量
        weather_stats = df.groupby('Weather_Condition').agg({
            'ID': 'count',
            'Severity_flag': 'mean'
        }).round(4)
        weather_stats.columns = ['Accident_Count', 'Severity_Flag_Ratio']
        weather_stats = weather_stats.sort_values('Accident_Count', ascending=False)
        
        print("\n各天气类型事故统计:")
        print(weather_stats)
        
        # 绘制天气类型与事故数量图
        plt.figure(figsize=(12, 6))
        top_weather = weather_stats.head(10)  # 显示前10个天气类型
        
        # 创建双Y轴图
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # 左侧Y轴：事故数量
        color = 'tab:blue'
        bars = ax1.bar(range(len(top_weather)), top_weather['Accident_Count'], 
                      color=color, alpha=0.7, label='事故数量')
        ax1.set_xlabel('天气类型', fontsize=12)
        ax1.set_ylabel('事故数量', color=color, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(range(len(top_weather)))
        ax1.set_xticklabels(top_weather.index, rotation=45, ha='right')
        
        # 右侧Y轴：严重性比例
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.plot(range(len(top_weather)), top_weather['Severity_Flag_Ratio'], 
                color=color, marker='o', linewidth=2, markersize=8, label='严重事故比例')
        ax2.set_ylabel('严重事故比例', color=color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('各天气类型下事故数量与严重性比例分析', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Weather_Severity_Dual_Axis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("已生成图表: Weather_Severity_Dual_Axis.png")
        
        # 输出前10个天气类型的详细统计
        print(f"\n前10个天气类型详细统计:")
        for i, (weather, row) in enumerate(top_weather.iterrows(), 1):
           print(f"{i:2d}. {weather:15s} 事故数: {int(row['Accident_Count']):6d} 严重比例: {row['Severity_Flag_Ratio']:.4f}")
    else:
        print("缺少 Weather_Condition 或 Severity_flag 列")

def plot_individual_boxplots(df):
    """为每个连续气象变量单独生成箱线图"""
    print("\n" + "="*60)
    print("连续气象变量与事故严重性箱线图分析")
    print("="*60)
    
    # 定义连续气象变量及其中文名称
    continuous_vars = {
        'Temperature(F)': '温度(标准化)',
        'PRCP': '降水量(标准化)', 
        'Wind_Speed(mph)': '风速(标准化)',
        'Visibility(mi)': '能见度(标准化)',
        'Humidity(%)': '湿度(标准化)',
        'Pressure(in)': '气压(标准化)'
    }
    
    # 检查哪些变量存在
    available_vars = [var for var in continuous_vars.keys() if var in df.columns]
    print(f"可用的连续气象变量: {available_vars}")
    
    if not available_vars:
        return
    
    # 为每个变量单独生成箱线图
    for var in available_vars:
        plt.figure(figsize=(10, 6))
        df_clean = df.dropna(subset=[var, 'Severity'])
        
        if len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='Severity', y=var)
            plt.title(f'{continuous_vars[var]}与事故严重性关系箱线图', fontsize=14, fontweight='bold')
            plt.xlabel('事故严重等级', fontsize=12)
            plt.ylabel(continuous_vars[var], fontsize=12)
            
            # 添加统计信息
            stats_text = f"总样本数: {len(df_clean)}\n"
            for severity in sorted(df_clean['Severity'].unique()):
                subset = df_clean[df_clean['Severity'] == severity][var]
                stats_text += f"等级{severity}: 中位数={subset.median():.3f}\n"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            plt.savefig(f'{var}_vs_Severity_Boxplot.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"已生成图表: {var}_vs_Severity_Boxplot.png")
        else:
            print(f"变量 {var} 无有效数据")

def analyze_weather_distance_impact(df):
    """分析气象站距离对事故的影响（单独图表）"""
    print("\n" + "="*60)
    print("气象站距离影响分析")
    print("="*60)
    
    if 'Weather_Distance' in df.columns:
        # 图表1: 距离分布直方图
        plt.figure(figsize=(12, 6))
        plt.hist(df['Weather_Distance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('气象站距离分布直方图', fontsize=14, fontweight='bold')
        plt.xlabel('与气象站距离(米)', fontsize=12)
        plt.ylabel('事故数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Weather_Distance_Distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("已生成图表: Weather_Distance_Distribution.png")
        
        # 图表2: 距离与严重等级散点图
        plt.figure(figsize=(12, 6))
        scatter_data = df.dropna(subset=['Weather_Distance', 'Severity'])
        sns.scatterplot(data=scatter_data, x='Weather_Distance', y='Severity', 
                       alpha=0.6, s=60)
        plt.title('事故严重等级与气象站距离关系', fontsize=14, fontweight='bold')
        plt.xlabel('与气象站距离(米)', fontsize=12)
        plt.ylabel('事故严重等级', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Severity_vs_Weather_Distance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("已生成图表: Severity_vs_Weather_Distance.png")
        
        # 图表3: 不同天气类型的距离分布
        plt.figure(figsize=(14, 8))
        # 选择出现频率较高的天气类型
        weather_counts = df['Weather_Condition'].value_counts()
        common_weather = weather_counts[weather_counts >= 3].index
        
        if len(common_weather) > 0:
            filtered_data = df[df['Weather_Condition'].isin(common_weather)]
            sns.boxplot(data=filtered_data, x='Weather_Condition', y='Weather_Distance')
            plt.title('不同天气类型下与气象站距离分布', fontsize=14, fontweight='bold')
            plt.xlabel('天气类型', fontsize=12)
            plt.ylabel('与气象站距离(米)', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('Weather_Distance_by_Weather_Type.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("已生成图表: Weather_Distance_by_Weather_Type.png")
        
        # 输出距离统计信息
        print(f"\n气象站距离统计:")
        print(f"平均距离: {df['Weather_Distance'].mean():.2f} 米")
        print(f"距离中位数: {df['Weather_Distance'].median():.2f} 米")
        print(f"距离标准差: {df['Weather_Distance'].std():.2f} 米")
        print(f"最大距离: {df['Weather_Distance'].max():.2f} 米")
        print(f"最小距离: {df['Weather_Distance'].min():.2f} 米")
        
    else:
        print("缺少 Weather_Distance 列")

def analyze_accident_density(df):
    """分析事故时空密度（单独图表）"""
    print("\n" + "="*60)
    print("事故时空密度分析")
    print("="*60)
    
    if all(col in df.columns for col in ['Start_Date_only', 'City', 'Weather_Condition']):
        # 计算每日每城市的事故数量
        daily_city_accidents = df.groupby(['Start_Date_only', 'City']).agg({
            'ID': 'count',
            'Severity': 'mean'
        }).reset_index()
        daily_city_accidents.columns = ['Start_Date_only', 'City', 'Accident_Count', 'Avg_Severity']
        
        # 合并回原始数据
        df_with_density = df.merge(
            daily_city_accidents[['Start_Date_only', 'City', 'Accident_Count']],
            on=['Start_Date_only', 'City'],
            how='left'
        )
        
        # 图表2: 天气类型与事故密度关系
        plt.figure(figsize=(14, 8))
        weather_density = df_with_density.groupby('Weather_Condition')['Accident_Count'].mean().sort_values(ascending=False)
        
        # 选择事故数量较多的天气类型
        weather_counts = df_with_density['Weather_Condition'].value_counts()
        common_weather = weather_counts[weather_counts >= 3].index
        
        if len(common_weather) > 0:
            filtered_weather_density = weather_density[weather_density.index.isin(common_weather)]
            filtered_weather_density.plot(kind='bar', color='lightgreen', alpha=0.7)
            plt.title('各天气类型下平均事故密度', fontsize=14, fontweight='bold')
            plt.xlabel('天气类型', fontsize=12)
            plt.ylabel('平均事故密度', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('Accident_Density_by_Weather_Type.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("已生成图表: Accident_Density_by_Weather_Type.png")

def generate_comprehensive_report(df, weather_stats):
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("综合分析报告")
    print("="*60)
    
    # 基本统计信息
    print(f"\n1. 数据概况:")
    print(f"   - 总事故数: {len(df)}")
    print(f"   - 时间范围: {df['Start_Time'].min()} 至 {df['Start_Time'].max()}")
    print(f"   - 涉及城市数: {df['City'].nunique()}")
    print(f"   - 天气类型数: {df['Weather_Condition'].nunique()}")
    
    # 严重等级分布
    print(f"\n2. 事故严重等级分布:")
    severity_dist = df['Severity'].value_counts().sort_index()
    for level, count in severity_dist.items():
        percentage = (count / len(df) * 100)
        print(f"   - 等级{level}: {count} 起 ({percentage:.1f}%)")
    
    # 天气类型分析
    if weather_stats is not None:
        print(f"\n3. 主要天气类型影响:")
        top_weather = weather_stats.head(5)
        for idx, (weather, row) in enumerate(top_weather.iterrows(), 1):
            print(f"   {idx}. {weather}: {row['Accident_Count']} 起事故, "
                  f"严重事故比例: {row['Severity_Flag_Ratio']*100:.1f}%")
    
    # 气象站距离分析
    if 'Weather_Distance' in df.columns:
        print(f"\n4. 气象站距离分析:")
        print(f"   - 平均距离: {df['Weather_Distance'].mean():.0f} 米")
        print(f"   - 距离中位数: {df['Weather_Distance'].median():.0f} 米")
        print(f"   - 距离范围: {df['Weather_Distance'].min():.0f} - {df['Weather_Distance'].max():.0f} 米")

def main():
    """主函数"""
    # 加载数据
    file_path = "工作簿1.xlsx"  # 修改为您的文件路径
    df = load_and_preprocess_data(file_path)
    
    if df is None:
        return
    
    print("\n开始气象因素分析...")
    
    # 1. 天气类型与事故严重性表格
    weather_stats = create_weather_severity_table(df)
    
    # 2. 天气类型与事故严重性双Y轴图
    plot_weather_severity_dual_axis(df)
    
    # 3. 连续气象变量箱线图（每个变量单独成图）
    plot_individual_boxplots(df)
    
    # 4. 气象站距离影响分析（每个分析单独成图）
    analyze_weather_distance_impact(df)
    
    # 5. 事故时空密度分析（每个分析单独成图）
    analyze_accident_density(df)
    
    # 6. 生成综合报告
    generate_comprehensive_report(df, weather_stats)
    
    print("\n" + "="*60)
    print("分析完成！所有图表已单独保存为PNG文件")
    print("="*60)

if __name__ == "__main__":
    main()