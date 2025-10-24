import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ------------------ 1. 数据读取 ------------------
file_path = r"F:\大三上\大数据\期初作业\Accidents_with_Weather_final_kdtree.csv"
df = pd.read_csv(file_path)

df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Month'] = df['Start_Time'].dt.to_period('M').astype(str)

# 用少量数据保证速度
df_plot = df.sample(5000).copy()

# ------------------ 2. 初始化 Dash ------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    html.H2("🚦 美国交通事故 - 交互式可视化仪表盘", className="text-center my-3"),

    # ----------- 图表选择 & 气象变量选择 -----------
    dbc.Row([
        dbc.Col([
            html.Label("选择气象变量"),
            dcc.Dropdown(
                id="feature_choice",
                options=[
                    {"label": "温度 Temperature(F)", "value": "Temperature(F)"},
                    {"label": "降水量 Precipitation(in)", "value": "Precipitation(in)"},
                    {"label": "能见度 Visibility(mi)", "value": "Visibility(mi)"},
                    {"label": "气压 Pressure(in)", "value": "Pressure(in)"}
                ],
                value="Temperature(F)"
            )
        ], width=4),
    ], className="my-2"),

    # ----------- 图表内容 -----------
    dbc.Tabs([
        dbc.Tab(label="📍 地理可视化（Plotly）", tab_id="tab-map", children=[
            dcc.Graph(id="map-plot")
        ]),
        dbc.Tab(label="📊 时间滑块动画", tab_id="tab-time", children=[
            dcc.Graph(id="time-slider-plot")
        ]),
        dbc.Tab(label="🌦 气象特征与事故分析", tab_id="tab-weather", children=[
            dcc.Graph(id="weather-analysis-plot")
        ])
    ])
], fluid=True)

# ------------------ 3. 回调布局 ------------------

# 地理分布图
@app.callback(
    Output("map-plot", "figure"),
    Input("feature_choice", "value")
)
def update_map(feature):
    fig = px.scatter_mapbox(
        df_plot,
        lat="Start_Lat", lon="Start_Lng",
        color="Severity",
        hover_data=["Start_Time", "Weather_Condition", feature],
        zoom=4, height=600
    )
    fig.update_layout(mapbox_style="open-street-map", title="📍 交通事故地理分布")
    return fig

# 时间滑块动画
@app.callback(
    Output("time-slider-plot", "figure"),
    Input("feature_choice", "value")
)
def update_time_plot(_):
    fig = px.scatter_mapbox(
        df_plot, lat="Start_Lat", lon="Start_Lng",
        color="Severity",
        animation_frame="Month",
        hover_data=["Start_Time", "Weather_Condition"],
        zoom=4, height=600
    )
    fig.update_layout(title="📊 按时间演化的事故分布", mapbox_style="open-street-map")
    return fig

# 气象特征 vs 事故分析
@app.callback(
    Output("weather-analysis-plot", "figure"),
    Input("feature_choice", "value")
)
def update_weather_plot(feature):
    fig = px.scatter(
        df_plot, x=feature, y="Severity",
        color="Severity",
        animation_frame="Month",
        labels={feature: feature, "Severity": "事故严重程度"},
        title=f"🌦 {feature} 与事故严重程度关系（动态）"
    )
    return fig

# ------------------ 4. 主函数启动 ------------------
if __name__ == "__main__":
    app.run(debug=True)
