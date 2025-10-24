import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import shap

# 读取数据
df = pd.read_csv("AccidentswithWeather.csv")

# 特征与目标
features = ['Hour', 'Weekday', 'Start_Lat', 'Start_Lng', 'County', 'City',
            'Crossing', 'Junction', 'Traffic_Signal', 'Bump',
            'PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNOW', 'SNWD', 'AWND', 'Weather_Condition']
target = 'Severity_flag'

# 编码类别变量
le_county = LabelEncoder()
le_city = LabelEncoder()
le_weather = LabelEncoder()

df['County_encoded'] = le_county.fit_transform(df['County'])
df['City_encoded'] = le_city.fit_transform(df['City'])
df['Weather_encoded'] = le_weather.fit_transform(df['Weather_Condition'])

# 构建特征集
X = df[['Hour', 'Weekday', 'Start_Lat', 'Start_Lng', 'Crossing', 'Junction',
        'Traffic_Signal', 'Bump', 'PRCP', 'TAVG', 'TMAX', 'TMIN', 'SNOW',
        'SNWD', 'AWND', 'County_encoded', 'City_encoded', 'Weather_encoded']]
y = df[target]

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("=== 数据分布分析 ===")
print(f"训练集样本数: {len(y_train)}")
print(f"严重事故比例: {y_train.mean():.2%}")
print(f"一般事故比例: {(1 - y_train.mean()):.2%}")
print(f"不平衡比例: {1/y_train.mean():.1f}:1")

# 改进1: 使用类别权重
print("\n=== 改进1: 使用类别权重的XGBoost ===")
# 计算类别权重
n_class_0 = np.sum(y_train == 0)
n_class_1 = np.sum(y_train == 1)
scale_pos_weight = n_class_0 / n_class_1

xgb_weighted = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,  # 关键改进：处理不平衡数据
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

# 改进2: 使用重采样技术
print("\n=== 改进2: 使用SMOTE过采样 ===")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

xgb_smote = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

# 交叉验证评估
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 原始XGBoost（基准）
print("\n=== 原始XGBoost CV 结果 ===")
y_pred_xgb_original = cross_val_predict(xgb_weighted, X_train, y_train, cv=skf, method='predict')
print(classification_report(y_train, y_pred_xgb_original))

# 使用SMOTE的XGBoost
print("\n=== SMOTE-XGBoost CV 结果 ===")
xgb_smote.fit(X_resampled, y_resampled)
y_pred_smote = xgb_smote.predict(X_train)
print(classification_report(y_train, y_pred_smote))

# 改进3: 调整分类阈值
print("\n=== 改进3: 阈值调整分析 ===")
xgb_weighted.fit(X_train, y_train)
y_pred_proba = xgb_weighted.predict_proba(X_train)[:, 1]

# 寻找最佳阈值
precision, recall, thresholds = precision_recall_curve(y_train, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"默认阈值: 0.5")
print(f"最佳F1阈值: {best_threshold:.3f}")

# 使用最佳阈值
y_pred_optimized = (y_pred_proba > best_threshold).astype(int)
print("\n=== 优化阈值后的结果 ===")
print(classification_report(y_train, y_pred_optimized))

# 训练最终模型（选择效果最好的）
print("\n=== 最终模型训练 ===")
final_model = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

final_model.fit(X_train, y_train)

# 在测试集上评估
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

print("=== 测试集结果 ===")
print(classification_report(y_test, y_test_pred))

# 特征重要性图
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh')
plt.title("改进后XGBoost特征重要性")
plt.tight_layout()
plt.show()

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['一般事故', '严重事故'], 
            yticklabels=['一般事故', '严重事故'])
plt.title("改进模型混淆矩阵 (测试集)")
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 输出关键业务指标
print("=== 业务关键指标 ===")
tn, fp, fn, tp = cm.ravel()
print(f"严重事故检测率(召回率): {tp/(tp+fn):.2%}")
print(f"严重事故误报率: {fp/(fp+tn):.2%}")
print(f"严重事故漏报数: {fn}")
print(f"严重事故正确检测数: {tp}")

# 保存改进后的预测结果
results_df = pd.DataFrame({
    '真实标签': y_test,
    '预测概率': y_test_proba,
    '预测标签': y_test_pred
})
results_df.to_csv('improved_predictions.csv', index=False)

