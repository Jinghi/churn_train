# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline   无法识别 SMOTE 作为有效步骤。
from imblearn.pipeline import Pipeline  #保持兼容性和数据流一致
from imblearn.over_sampling import SMOTE  # 用于处理类别不平衡

# 1. 数据加载与初步探索
data = pd.read_csv('d:\OneDrive\桌面\churn_train.csv')  # 数据文件名

# 查看数据前5行
print(data.head())

# 查看数据基本信息
print(data.info())

# 查看目标变量分布（检查类别不平衡）
print(data['Exited'].value_counts())

# 2. 数据预处理
# 删除无关特征
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 处理缺失值（如果有）
if data.isnull().sum().sum() > 0:
    data = data.dropna()  # 或使用填充策略 data.fillna(...)

# 定义数值和分类特征
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 分割特征和目标变量
X = data.drop('Exited', axis=1)
y = data['Exited']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 探索性数据分析(EDA)
# 绘制目标变量分布
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=data)
plt.title('Distribution of Churned vs Retained Customers')
plt.show()

# 数值特征分布
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# 分类特征与流失率的关系
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_features):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=col, y='Exited', data=data)
    plt.title(f'Churn Rate by {col}')
plt.tight_layout()
plt.show()

# 特征相关性热力图
# 选择数值列
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# 计算相关性并绘制热力图
plt.figure(figsize=(10, 8))
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap (Numerical Features Only)')
plt.show()

# 4. 模型训练与评估
# 4.1定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

#4.2
# 该步骤因为SMOTE需要所有输入特征为数值型，但原始数据包含字符串分类变量（如Geography="France"）。
# 直接调用会报错ValueError: could not convert string to float。所以该步骤对原代码进行修改
# 处理类别不平衡（使用SMOTE）
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 定义特征类型
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']  # 根据实际数据调整

# 创建预处理转换器（数值列标准化，分类列独热编码）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 对训练集预处理后再应用SMOTE
X_train_processed = preprocessor.fit_transform(X_train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

# 测试集预处理（不应用SMOTE）
X_test_processed = preprocessor.transform(X_test)


# 创建模型管道
'''该步骤仍是上述无法识别字符串分类变量问题
 models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
}

# 训练并评估模型
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train_res, y_train_res)
    evaluate_model(model, X_test, y_test)
'''
# 修改后采用下面代码
# 4.3 定义统一的预处理和模型Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

models = {
    'Logistic Regression':
    Pipeline([('preprocessor', preprocessor),
              ('classifier', LogisticRegression(random_state=42))]),
    'Random Forest':
    Pipeline([('preprocessor', preprocessor),
              ('classifier', RandomForestClassifier(random_state=42))]),
    'XGBoost':
    Pipeline([('preprocessor', preprocessor),
              ('classifier',
               XGBClassifier(random_state=42, eval_metric='logloss'))])
}

# 4.4 训练和评估
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)  # 输入原始数据
    evaluate_model(model, X_test, y_test)  # Pipeline会自动预处理测试数据


# 5. 模型优化（以XGBoost为例）
# 定义完整的 Pipeline（包含预处理、SMOTE、模型）
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),  # SMOTE 在 Pipeline 内部
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

# 参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

# 网格搜索
grid_search = GridSearchCV(
    xgb_pipe, 
    param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1
)

# 直接传入原始数据（未处理的 X_train）
grid_search.fit(X_train, y_train)  # 不是 X_train_res

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best ROC AUC: {grid_search.best_score_:.4f}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
evaluate_model(best_model, X_test, y_test)

# 6. 特征重要性分析（以优化后的XGBoost为例）
# 获取特征名称
numeric_features_transformed = numeric_features  # 标准化后的数值特征名称不变
categorical_features_transformed = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = list(numeric_features_transformed) + list(categorical_features_transformed)

# 获取特征重要性
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 绘制特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 7. 业务建议
top_features = feature_importance.head(5)['Feature'].tolist()
print("\n基于分析结果，建议银行重点关注以下客户特征来降低流失率:")
for feature in top_features:
    print(f"- {feature}")
