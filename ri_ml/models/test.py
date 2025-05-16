import pretrait_tools as pt
import feature_engineering as fe  

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pandas as pd



INPUT_FILE = "../data/all.tsv"

# ---------- Run 全流程 ----------
dialogs = pt.load_dialogs(INPUT_FILE)
df = pt.flatten_dialogs(dialogs)
df, mlb = pt.binarize_labels(df)

df = fe.extract_tfidf_similarity_features(df)

# 展示结果
print(df.head()) 

X = pt.build_feature_matrix(df)
Y = df[['OQ', 'PA', 'FD', 'PF', 'NF', 'CQ', 'FQ', 'IR', 'GG', 'JK', 'RQ', 'O']]

print("-------------------------------------")
print("Feature matrix shape:", X.shape)
print("Label matrix shape:", Y.shape)


# Step 1：划分训练/测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2：构建分类器链 + 随机森林（你也可以试 SVM 或 AdaBoost）
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = ClassifierChain(base_model)
model.fit(X_train, Y_train)

# Step 3：进行预测
Y_pred = model.predict(X_test)

# Step 4：多标签评估指标（宏平均）
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def multilabel_metrics(y_true, y_pred):
    # 确保是 NumPy int 数组（不能是 float！）
    y_true_bin = y_true.to_numpy(dtype=int)
    y_pred_bin = (pd.DataFrame(y_pred, columns=y_true.columns) >= 0.5).to_numpy(dtype=int)

    # 使用 NumPy 做位运算，不走 Pandas 路径
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum(axis=1)
    union = np.logical_or(y_true_bin, y_pred_bin).sum(axis=1)
    acc = np.mean(np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union!=0))

    # 回到 DataFrame 格式再用 sklearn 的 macro 指标
    y_true_df = pd.DataFrame(y_true_bin, columns=y_true.columns)
    y_pred_df = pd.DataFrame(y_pred_bin, columns=y_true.columns)

    precision = precision_score(y_true_df, y_pred_df, average='macro', zero_division=0)
    recall = recall_score(y_true_df, y_pred_df, average='macro', zero_division=0)
    f1 = f1_score(y_true_df, y_pred_df, average='macro', zero_division=0)

    print("Multilabel Accuracy (IoU):", round(acc, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1 Score:", round(f1, 4))



multilabel_metrics(Y_test, Y_pred)



