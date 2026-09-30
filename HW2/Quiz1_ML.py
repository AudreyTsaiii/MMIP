import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("bank.csv", sep=",")
# print(df.shape)
# print(df.head())


# 將 deposit 那欄轉成 0 or 1，這樣才能做 binary classification
df["deposit"] = df["deposit"].map({"no": 0, "yes": 1})

# 定義 categorical / numerical features
categorical_cols = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

numeric_cols = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]


# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop(columns="deposit")
y = df_encoded["deposit"]

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()

# 保留原始資料
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

# Evaluation function
def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "threshold": threshold,
        "confusion_matrix": cm,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

# Model 1: Logistic Regression
print("Model 1: Logistic Regression")
model1 = LogisticRegression(max_iter=1000, random_state=42)
model1.fit(X_train_scaled, y_train)

# 取得 Validation probability
proba1 = model1.predict_proba(X_val_scaled)[:, 1]
pred1 = (proba1 >= 0.5).astype(int)

print("Validation size:", len(y_val))
print("\nActual class distribution:")
print(y_val.value_counts())
print("\nPredicted class distribution:")
print(pd.Series(pred1).value_counts())
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, pred1))
print("\nEvaluation:")
print(evaluate(y_val, proba1, threshold=0.5))

# Model 1: Threshold tuning
print("\nModel 1 - Threshold = 0.3")
print(evaluate(y_val, proba1, threshold=0.3))

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]
results1 = []

for threshold in thresholds:
    result = evaluate(y_val,proba1,threshold)
    results1.append(result)

results1_df = pd.DataFrame(results1)

print("\nModel 1 Threshold Results:")
print(results1_df[["threshold", "accuracy", "precision", "recall", "f1"]])


# Model 2: Random Forest
print("Model 2: Random Forest")

model2 = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
model2.fit(X_train_scaled, y_train)

# 取得 Validation probability
proba2 = model2.predict_proba(X_val_scaled)[:, 1]

# Threshold = 0.5
pred2 = (proba2 >= 0.5).astype(int)

print("\nValidation size:", len(y_val))

print("\nActual class distribution:")
print(y_val.value_counts())
print("\nPredicted class distribution:")
print(pd.Series(pred2).value_counts())
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, pred2))
print("\nEvaluation:")
print(evaluate(y_val, proba2, threshold=0.5))


# Model 2: Threshold tuning
results2 = []
for threshold in thresholds:
    result = evaluate(y_val,proba2,threshold)
    results2.append(result)


results2_df = pd.DataFrame(results2)

print("\nModel 2 Threshold Results:")
print(results2_df[["threshold", "accuracy", "precision", "recall", "f1"]])


# 找出兩個模型各自的最佳 Threshold
best1 = results1_df.loc[
    results1_df["f1"].idxmax()
]

best2 = results2_df.loc[
    results2_df["f1"].idxmax()
]


print("Best Threshold Comparison")
print("\nLogistic Regression:")
print(best1)

print("\nRandom Forest:")
print(best2)



# 使用各自選定 Threshold 比較兩個模型
comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Threshold": [
        best1["threshold"],
        best2["threshold"]
    ],

    "Accuracy": [
        best1["accuracy"],
        best2["accuracy"]
    ],

    "Precision": [
        best1["precision"],
        best2["precision"]
    ],

    "Recall": [
        best1["recall"],
        best2["recall"]
    ],

    "F1": [
        best1["f1"],
        best2["f1"]
    ]
})

print("Model Comparison")
print(comparison)