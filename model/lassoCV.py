import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# 분석 기간 필터링
df = df.loc["2005-01-01":"2025-04-30"]

# 입력/타겟 분리
target_col = "Total CPI"
X = df.drop(columns=[target_col])
y = df[target_col]

# 결측치 처리
X = X.fillna(method="ffill").fillna(method="bfill")

# 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# LassoCV로 최적 alpha 찾기 및 학습
lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso.fit(X_scaled, y)

# 계수 정리
coefs = pd.Series(lasso.coef_, index=X.columns)
non_zero_coefs = coefs[coefs != 0]
importance = np.abs(non_zero_coefs)
importance_pct = (importance / importance.sum()) * 100

# 결과 저장
result_df = pd.DataFrame({
    "feature": importance_pct.index,
    "coef": non_zero_coefs.values,
    "importance": importance_pct.values
}).sort_values(by="importance", ascending=False).reset_index(drop=True)
result_df["rank"] = result_df.index + 1
result_df = result_df[["rank", "feature", "coef", "importance"]]


output_dir = "C:/Users/bjh20/source/repos/딥러닝/딥러닝"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lasso_importance_cv_2025.csv")
result_df.to_csv(output_path, index=False)

print("저장 완료:", output_path)
print("최적 alpha 값:", lasso.alpha_)
