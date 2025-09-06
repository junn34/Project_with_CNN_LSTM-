# -*- coding: utf-8 -*-
import os, calendar, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===== 하이퍼파라미터 =====
PAST_STEPS   = 310
FUTURE_STEPS = 365
BATCH_SIZE   = 128

# ===== 경로/컬럼 =====
MERGED_PATH = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv"
LASSO_PATH  = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/lasso_importance_cv_2025.csv"
MODEL_PATH  = "cnn_lstm_model_core4.pth"
SCALER_PATH = "scaler.pkl"
CORE4_PATH  = "core4_targets.csv"  # train에서 저장한 상위4 목록

TARGET_COL  = "Total CPI"
SENTI_COL   = "sentiment_score"

# ===== 디바이스 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
if device.type == "cuda":
    print(" CUDA", torch.cuda.get_device_name(0))

# ===== 모델 정의 (train과 동일 구조) =====
class CNNLSTM(nn.Module):
    def __init__(self, input_features, n_targets, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS,
                 hidden_dim=512, kernel_size=3, dropout=0.5):
        super().__init__()
        self.future_steps = future_steps
        self.n_targets = n_targets
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_dim, future_steps * n_targets)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x.view(-1, self.future_steps, self.n_targets)

# ===== 로드 =====
scaler = joblib.load(SCALER_PATH)
core4  = pd.read_csv(CORE4_PATH)["feature"].tolist()  # 상위4 타깃명
raw_df = pd.read_csv(MERGED_PATH, parse_dates=["Date"]).set_index("Date")
lasso_df = pd.read_csv(LASSO_PATH)
# (안전) core4가 lasso top4와 다르면 core4 우선
top4 = core4 if core4 else lasso_df["feature"].head(4).tolist()

use_cols = [TARGET_COL, SENTI_COL] + top4
df_m = raw_df[use_cols].dropna()

# ===== 월→일 확장 + 보간 =====
frames = []
for dt, row in df_m.iterrows():
    y, m = dt.year, dt.month
    days = calendar.monthrange(y, m)[1]
    idx  = pd.date_range(start=f"{y}-{m:02d}-01", periods=days, freq="D")
    temp = pd.DataFrame(index=idx, columns=df_m.columns, dtype=float)
    temp.iloc[0] = row.values.astype(float)
    frames.append(temp)

df_daily = pd.concat(frames).sort_index()
df_daily.index = pd.to_datetime(df_daily.index)
df_daily = df_daily.rename_axis("Date")
df_daily = df_daily.interpolate(method="linear")

data = df_daily.values
F = data.shape[1]
target_idxs = [use_cols.index(c) for c in top4]
n_targets = len(top4)

# ===== 모델 준비 & 체크포인트 검증 =====
model = CNNLSTM(input_features=F, n_targets=n_targets).to(device)
state = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# 저장된 타깃 수 추론(안전장치)
fc_out = state["fc.weight"].shape[0]      # 예: 365 또는 1460
saved_n_targets = fc_out // FUTURE_STEPS  # 365//365=1, 1460//365=4
print(f"[Info] saved_n_targets={saved_n_targets}, current_n_targets={n_targets}")
if saved_n_targets != n_targets:
    raise RuntimeError("체크포인트의 타깃 수와 현재 스크립트의 타깃 수가 다릅니다. MODEL_PATH/core4 목록 확인.")

model.load_state_dict(state, strict=True)
model.eval()

# ===== 전체 과거구간 슬라이딩 예측 =====
X_list, date_list = [], []
limit = len(data) - PAST_STEPS - FUTURE_STEPS
for i in range(limit):
    past_block = data[i:i+PAST_STEPS, :]  # (T,F)
    past_block_s = scaler.transform(past_block)
    X_list.append(past_block_s)
    date_list.append(df_daily.index[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS])

pred_batches = []
with torch.no_grad():
    for i in range(0, len(X_list), BATCH_SIZE):
        xb = torch.tensor(np.array(X_list[i:i+BATCH_SIZE], dtype=np.float32), dtype=torch.float32, device=device)
        yb = model(xb).cpu().numpy()  # (B, 365, 4) — 스케일된 값
        pred_batches.append(yb)
pred_scaled = np.vstack(pred_batches) if pred_batches else np.empty((0, FUTURE_STEPS, n_targets))

# ===== 역정규화 (타깃만) + (Date, Feature) 평균 집계 =====
records = []
for dates, yhat in zip(date_list, pred_scaled):  # yhat: (365,4)
    for k, tidx in enumerate(target_idxs):
        dummy = np.zeros((FUTURE_STEPS, F), dtype=np.float32)
        dummy[:, tidx] = yhat[:, k]
        inv = scaler.inverse_transform(dummy)[:, tidx]  # (365,)
        records.append(pd.DataFrame({
            "Date": dates,
            "Feature": top4[k],
            "Predicted": inv
        }))

pred_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["Date","Feature","Predicted"])
pred_df = pred_df.groupby(["Date","Feature"], as_index=False)["Predicted"].mean().sort_values(["Date","Feature"])

# ===== 실제값 (롱) =====
true_long = (df_daily[top4].reset_index().melt(id_vars="Date", var_name="Feature", value_name="Actual"))

# ===== 비교(롱) 저장 =====
cmp_df = pd.merge(true_long, pred_df, on=["Date","Feature"], how="inner").sort_values(["Feature","Date"])
cmp_df["Abs Error"] = (cmp_df["Predicted"] - cmp_df["Actual"]).abs()
cmp_df["PE(%)"]     = cmp_df["Abs Error"] / (cmp_df["Actual"].replace(0, np.nan)).abs() * 100
cmp_df.to_csv("core4_actual_vs_predicted_daily.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
print("Saved: core4_actual_vs_predicted_daily.csv")

# ===== 피처별 지표 =====
rows = []
for feat, g in cmp_df.groupby("Feature"):
    if g.empty:
        rows.append({"Feature": feat, "MAE": np.nan, "RMSE": np.nan, "MAPE(%)": np.nan})
        continue
    mae  = mean_absolute_error(g["Actual"], g["Predicted"])
    rmse = np.sqrt(mean_squared_error(g["Actual"], g["Predicted"]))
    mape = (g["PE(%)"].replace([np.inf, -np.inf], np.nan)).mean()
    rows.append({"Feature": feat, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape})
pd.DataFrame(rows).to_csv("core4_metrics.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
print("Saved: core4_metrics.csv")

# ===== 가로(와이드) 형식 생성 =====
# 1) Actual/Predicted 각각 피벗
act_wide  = cmp_df.pivot(index="Date", columns="Feature", values="Actual")
pred_wide = cmp_df.pivot(index="Date", columns="Feature", values="Predicted")
# 헤더 상단에 'Feature' 같은 이름이 써지지 않도록 제거
act_wide.columns.name = None
pred_wide.columns.name = None

# 2) 접두사 붙여서 병합
act_wide  = act_wide.add_prefix("Actual_")
pred_wide = pred_wide.add_prefix("Predicted_")
compare_wide = act_wide.join(pred_wide, how="inner").reset_index()

# 3) 피처 순서 기준으로 'Actual_x, Predicted_x' 교차 배치
out_cols = ["Date"]
for f in top4:
    out_cols += [f"Actual_{f}", f"Predicted_{f}"]
# 누락된 컬럼이 있을 수 있으니 존재하는 컬럼만 선택
out_cols = [c for c in out_cols if c in compare_wide.columns]
compare_wide = compare_wide.reindex(columns=out_cols)

compare_wide.to_csv("core4_compare_wide.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
print("Saved: core4_compare_wide.csv")

# ======== 미래 365일 예측 ========
last_window = df_daily.values[-PAST_STEPS:, :]
last_window_s = scaler.transform(last_window)
with torch.no_grad():
    xb = torch.tensor(last_window_s[None, ...], dtype=torch.float32, device=device)
    yb = model(xb).cpu().numpy().reshape(FUTURE_STEPS, n_targets)  # (365,4)

# 역정규화하여 피처별로 저장(롱)
future_records = []
start_next  = df_daily.index[-1] + pd.Timedelta(days=1)
future_days = pd.date_range(start=start_next, periods=FUTURE_STEPS, freq="D")
for k, feat in enumerate(top4):
    dummy = np.zeros((FUTURE_STEPS, F), dtype=np.float32)
    t_idx = use_cols.index(feat)
    dummy[:, t_idx] = yb[:, k]
    inv = scaler.inverse_transform(dummy)[:, t_idx]
    future_records.append(pd.DataFrame({"Date": future_days, "Feature": feat, "Predicted": inv}))
future_df = pd.concat(future_records, ignore_index=True)

# 미래 365일 와이드
future_wide = future_df.pivot(index="Date", columns="Feature", values="Predicted")
future_wide.columns.name = None
future_wide = future_wide.reset_index()
future_wide.to_csv("core4_future_daily_wide.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
print("Saved: core4_future_daily_wide.csv")
