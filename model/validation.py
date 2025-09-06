# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 하이퍼파라미터 =====
PAST_STEPS   = 310
FUTURE_STEPS = 365
BATCH_SIZE   = 128

# ===== 경로/컬럼 =====
MERGED_PATH = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv"
LASSO_PATH  = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/lasso_importance_cv_2025.csv"
MODEL_PATH  = "cnn_lstm_model.pth"
SCALER_PATH = "scaler.pkl"

TARGET_COL  = "Total CPI"
SENTI_COL   = "sentiment_score"

# ===== 디바이스 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
if device.type == "cuda":
    print(" CUDA", torch.cuda.get_device_name(0))

# ===== 모델 정의 =====
class CNNLSTM(nn.Module):
    def __init__(self, input_features, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS,
                 hidden_dim=512, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, future_steps)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)                 # (B, F, T)
        x = self.pool(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1)                 # (B, T', H)
        x, _ = self.lstm(x)                    # (B, T', H)
        x = self.dropout(x[:, -1, :])          # (B, H)
        return self.fc(x)                      # (B, FUTURE_STEPS)

# ===== 모델 & 스케일러 로드 =====
scaler = joblib.load(SCALER_PATH)
model  = CNNLSTM(input_features=6).to(device)
state  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state)
model.eval()

# ===== 데이터 로드 =====
raw_df   = pd.read_csv(MERGED_PATH, parse_dates=["Date"]).set_index("Date")
lasso_df = pd.read_csv(LASSO_PATH)
top4     = lasso_df["feature"].head(4).tolist()
use_cols = [TARGET_COL, SENTI_COL] + top4
df_m     = raw_df[use_cols].dropna()

# ===== 월→일 확장 + 선형보간 =====
frames = []
for dt, row in df_m.iterrows():
    y, m  = dt.year, dt.month
    days  = calendar.monthrange(y, m)[1]
    idx   = pd.date_range(start=f"{y}-{m:02d}-01", periods=days, freq="D")
    temp  = pd.DataFrame(index=idx, columns=df_m.columns, dtype=float)
    temp.iloc[0] = row.values.astype(float)   # 그 달 1일에만 월값 배치
    frames.append(temp)

df_daily = pd.concat(frames).sort_index()
df_daily.index = pd.to_datetime(df_daily.index)
df_daily = df_daily.rename_axis("Date")       # ★ 인덱스 이름 보장
df_daily = df_daily.interpolate(method="linear")

# ===== 스케일 변환 =====
scaled = scaler.transform(df_daily)  # (N_days, F)

# ===== 슬라이딩 윈도우 (과거구간 예측) =====
X_list, date_list = [], []
limit = len(scaled) - PAST_STEPS - FUTURE_STEPS
for i in range(limit):
    X_list.append(scaled[i:i+PAST_STEPS])  # (T, F)
    date_list.append(df_daily.index[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS])  # 예측 대상 날짜들

# 배치 예측
preds_scaled = []
with torch.no_grad():
    for i in range(0, len(X_list), BATCH_SIZE):
        batch = np.array(X_list[i:i+BATCH_SIZE], dtype=np.float32)
        xb = torch.tensor(batch, dtype=torch.float32, device=device)
        yb = model(xb).cpu().numpy()                 # (B, FUTURE_STEPS)
        preds_scaled.append(yb)
preds_scaled = np.vstack(preds_scaled) if preds_scaled else np.empty((0, FUTURE_STEPS))

# ===== 역정규화(타깃만) & 날짜별 평균 집계 =====
flat_records = []
for dates, yhat_row in zip(date_list, preds_scaled):
    dummy = np.zeros((FUTURE_STEPS, scaled.shape[1]))
    dummy[:, 0] = yhat_row
    inv = scaler.inverse_transform(dummy)[:, 0]     # (FUTURE_STEPS, )
    flat_records.append(pd.DataFrame({"Date": dates, "Predicted CPI": inv}))

pred_df = pd.concat(flat_records, ignore_index=True) if flat_records else pd.DataFrame(columns=["Date","Predicted CPI"])
pred_df = (pred_df
           .groupby("Date", as_index=False)["Predicted CPI"]
           .mean()
           .sort_values("Date"))

# ===== 실제값과 일별 비교 CSV =====
true_df = (df_daily
           .reset_index()[["Date", TARGET_COL]]
           .rename(columns={TARGET_COL: "Actual CPI"}))

# 방어: 예측 df에 Date 없는 경우 대비
if "Date" not in pred_df.columns:
    pred_df = pred_df.reset_index().rename(columns={"index":"Date"})

compare_df = pd.merge(true_df, pred_df, on="Date", how="inner").sort_values("Date")
compare_df["Abs Error"] = (compare_df["Predicted CPI"] - compare_df["Actual CPI"]).abs()
compare_df["PE(%)"]     = compare_df["Abs Error"] / compare_df["Actual CPI"] * 100
compare_df.to_csv("cpi_actual_vs_predicted_daily.csv", index=False, float_format="%.2f")
print("Saved: cpi_actual_vs_predicted_daily.csv")

# ===== 과거 구간 지표 저장 =====
if not compare_df.empty:
    mae  = mean_absolute_error(compare_df["Actual CPI"], compare_df["Predicted CPI"])
    rmse = np.sqrt(mean_squared_error(compare_df["Actual CPI"], compare_df["Predicted CPI"]))
    mape = compare_df["PE(%)"].mean()
else:
    mae = rmse = mape = float("nan")

pd.DataFrame([{"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}]).to_csv(
    "cpi_metrics.csv", index=False, float_format="%.2f"
)
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%  -> Saved: cpi_metrics.csv")

# ======== 미래 1년(365일) 예측 ========
last_window = scaled[-PAST_STEPS:]                    # (T, F)
with torch.no_grad():
    xb = torch.tensor(last_window[None, ...], dtype=torch.float32, device=device)  # (1, T, F)
    yb = model(xb).cpu().numpy().reshape(-1)          # (FUTURE_STEPS,)

dummy = np.zeros((FUTURE_STEPS, scaled.shape[1]))
dummy[:, 0] = yb
future_cpi = scaler.inverse_transform(dummy)[:, 0]     # (365,)

start_next  = df_daily.index[-1] + pd.Timedelta(days=1)
future_days = pd.date_range(start=start_next, periods=FUTURE_STEPS, freq="D")

future_df = pd.DataFrame({"Date": future_days, "Predicted CPI": future_cpi})
future_df.to_csv("cpi_future_daily.csv", index=False, float_format="%.2f")
print("Saved: cpi_future_daily.csv")


