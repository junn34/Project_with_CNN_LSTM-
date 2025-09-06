# -*- coding: cp949 -*-
"""
[INFERENCE ONLY | no-sentiment]
상위 4개(target_features) 일별 예측 생성 (감성점수 제외)
- 월→일 확장 + 선형 보간
- 저장된 MinMaxScaler 로만 transform (fit 금지)
- CNN+LSTM 아키텍처: 학습과 동일 (conv padding 없음, dropout=0.5)
- 전체 구간 슬라이딩 일별 예측
- 역정규화 후 (date, feature) 중복 평균
- 실제와 병합하여 비교 CSV 저장
- 지표: MAE / RMSE / MAPE(%) / NRMSE
"""

import os, calendar, numpy as np, pandas as pd, torch, joblib
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===================== 설정 =====================
PAST_STEPS   = 310
FUTURE_STEPS = 365
BATCH_SIZE   = 128

# 반드시 nosenti 아티팩트 사용(학습도 감성 제외로 했다고 가정)
MODEL_PATH  = "cnn_lstm_core4_nosenti.pth"
SCALER_PATH = "scaler_core4_nosenti.pkl"
CORE4_PATH  = "core4_targets.csv"     # 라쏘 상위 4개(감성 없음)
INPUT_CSV   = "merged_data_2025.csv"

# 출력 파일
PRED_ONLY_CSV = "core4_daily_pred_only.csv"
COMPARE_CSV   = "validation_core4_daily_2005_2025.csv"
METRICS_CSV   = "validation_core4_daily_metrics.csv"

# ===================== 유틸 =====================
def safe_mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(y_true), eps, None)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def nrmse(y_true, y_pred):
    std = float(np.std(y_true, ddof=0))
    return (rmse(y_true, y_pred) / std) if std >= 1e-12 else np.nan

# ===================== 데이터 로드 =====================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"입력 파일 없음: {INPUT_CSV}")
raw_df = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).set_index("Date").sort_index()

if not os.path.exists(CORE4_PATH):
    raise FileNotFoundError("core4_targets.csv 가 필요합니다.")
target_features = pd.read_csv(CORE4_PATH, header=None)[0].astype(str).tolist()
# 방어적으로 감성 점수 제거
target_features = [f for f in target_features if f != "sentiment_score"]

# 입력 컬럼(감성 제외)
selected_cols = ["Total CPI"] + target_features

missing = [c for c in selected_cols if c not in raw_df.columns]
if missing:
    raise ValueError(f"데이터에 없는 컬럼: {missing}")

df = raw_df[selected_cols].dropna().copy()

# ===== 월→일 확장 + 선형 보간 =====
frames = []
for d, row in df.iterrows():
    y, m = d.year, d.month
    days = calendar.monthrange(y, m)[1]
    idx = pd.date_range(f"{y:04d}-{m:02d}-01", periods=days, freq="D")
    t = pd.DataFrame(index=idx, columns=df.columns, dtype=float)
    t.iloc[0] = row.values.astype(float)
    frames.append(t)

df_daily = pd.concat(frames).sort_index().interpolate("linear").ffill().bfill()

# ===================== 스케일러 로드 =====================
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"스케일러 파일 없음: {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

# 스케일러-입력 컬럼 검증
sf = getattr(scaler, "n_features_in_", None)
if sf is not None and sf != len(selected_cols):
    raise ValueError(
        f"[불일치] scaler expects {sf} features, but selected_cols={len(selected_cols)}.\n"
        f"SCALER_PATH={SCALER_PATH}, selected_cols={selected_cols}"
    )

# 스케일러 적용
scaled_vals = scaler.transform(df_daily[selected_cols].values)
scaled = pd.DataFrame(scaled_vals, columns=selected_cols, index=df_daily.index)

n_features = scaled.shape[1]
tidx = [scaled.columns.get_loc(c) for c in target_features]
n_targets = len(target_features)

# ===================== 슬라이딩 윈도우 =====================
S = scaled.values.astype(np.float32)
X, future_dates = [], []
limit = len(scaled) - PAST_STEPS - FUTURE_STEPS
for i in range(limit):
    X.append(S[i:i+PAST_STEPS])
    future_dates.append(df_daily.index[i+PAST_STEPS : i+PAST_STEPS+FUTURE_STEPS])
X = np.asarray(X, dtype=np.float32)

# ===================== 모델 정의/로드 =====================
class CNNLSTM(nn.Module):
    def __init__(self, input_features, n_targets, future_steps=FUTURE_STEPS):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, 512, 3)  # padding 없음
        self.conv2 = nn.Conv1d(512, 512, 3)
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(512, 512, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # 학습과 동일
        self.fc = nn.Linear(512, future_steps * n_targets)
        self.future_steps, self.n_targets = future_steps, n_targets

    def forward(self, x):
        x = x.permute(0, 2, 1)    # (B, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)    # (B, T', C)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x.view(-1, self.future_steps, self.n_targets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(n_features, n_targets).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}")
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# 모델-입력 컬럼 검증
if model.conv1.in_channels != len(selected_cols):
    raise ValueError(
        f"[불일치] model in_channels={model.conv1.in_channels}, "
        f"but selected_cols={len(selected_cols)}.\nMODEL_PATH={MODEL_PATH}"
    )

# ===================== 배치 추론 =====================
preds_scaled_list = []
with torch.no_grad():
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[i:i+BATCH_SIZE], device=device)
        out = model(xb).cpu().numpy()
        preds_scaled_list.append(out)
preds_scaled = np.vstack(preds_scaled_list)   # (N, FUTURE_STEPS, n_targets)

# ===================== 역정규화 & (date,feature) 평균 =====================
pred_records = []
for i in range(preds_scaled.shape[0]):
    dates_i = future_dates[i]
    dummy = np.zeros((FUTURE_STEPS, n_features), dtype=np.float32)
    dummy[:, tidx] = preds_scaled[i]
    inv = scaler.inverse_transform(dummy)
    inv_targets = inv[:, tidx]
    for t_idx, dt in enumerate(dates_i):
        for j, feat in enumerate(target_features):
            pred_records.append((dt, feat, float(inv_targets[t_idx, j])))

pred_df_long = pd.DataFrame(pred_records, columns=["date", "feature", "pred"])
pred_daily_avg = pred_df_long.groupby(["date", "feature"])["pred"].mean().reset_index()

pred_daily_wide = pred_daily_avg.pivot(index="date", columns="feature", values="pred").sort_index()
pred_daily_wide.columns = [f"{c}_pred" for c in pred_daily_wide.columns]

true_daily_wide = df_daily[target_features].copy()
true_daily_wide.columns = [f"{c}_true" for c in true_daily_wide.columns]

common_idx = pred_daily_wide.index.intersection(true_daily_wide.index)
comp_daily = pd.concat([true_daily_wide.loc[common_idx], pred_daily_wide.loc[common_idx]], axis=1).reset_index()

# ===================== 저장 =====================
pred_daily_wide.to_csv(PRED_ONLY_CSV, encoding="utf-8-sig")
comp_daily.to_csv(COMPARE_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] 예측 저장: {PRED_ONLY_CSV}, {COMPARE_CSV}")

# ===================== 지표 계산 =====================
metrics = []
for feat in target_features:
    y_true = comp_daily[f"{feat}_true"].values
    y_pred = comp_daily[f"{feat}_pred"].values
    mae = mean_absolute_error(y_true, y_pred)
    r = rmse(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    nr = nrmse(y_true, y_pred)
    metrics.append((feat, mae, r, mape, nr))
met_df = pd.DataFrame(metrics, columns=["feature", "MAE", "RMSE", "MAPE", "NRMSE"])
met_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] 메트릭 저장: {METRICS_CSV}")
print(met_df)

with torch.no_grad():
    last_in = torch.tensor(S[-PAST_STEPS:], dtype=torch.float32).unsqueeze(0).to(device)
    fut_scaled = model(last_in).cpu().numpy()[0]

dummy = np.zeros((FUTURE_STEPS, n_features), dtype=np.float32)
dummy[:, tidx] = fut_scaled
fut_inv = scaler.inverse_transform(dummy)[:, tidx]

start_date = df_daily.index[-1] + pd.Timedelta(days=1)
fut_dates = pd.date_range(start=start_date, periods=FUTURE_STEPS, freq="D")

fut_df = pd.DataFrame(fut_inv, index=fut_dates, columns=target_features)
fut_df.columns = [f"{c}_forecast" for c in fut_df.columns]
fut_df = fut_df.reset_index().rename(columns={"index":"date"})
fut_df.to_csv("next365_core4_daily.csv", index=False, encoding="utf-8-sig")
print("[OK] 저장: next365_core4_withoutsenti_daily.csv")
