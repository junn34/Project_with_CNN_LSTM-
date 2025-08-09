import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 하이퍼파라미터
PAST_STEPS = 310
FUTURE_STEPS = 365

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 현재 디바이스: {device}")
if device.type == "cuda":
    print(" CUDA", torch.cuda.get_device_name(0))
else:
    print("CPU 사용")

# 모델 클래스 정의
class CNNLSTM(nn.Module):
    def __init__(self, input_features, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS, hidden_dim=512, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, future_steps)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# 모델과 스케일러 불러오기
scaler = joblib.load("scaler.pkl")
model = CNNLSTM(input_features=5).to(device)  # ← CUDA로 이동
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=device))
model.eval()

# 정규화된 전체 데이터 불러오기
full_scaled = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv", parse_dates=["Date"]).set_index("Date")
lasso_df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/lasso_importance_cv_2025.csv")
selected_features = lasso_df["feature"].head(4).tolist()
selected_cols = ["Total CPI"] + selected_features
full_df = full_scaled[selected_cols].dropna()

#  MinMaxScaler 다시 적용
from sklearn.preprocessing import MinMaxScaler
daily_frames = []
import calendar
for date, row in full_df.iterrows():
    year, month = date.year, date.month
    days = calendar.monthrange(year, month)[1]
    dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=days)
    temp = pd.DataFrame(index=dates, columns=full_df.columns, dtype=float)
    temp.iloc[0] = row.values.astype(float)
    daily_frames.append(temp)
df_daily = pd.concat(daily_frames)
df_daily = df_daily.interpolate(method="linear")

scaled_data = scaler.transform(df_daily)
X_list, Y_list, date_list = [], [], []
for i in range(len(scaled_data) - PAST_STEPS - FUTURE_STEPS):
    X_list.append(scaled_data[i:i+PAST_STEPS])
    Y_list.append(scaled_data[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS, 0])
    date_list.append(df_daily.index[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS])

# 내 gpu 크기 6gb라 배치크기 나눠서 진행했음
BATCH_SIZE = 128
preds_list = []

model.eval()
with torch.no_grad():
    for i in range(0, len(X_list), BATCH_SIZE):
        batch = X_list[i:i+BATCH_SIZE]
        x_batch = torch.tensor(batch, dtype=torch.float32).to(device)
        batch_preds = model(x_batch).cpu().numpy()
        preds_list.append(batch_preds)

preds = np.vstack(preds_list)


# 예측 CPI 역정규화
pred_cpis = []
for i in range(len(preds)):
    dummy = np.zeros((FUTURE_STEPS, scaled_data.shape[1]))
    dummy[:, 0] = preds[i]
    inverse = scaler.inverse_transform(dummy)[:, 0]
    pred_cpis.append(inverse)

# 예측 cpi 압축
flat_preds = []
flat_dates = []

for dates, values in zip(date_list, pred_cpis):
    pred_df = pd.DataFrame({"date": dates, "value": values})
    pred_df["date"] = pred_df["date"].dt.to_period("M").dt.to_timestamp()
    flat_preds.extend(pred_df["value"].values)
    flat_dates.extend(pred_df["date"].values)

# 날짜 기준으로 평균
agg_df = pd.DataFrame({"date": flat_dates, "value": flat_preds})
avg_pred = agg_df.groupby("date").mean()

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(df_daily.index, df_daily["Total CPI"], label="Actual CPI", color="blue", linewidth=1)
plt.plot(avg_pred.index, avg_pred["value"], color='red', label="Model Prediction (평균)", linewidth=2)
plt.title("예측 CPI vs 실제 CPI (2005~2025.4)")
plt.xlabel("Date")
plt.ylabel("CPI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cpi_train_single_avg_prediction.png")
plt.show()

# 성능 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 실제 CPI 월별 평균 계산
true_monthly = df_daily["Total CPI"].groupby(df_daily.index.to_period("M")).mean()
true_monthly.index = true_monthly.index.to_timestamp()

# 예측값과 날짜 일치시키기 (교집합만)
common_idx = avg_pred.index.intersection(true_monthly.index)
true_vals = true_monthly.loc[common_idx]
pred_vals = avg_pred.loc[common_idx, "value"]

# 평가 
mae = mean_absolute_error(true_vals, pred_vals)
rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))  
mape = np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100

print(f"예측 성능 평가:")
print(f"MAE  (평균 절대 오차): {mae:.4f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.4f}")
print(f"MAPE (평균 절대 백분율 오차): {mape:.2f}%")
