import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파라미터
PAST_STEPS = 310
FUTURE_STEPS = 365

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

# 데이터 및 모델 불러오기
latest_input = np.load("latest_input.npy")
scaler = joblib.load("scaler.pkl")
raw_df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv", parse_dates=["Date"]).set_index("Date")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(input_features=latest_input.shape[1])
model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location=device))
model.to(device)
model.eval()

# 예측
input_tensor = torch.tensor(latest_input, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pred_scaled = model(input_tensor).cpu().numpy().flatten()

# 역정규화
dummy = np.zeros((FUTURE_STEPS, latest_input.shape[1]))
dummy[:, 0] = pred_scaled
pred_cpi = scaler.inverse_transform(dummy)[:, 0]

# 날짜 생성
last_date = pd.to_datetime("2025-04-30")
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_STEPS)

# 실제 데이터 추출 (2005년~2025년 4월까지)
actual_df = raw_df.loc["2005-01-01":"2025-04-30"]
actual_dates = actual_df.index
actual_cpi = actual_df["Total CPI"]

# 예측 결과 저장
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted CPI": pred_cpi})
forecast_df.to_csv("cpi_forecast_1year.csv", index=False)

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(actual_dates, actual_cpi, label="Actual CPI (2005–2025.4)", color="blue")
plt.plot(future_dates, pred_cpi, label="Forecasted CPI (2025.5–2026.4)", color="red")
plt.axvline(x=last_date, linestyle='--', color='gray', label="Forecast Start")
plt.title("CPI 시계열: 실제 vs 1년 예측")
plt.xlabel("Date")
plt.ylabel("CPI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cpi_forecast_full_view.png")
plt.show()
