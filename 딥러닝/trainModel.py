import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import calendar


PAST_STEPS = 310
FUTURE_STEPS = 365
BATCH_SIZE = 64
EPOCHS = 50

raw_df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv", parse_dates=["Date"]).set_index("Date")


lasso_df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/lasso_importance_cv_2025.csv")
selected_features = lasso_df["feature"].head(4).tolist()
selected_cols = ["Total CPI"] + selected_features
df = raw_df[selected_cols].dropna()


# 월별 데이터를 각 월의 일수로 확장 (첫날만 값 있고 나머지는 NaN)


daily_frames = []
for date, row in df.iterrows():
    year = date.year
    month = date.month
    days = calendar.monthrange(year, month)[1]
    dates = pd.date_range(start=f"{year}-{month:02d}-01", periods=days)
    temp = pd.DataFrame(index=dates, columns=df.columns, dtype=float)

    temp.iloc[0] = row.values.astype(float) 
    daily_frames.append(temp)


# 연결 후 선형 보간
df_daily = pd.concat(daily_frames)
df_daily = df_daily.interpolate(method="linear")


scaler = MinMaxScaler()
df_daily_scaled = pd.DataFrame(scaler.fit_transform(df_daily),
                               columns=df_daily.columns,
                               index=df_daily.index)


# 슬라이딩 윈도우 구성
data = df_daily_scaled.values
X_list, Y_list = [], []
for i in range(len(data) - PAST_STEPS - FUTURE_STEPS):
    X_list.append(data[i:i+PAST_STEPS])
    Y_list.append(data[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS, 0])
X_np = np.array(X_list)
Y_np = np.array(Y_list)


class CPITimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

dataset = CPITimeSeriesDataset(X_np, Y_np)
train_len = int(len(dataset)*0.7)
val_len = int(len(dataset)*0.15)
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=1)


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

# 학습 및 테스트
model = CNNLSTM(input_features=X_np.shape[2]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = next(model.parameters()).device
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

def train(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_losses, val_losses = [], []
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), Yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                val_losses.append(criterion(model(Xb), Yb).item())
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

def test(model, test_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            preds.append(model(Xb).cpu().numpy())
            trues.append(Yb.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)

train(model, train_loader, val_loader, epochs=EPOCHS)
preds, trues = test(model, test_loader)

# 정규화 상태 성능 평가
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.std(y_true)

y_true = trues.flatten()
y_pred = preds.flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape_val = mape(y_true, y_pred)
smape_val = smape(y_true, y_pred)
nrmse_val = nrmse(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mse_val = mean_squared_error(y_true, y_pred)

print("\n📊 정규화 상태 성능 평가 결과:")
print(f" RMSE   : {rmse:.4f}")
print(f" MAE    : {mae:.4f}")
print(f" MAPE   : {mape_val:.2f}%")
print(f" SMAPE  : {smape_val:.2f}%")
print(f" NRMSE  : {nrmse_val:.4f}")
print(f" R²     : {r2:.4f}")
print(f" MSE    : {mse_val:.6f}")

# 모델 및 스케일러 저장
# torch.save(model.state_dict(), "cnn_lstm_model.pth")
# np.save("latest_input.npy", df_daily_scaled.values[-PAST_STEPS:])
# import joblib
# joblib.dump(scaler, "scaler.pkl")
