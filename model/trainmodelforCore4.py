# -*- coding: utf-8 -*-
import os, json, calendar, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== 하이퍼파라미터 =====
PAST_STEPS   = 310
FUTURE_STEPS = 365
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 5e-4
HIDDEN_DIM   = 512
KERNEL_SIZE  = 3
DROPOUT      = 0.5
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15   # TEST = 0.15

# ===== 경로 =====
MERGED_PATH = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/merged_data_2025.csv"
LASSO_PATH  = r"C:/Users/bjh20/source/repos/딥러닝/딥러닝/lasso_importance_cv_2025.csv"
MODEL_PATH  = "cnn_lstm_model_core4.pth"
SCALER_PATH = "scaler.pkl"

TARGET_COL  = "Total CPI"
SENTI_COL   = "sentiment_score"

# ===== 디바이스 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
if device.type == "cuda":
    print(" CUDA", torch.cuda.get_device_name(0))

# ===== 데이터 로드 =====
raw_df   = pd.read_csv(MERGED_PATH, parse_dates=["Date"]).set_index("Date")
lasso_df = pd.read_csv(LASSO_PATH)
top4     = lasso_df["feature"].head(4).tolist()

# 입력 피처 구성 (입력은 6개, 타깃은 top4(4개))
use_cols = [TARGET_COL, SENTI_COL] + top4
df_m     = raw_df[use_cols].dropna()

# ===== 월→일 확장 + 선형보간 =====
frames = []
for dt, row in df_m.iterrows():
    y, m  = dt.year, dt.month
    days  = calendar.monthrange(y, m)[1]
    idx   = pd.date_range(start=f"{y}-{m:02d}-01", periods=days, freq="D")
    temp  = pd.DataFrame(index=idx, columns=df_m.columns, dtype=float)
    temp.iloc[0] = row.values.astype(float)  # 그 달 1일에만 월값 배치
    frames.append(temp)

df_daily = pd.concat(frames).sort_index()
df_daily.index = pd.to_datetime(df_daily.index)
df_daily = df_daily.rename_axis("Date")
df_daily = df_daily.interpolate(method="linear")

# ===== 슬라이딩 윈도우 전체 구성 (스케일 전) =====
data = df_daily.values  # (N_days, F=6)
F = data.shape[1]
target_idxs = [use_cols.index(c) for c in top4]  # 예측대상(4개)의 컬럼 인덱스

X_list, Y_list = [], []
limit = len(data) - PAST_STEPS - FUTURE_STEPS
for i in range(limit):
    past_block  = data[i:i+PAST_STEPS, :]                     # (T, F)
    future_block = data[i+PAST_STEPS:i+PAST_STEPS+FUTURE_STEPS, :]  # (365, F)
    X_list.append(past_block)
    # Y: (365, 4) — top4 타깃만 추출
    Y_list.append(future_block[:, target_idxs])
X_np = np.array(X_list, dtype=np.float32)          # (N, T, F)
Y_np = np.array(Y_list, dtype=np.float32)          # (N, 365, 4)
print("[Windows] X:", X_np.shape, " Y:", Y_np.shape)

# ===== 시간순 분할 (정보누수 방지 & 스케일러는 Train만 fit) =====
N = len(X_np)
n_train = int(N * TRAIN_RATIO)
n_val   = int(N * VAL_RATIO)
n_test  = N - n_train - n_val

X_train, X_val, X_test = X_np[:n_train], X_np[n_train:n_train+n_val], X_np[n_train+n_val:]
Y_train, Y_val, Y_test = Y_np[:n_train], Y_np[n_train:n_train+n_val], Y_np[n_train+n_val:]
print(f"[Split] train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")

# ===== 스케일러 (Train 구간만 fit) =====
scaler = MinMaxScaler()
train_flat = X_train.reshape(-1, F)  # 과거구간만으로 fit
scaler.fit(train_flat)
# 변환
def scale_blocks(X):  # X: (N, T, F)
    Nn, Tt, Ff = X.shape
    flat = X.reshape(-1, Ff)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(Nn, Tt, Ff)

X_train_s = scale_blocks(X_train)
X_val_s   = scale_blocks(X_val)
X_test_s  = scale_blocks(X_test)

# Y(타깃)도 스케일: 같은 스케일러로 전체 F 차원 더미 구성 후 타깃 축만 추출
def scale_targets(Y):  # (N, 365, 4)
    Nn, Tt, Kk = Y.shape
    out = np.empty_like(Y)
    for n in range(Nn):
        dummy = np.zeros((Tt, F), dtype=np.float32)
        dummy[:, target_idxs] = Y[n]
        dummy_s = scaler.transform(dummy)
        out[n] = dummy_s[:, target_idxs]
    return out

Y_train_s = scale_targets(Y_train)
Y_val_s   = scale_targets(Y_val)
Y_test_s  = scale_targets(Y_test)

# ===== PyTorch Dataset =====
class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)  # (N, 365, 4)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

train_loader = DataLoader(SeqDataset(X_train_s, Y_train_s), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val_s,   Y_val_s),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SeqDataset(X_test_s,  Y_test_s),  batch_size=1,          shuffle=False)

# ===== 모델 정의 (멀티타깃) =====
class CNNLSTM(nn.Module):
    def __init__(self, input_features, n_targets, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS,
                 hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        self.future_steps = future_steps
        self.n_targets = n_targets
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_dim, future_steps * n_targets)

    def forward(self, x):         # x: (B, T, F)
        x = x.permute(0, 2, 1)    # (B, F, T)
        x = self.pool(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1)    # (B, T', H)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)            # (B, 365*4)
        return x.view(-1, self.future_steps, self.n_targets)  # (B, 365, 4)

n_targets = len(top4)
model = CNNLSTM(input_features=F, n_targets=n_targets).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 학습 =====
for epoch in range(1, EPOCHS+1):
    model.train()
    tr_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())

    model.eval()
    va_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va_losses.append(criterion(model(xb), yb).item())

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train {np.mean(tr_losses):.4f} | Val {np.mean(va_losses):.4f}")

# ===== 테스트(참고용) =====
model.eval()
te_losses = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        te_losses.append(criterion(model(xb), yb).item())
print(f"[Test MSE] {np.mean(te_losses):.6f}")

# ===== 산출물 저장 =====
torch.save(model.state_dict(), MODEL_PATH)
np.save("latest_input.npy", scale_blocks(df_daily.values[-PAST_STEPS:].reshape(1, PAST_STEPS, F))[0])
joblib.dump(scaler, SCALER_PATH)

pd.DataFrame({"feature": top4}).to_csv("core4_targets.csv", index=False, encoding="utf-8-sig")

with open("config.json", "w", encoding="utf-8") as f:
    json.dump({
        "PAST_STEPS": PAST_STEPS, "FUTURE_STEPS": FUTURE_STEPS,
        "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
        "HIDDEN_DIM": HIDDEN_DIM, "KERNEL_SIZE": KERNEL_SIZE, "DROPOUT": DROPOUT,
        "inputs": use_cols, "targets": top4
    }, f, ensure_ascii=False, indent=2)

print("Saved:", MODEL_PATH, SCALER_PATH, "latest_input.npy", "core4_targets.csv", "config.json")
