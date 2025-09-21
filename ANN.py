# ANN_player_performance.py
import pandas as pd, numpy as np, json, joblib, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Load CSV
# --------------------------
csv_path = "NBA_players.csv"
df = pd.read_csv(csv_path)

# --------------------------
# Choose 5-year window & pool of 100 players
# --------------------------
start_year, end_year = 2010, 2015
df['season_start'] = df['season'].apply(lambda x: int(x.split('-')[0]))
df = df[(df['season_start'] >= start_year) & (df['season_start'] <= end_year)]
if len(df) < 100:
    raise ValueError("Not enough players in the chosen window.")
df = df.sample(100, random_state=42)  # pool of 100

# --------------------------
# Features & target
# --------------------------
features = ['pts', 'reb', 'ast', 'oreb_pct', 'dreb_pct', 
            'usg_pct', 'ts_pct', 'ast_pct', 'gp', 'net_rating']

df['total_contrib'] = df['pts'] + df['reb'] + df['ast']
median_contrib = df['total_contrib'].median()
df['performance'] = (df['total_contrib'] >= median_contrib).astype(int)

X = df[features].values
y = df['performance'].values

# --------------------------
# Scale features
# --------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")  # save for Streamlit

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------
# Torch tensors & dataloaders
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test, dtype=torch.long).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# --------------------------
# Neural Network
# --------------------------
class PlayerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

model = PlayerMLP(input_dim=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# Training loop
# --------------------------
num_epochs = 30
loss_history = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model_state_dict.pt")
with open("label_names.json", "w") as f: 
    json.dump(["Low", "High"], f)

# --------------------------
# Evaluation
# --------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct/total:.4f}")

# --------------------------
# Select Optimal Team (unique players)
# --------------------------
df['prob_high'] = torch.softmax(model(torch.tensor(X, dtype=torch.float32).to(device)), dim=1)[:,1].cpu().detach().numpy()

assigned_players = set()
roles = {}

# Role heuristics with uniqueness
def pick_unique(df_sorted):
    for idx, row in df_sorted.iterrows():
        if row['player_name'] not in assigned_players:
            assigned_players.add(row['player_name'])
            return row
    return df_sorted.iloc[0]  # fallback, shouldn't happen

roles['PG'] = pick_unique(df.sort_values("ast_pct", ascending=False))
roles['SG'] = pick_unique(df.sort_values(["ts_pct","usg_pct"], ascending=[False,False]))
roles['SF'] = pick_unique(df.iloc[((df[['pts','reb','ast']].sum(axis=1) - df[['pts','reb','ast']].sum(axis=1).mean()).abs()).argsort()])
roles['PF'] = pick_unique(df.sort_values(["reb","oreb_pct"], ascending=[False,False]))
roles['C']  = pick_unique(df.sort_values(["reb","dreb_pct"], ascending=[False,False]))

print("\nOptimal Team (Unique Players):")
for role, player in roles.items():
    print(f"{role}: {player['player_name']} (P={player['prob_high']:.2f})")
