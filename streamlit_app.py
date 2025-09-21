# streamlit_app.py
import json, joblib, torch, torch.nn as nn, numpy as np, pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Optimal Team Selector", layout="wide")

# --------------------------
# Load resources
# --------------------------
@st.cache_resource
def load_resources():
    scaler = joblib.load("scaler.pkl")
    with open("label_names.json") as f: label_names = json.load(f)

    class PlayerMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
        def forward(self, x): return self.net(x)

    model = PlayerMLP(input_dim=10, num_classes=len(label_names))
    model.load_state_dict(torch.load("model_state_dict.pt", map_location="cpu"))
    model.eval()
    return scaler, label_names, model

scaler, label_names, model = load_resources()

# --------------------------
# App interface
# --------------------------
st.title("🏀 NBA Optimal Team Selector (MLP)")
st.caption("Select a 5-year window, predict player performance, and build an optimal team!")

uploaded = st.file_uploader("Upload NBA CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df['season_start'] = df['season'].apply(lambda x: int(x.split('-')[0]))

    years = sorted(df['season_start'].unique())
    start_year = st.selectbox("Start Year", years)
    end_year = start_year + 4

    pool = df[(df['season_start'] >= start_year) & (df['season_start'] <= end_year)]
    if len(pool) < 100:
        st.warning("Not enough players in this window.")
    else:
        pool = pool.sample(100, random_state=42)

        features = ['pts','reb','ast','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp','net_rating']
        X = scaler.transform(pool[features].values)
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()[:,1]
        pool['prob_high'] = probs

        # Define roles
        roles = {
            "PG": pool.sort_values("ast_pct", ascending=False).iloc[0],
            "SG": pool.sort_values(["ts_pct","usg_pct"], ascending=[False,False]).iloc[0],
            "SF": pool.iloc[((pool[['pts','reb','ast']].sum(axis=1) - pool[['pts','reb','ast']].sum(axis=1).mean()).abs()).argmin()],
            "PF": pool.sort_values(["reb","oreb_pct"], ascending=[False,False]).iloc[0],
            "C": pool.sort_values(["reb","dreb_pct"], ascending=[False,False]).iloc[0],
        }

        st.subheader(f"Optimal Team ({start_year}-{end_year})")
        for role, player in roles.items():
            st.write(f"**{role}:** {player['player_name']} (P={player['prob_high']:.2f})")

        st.subheader("Top Predicted Performers")
        st.dataframe(pool[['player_name','pts','reb','ast','prob_high']].sort_values("prob_high", ascending=False).head(10))
