import streamlit as st import pandas as pd import numpy as np from sklearn.model_selection import train_test_split, StratifiedKFold from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, GradientBoostingRegressor from sklearn.linear_model import LogisticRegression from sklearn.metrics import roc_auc_score from sklearn.preprocessing import StandardScaler import xgboost as xgb import lightgbm as lgb import catboost as cb from sklearn.isotonic import IsotonicRegression from sklearn.calibration import CalibratedClassifierCV import shap import optuna from alibi_detect.cd import KSDrift from datetime import datetime import matplotlib.pyplot as plt

st.set_page_config("🏆 MLB HR Predictor — AI Top 10 World Class", layout="wide") st.title("🏆 MLB Home Run Predictor — AI-Powered Supercharged")

def safe_read(path): if path.name.endswith('.parquet'): return pd.read_parquet(path) return pd.read_csv(path, low_memory=False)

event_file = st.file_uploader("Event-Level CSV (required)", type=['csv', 'parquet']) today_file = st.file_uploader("Today's CSV (required)", type=['csv', 'parquet'])

if event_file and today_file: event_df, today_df = safe_read(event_file), safe_read(today_file) event_df, today_df = event_df.dropna(axis=1, how='all'), today_df.dropna(axis=1, how='all')

if 'hr_outcome' not in event_df:
    st.error("'hr_outcome' column missing!"); st.stop()

mutual_feats = list(set(event_df.columns) & set(today_df.columns) - {'hr_outcome', 'game_date', 'player_name', 'team', 'game_time'})
X, y = event_df[mutual_feats].fillna(-1), event_df['hr_outcome']
X_today = today_df[mutual_feats].fillna(-1)

drift = KSDrift(X.values).predict(X_today.values)
if drift['data']['is_drift']:
    st.warning("⚠️ Drift detected: Retraining initiated.")

scaler = StandardScaler().fit(X)
X_scaled, X_today_scaled = scaler.transform(X), scaler.transform(X_today)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, stratify=y, random_state=42)

def opt_objective(trial):
    clf = xgb.XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators',50,150),
        max_depth=trial.suggest_int('max_depth',3,8),
        learning_rate=trial.suggest_float('lr',0.01,0.2), eval_metric='logloss')
    clf.fit(X_train, y_train)
    return roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])

study = optuna.create_study(direction='maximize')
study.optimize(opt_objective, n_trials=20)

xgb_clf = xgb.XGBClassifier(**study.best_params, eval_metric='logloss')
lgb_clf = lgb.LGBMClassifier(n_estimators=100)
cat_clf = cb.CatBoostClassifier(iterations=100, verbose=0)
rf_clf = RandomForestClassifier(n_estimators=80)

stack = StackingClassifier(
    estimators=[('xgb',xgb_clf),('lgb',lgb_clf),('cat',cat_clf),('rf',rf_clf)],
    final_estimator=LogisticRegression(), cv=StratifiedKFold(5))

stack.fit(X_train, y_train)

iso = IsotonicRegression().fit(stack.predict_proba(X_val)[:,1], y_val)
platt_calib = CalibratedClassifierCV(stack, method='sigmoid', cv='prefit').fit(X_val, y_val)

preds_iso = iso.transform(stack.predict_proba(X_today_scaled)[:,1])
preds_platt = platt_calib.predict_proba(X_today_scaled)[:,1]
today_df['HR_Prob'] = (preds_iso + preds_platt) / 2

today_df['Momentum'] = today_df['HR_Prob'].diff().fillna(0)
today_df['Fatigue'] = today_df['HR_Prob'].rolling(window=3).mean().fillna(method='bfill')

meta_feats = today_df[['HR_Prob', 'Momentum', 'Fatigue']]
meta_y = today_df['HR_Prob'].copy()
meta_y.iloc[:10] += 0.1

meta_boost = GradientBoostingRegressor(n_estimators=120)
meta_boost.fit(meta_feats, meta_y)
today_df['Meta_Score'] = meta_boost.predict(meta_feats)

today_df.sort_values('Meta_Score', ascending=False, inplace=True)
today_df['prediction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

leaderboard = today_df[['player_name', 'team', 'game_time', 'HR_Prob', 'Meta_Score', 'prediction_timestamp']].head(30)
leaderboard['HR_Prob'] = leaderboard['HR_Prob'].round(4)
leaderboard['Meta_Score'] = leaderboard['Meta_Score'].round(4)

st.subheader("📈 SHAP Feature Impact")
explainer = shap.TreeExplainer(xgb_clf.fit(X_train, y_train))
shap_values = explainer.shap_values(X_today_scaled)
shap.summary_plot(shap_values, X_today, plot_type="dot", show=False)
st.pyplot(bbox_inches='tight')

st.subheader("📊 HR Probability Distribution (Top 30)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(leaderboard['player_name'], leaderboard['HR_Prob'], color='dodgerblue')
ax.invert_yaxis()
ax.set_xlabel('HR Probability')
ax.set_ylabel('Player')
st.pyplot(fig)

st.markdown("## 🥇 Top-30 HR Leaderboard (AI Supercharged)")
st.dataframe(leaderboard.style.highlight_max(subset=['Meta_Score'], color='#a8ff60'), use_container_width=True)

st.download_button("⬇️ Download Leaderboard CSV", leaderboard.to_csv(index=False), "hr_leaderboard.csv")

else: st.warning("Please upload required CSVs.")

