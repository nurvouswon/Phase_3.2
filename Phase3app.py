import streamlit as st import pandas as pd import numpy as np from sklearn.model_selection import train_test_split, StratifiedKFold from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier from sklearn.linear_model import LogisticRegression from sklearn.metrics import roc_auc_score, log_loss from sklearn.preprocessing import StandardScaler import xgboost as xgb import lightgbm as lgb import catboost as cb from sklearn.isotonic import IsotonicRegression import shap import optuna from alibi_detect.cd import KSDrift

st.set_page_config("🏆 MLB HR Predictor — AI Top 10 World Class", layout="wide") st.title("🏆 MLB Home Run Predictor — AI-Powered Supercharged")

def safe_read(path): if path.name.endswith('.parquet'): return pd.read_parquet(path) return pd.read_csv(path, low_memory=False)

File Upload

event_file = st.file_uploader("Event-Level CSV (required)", type=['csv', 'parquet']) today_file = st.file_uploader("Today's CSV (required)", type=['csv', 'parquet'])

if event_file and today_file: event_df, today_df = safe_read(event_file), safe_read(today_file) event_df, today_df = event_df.dropna(axis=1, how='all'), today_df.dropna(axis=1, how='all')

if 'hr_outcome' not in event_df:
    st.error("'hr_outcome' column missing!"); st.stop()

# Features
mutual_feats = list(set(event_df.columns) & set(today_df.columns) - {'hr_outcome', 'game_date', 'player_name', 'team', 'game_time'})
X, y = event_df[mutual_feats].fillna(-1), event_df['hr_outcome']
X_today = today_df[mutual_feats].fillna(-1)

# Drift Detection
drift = KSDrift(X.values).predict(X_today.values)
if drift['data']['is_drift']:
    st.warning("⚠️ Drift detected: Consider retraining.")

# Scaling
scaler = StandardScaler().fit(X)
X_scaled, X_today_scaled = scaler.transform(X), scaler.transform(X_today)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, stratify=y, random_state=42)

# Bayesian Optimization
def opt_objective(trial):
    clf = xgb.XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators',50,150),
        max_depth=trial.suggest_int('max_depth',3,8),
        learning_rate=trial.suggest_float('lr',0.01,0.2), eval_metric='logloss')
    clf.fit(X_train, y_train)
    return roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])

study = optuna.create_study(direction='maximize')
study.optimize(opt_objective, n_trials=20)

# Base Models
xgb_clf = xgb.XGBClassifier(**study.best_params, eval_metric='logloss')
lgb_clf = lgb.LGBMClassifier(n_estimators=80)
cat_clf = cb.CatBoostClassifier(iterations=80, verbose=0)
rf_clf = RandomForestClassifier(n_estimators=60)

# Stacking
stack = StackingClassifier(
    estimators=[('xgb',xgb_clf),('lgb',lgb_clf),('cat',cat_clf),('rf',rf_clf)],
    final_estimator=LogisticRegression(), cv=StratifiedKFold(5))

stack.fit(X_train, y_train)

# SHAP Interpretation
explainer = shap.TreeExplainer(xgb_clf.fit(X_train, y_train))
shap_values = explainer.shap_values(X_today)
st.markdown("### 🌳 SHAP Feature Importance")
shap.summary_plot(shap_values, X_today, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')

# Predictions & Calibration
val_preds = stack.predict_proba(X_val)[:,1]
ir = IsotonicRegression().fit(val_preds, y_val)
today_preds = ir.transform(stack.predict_proba(X_today)[:,1])

today_df['HR_Prob'] = today_preds

# Meta-Booster
today_df.sort_values('HR_Prob', ascending=False, inplace=True)
today_df['Rank_Gap'] = today_df['HR_Prob'].diff().fillna(0)

meta_feats = today_df[['HR_Prob', 'Rank_Gap']]
meta_y = today_df['HR_Prob'].copy()
meta_y.iloc[:10] += 0.1

meta_boost = GradientBoostingClassifier(n_estimators=100)
meta_boost.fit(meta_feats, meta_y > meta_y.mean())
today_df['Meta_Score'] = meta_boost.predict_proba(meta_feats)[:,1]

today_df.sort_values('Meta_Score', ascending=False, inplace=True)

# Leaderboard
leaderboard = today_df[['player_name', 'team', 'game_time', 'HR_Prob', 'Meta_Score']].head(30)
leaderboard['HR_Prob'] = leaderboard['HR_Prob'].round(4)
leaderboard['Meta_Score'] = leaderboard['Meta_Score'].round(4)

st.markdown("## 🥇 Top-30 HR Leaderboard (AI Supercharged)")
st.dataframe(leaderboard, use_container_width=True)

st.download_button("⬇️ Download Leaderboard CSV", leaderboard.to_csv(index=False), "hr_leaderboard.csv")

else: st.warning("Please upload required CSVs.")
