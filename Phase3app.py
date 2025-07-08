import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression

st.set_page_config("🚀 MLB HR Predictor: Next-Gen AI", layout="wide")
st.title("🚀 MLB HR Predictor — Fully Automated, AI-Supercharged, Pioneer Edition")

def safe_read(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def downcast_df(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int', 'int64', 'int32'])
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def nan_inf_check(X, name):
    arr = np.asarray(X, dtype=np.float32)
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def filter_features(X, X_today, max_feats=200, corr_thresh=0.97, importance_clip=0.0001):
    # Remove near-constant/duplicate features
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=const_cols, errors='ignore')
    X_today = X_today.drop(columns=const_cols, errors='ignore')

    # Remove highly correlated features (keep only one of each group)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    X = X.drop(columns=to_drop, errors='ignore')
    X_today = X_today.drop(columns=to_drop, errors='ignore')

    # Model-based feature importance
    model = xgb.XGBClassifier(n_estimators=30, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected = importances[importances > importance_clip].sort_values(ascending=False).head(max_feats).index.tolist()
    X = X[selected]
    X_today = X_today[selected]
    return X, X_today, importances

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)
        event_df = event_df.dropna(axis=1, how='all')
        today_df = today_df.dropna(axis=1, how='all')
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.reset_index(drop=True)
        today_df = today_df.reset_index(drop=True)

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # --- Feature selection ---
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    X = event_df[feature_cols].copy()
    y = event_df[target_col].astype(int)
    X_today = today_df[feature_cols].copy()

    X = downcast_df(X)
    X_today = downcast_df(X_today)
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # --- Automated Filtering ---
    X, X_today, importances = filter_features(X, X_today, max_feats=200, corr_thresh=0.97, importance_clip=0.00005)
    st.success(f"Final number of features after selection/filtering: {X.shape[1]}")
    st.write("Top 15 features and their importance:", importances.sort_values(ascending=False).head(15))

    # --- Split/scale
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # --- Meta-ensemble stacking
    st.write("Training meta-ensemble base models...")
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=90, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=90, max_depth=6, learning_rate=0.07, n_jobs=1)),
        ('cat', cb.CatBoostClassifier(iterations=90, depth=6, learning_rate=0.08, verbose=0, thread_count=1)),
        ('rf', RandomForestClassifier(n_estimators=70, max_depth=8, n_jobs=1)),
        ('gb', GradientBoostingClassifier(n_estimators=70, max_depth=6, learning_rate=0.08)),
        ('lr', LogisticRegression(max_iter=700, solver='lbfgs', n_jobs=1))
    ]
    meta_train = np.zeros((X_train_scaled.shape[0], len(base_models)))
    meta_val = np.zeros((X_val_scaled.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        model.fit(X_train_scaled, y_train)
        meta_train[:, i] = model.predict_proba(X_train_scaled)[:,1]
        meta_val[:, i] = model.predict_proba(X_val_scaled)[:,1]

    meta_learner = LogisticRegression(max_iter=500)
    meta_learner.fit(meta_train, y_train)
    val_meta_pred = meta_learner.predict_proba(meta_val)[:,1]
    auc = roc_auc_score(y_val, val_meta_pred)
    ll = log_loss(y_val, val_meta_pred)
    st.info(f"Meta-Stacked Ensemble Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # Calibrate
    st.write("Calibrating ensemble (isotonic regression)...")
    ir = IsotonicRegression(out_of_bounds="clip")
    val_meta_pred_cal = ir.fit_transform(val_meta_pred, y_val)

    # --- Predict for today
    meta_today = np.zeros((X_today_scaled.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        meta_today[:, i] = model.predict_proba(X_today_scaled)[:,1]
    y_today_pred = meta_learner.predict_proba(meta_today)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Automated AI)**")
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability"]
    leaderboard_top = today_df[leaderboard_cols].head(top_n)
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )
else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
