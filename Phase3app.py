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
import matplotlib.pyplot as plt

st.set_page_config("2️⃣ MLB HR Predictor — Best-in-World AI Stack", layout="wide")
st.title("2️⃣ MLB HR Predictor — Best-in-World AI Feature Stack")

# ==== FILE HELPERS ====
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

def find_duplicate_columns(df):
    return [col for col in df.columns if list(df.columns).count(col) > 1]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

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

def nan_inf_check(df, name):
    numeric_df = df.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors='coerce')
    arr = numeric_df.to_numpy(dtype=np.float64, copy=False)
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def overlay_multiplier(row):
    multiplier = 1.0
    wind_col = 'wind_mph'
    wind_dir_col = 'wind_dir_string'
    if wind_col in row and wind_dir_col in row:
        wind = row[wind_col]
        wind_dir = str(row[wind_dir_col]).lower()
        if pd.notnull(wind) and wind >= 10:
            if 'out' in wind_dir:
                multiplier *= 1.08
            elif 'in' in wind_dir:
                multiplier *= 0.93
    temp_col = 'temp'
    if temp_col in row and pd.notnull(row[temp_col]):
        base_temp = 70
        delta = row[temp_col] - base_temp
        multiplier *= 1.03 ** (delta / 10)
    humidity_col = 'humidity'
    if humidity_col in row and pd.notnull(row[humidity_col]):
        hum = row[humidity_col]
        if hum > 60:
            multiplier *= 1.02
        elif hum < 40:
            multiplier *= 0.98
    park_hr_col = 'park_hr_rate'
    if park_hr_col in row and pd.notnull(row[park_hr_col]):
        pf = max(0.85, min(1.20, float(row[park_hr_col])))
        multiplier *= pf
    return multiplier

# ==== UI ====
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
        dupes_event = find_duplicate_columns(event_df)
        if dupes_event:
            st.error(f"Duplicate columns in event file after deduplication: {set(dupes_event)}")
            st.stop()
        dupes_today = find_duplicate_columns(today_df)
        if dupes_today:
            st.error(f"Duplicate columns in today file after deduplication: {set(dupes_today)}")
            st.stop()
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    value_counts = event_df[target_col].value_counts(dropna=False).reset_index()
    value_counts.columns = [target_col, 'count']
    st.write("Value counts for hr_outcome:")
    st.dataframe(value_counts)

    # ==== SMART FEATURE SELECTION ====
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of initial features (all): {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col].astype(int)
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # --- Step 1: XGBoost Model-Based Selection ---
    N_KEEP = 200   # <---- HARD CODED: Keep top 200 features from XGB
    st.write("⚡ Running XGBoost for model-based feature importances...")
    xgb_fs = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    xgb_fs.fit(X, y)
    feature_importances = pd.Series(xgb_fs.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(N_KEEP).index.tolist()
    st.write(f"Top {N_KEEP} features by XGB importance: {top_features}")

    # --- Step 2: Permutation Importances on Validation ---
    N_PI = 30  # Number to keep after permutation (for sanity/stacking/robustness)
    st.write("⚡ Calculating permutation importances (validation set)...")
    X_train, X_val, y_train, y_val = train_test_split(X[top_features], y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today[top_features])
    xgb_fs.fit(X_train_scaled, y_train)
    perm_imp = permutation_importance(xgb_fs, X_val_scaled, y_val, n_repeats=10, random_state=42)
    pi_scores = pd.Series(perm_imp.importances_mean, index=top_features)
    pi_features = pi_scores.sort_values(ascending=False).head(N_PI).index.tolist()
    st.write(f"Top {N_PI} features by permutation importance: {pi_features}")

    # === Final features: intersection of XGB and Permutation top features ===
    final_features = [f for f in pi_features if f in top_features]
    st.write(f"Features used in final model: {final_features}")

    X_train_final = X_train[final_features]
    X_val_final = X_val[final_features]
    X_today_final = X_today[final_features]

    # --- Step 3: Meta-Feature Interactions (pairwise top 10) ---
    st.write("⚡ Creating meta-feature interactions for top features...")
    from itertools import combinations
    interaction_features = []
    top10 = final_features[:10]
    for f1, f2 in combinations(top10, 2):
        X_train_final[f'{f1}_x_{f2}'] = X_train_final[f1] * X_train_final[f2]
        X_val_final[f'{f1}_x_{f2}'] = X_val_final[f1] * X_val_final[f2]
        X_today_final[f'{f1}_x_{f2}'] = X_today_final[f1] * X_today_final[f2]
        interaction_features.append(f'{f1}_x_{f2}')
        X_train_final[f'{f1}_div_{f2}'] = np.where(X_train_final[f2]!=0, X_train_final[f1]/X_train_final[f2], 0)
        X_val_final[f'{f1}_div_{f2}'] = np.where(X_val_final[f2]!=0, X_val_final[f1]/X_val_final[f2], 0)
        X_today_final[f'{f1}_div_{f2}'] = np.where(X_today_final[f2]!=0, X_today_final[f1]/X_today_final[f2], 0)
        interaction_features.append(f'{f1}_div_{f2}')
    st.write(f"Added {len(interaction_features)} interaction features.")

    # --- Step 4: Robust Stacked Ensemble ---
    st.write("🧠 Training full stacked ensemble...")
    def make_base_learners():
        return [
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)),
            ('lgb', lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)),
            ('cat', cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)),
            ('gb', GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)),
            ('lr', LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1))
        ]

    # Out-of-fold meta-preds for stacking
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_preds = np.zeros(X_val_final.shape[0])
    meta_train = np.zeros((X_train_final.shape[0], 6))
    meta_val = np.zeros((X_val_final.shape[0], 6))
    base_models = make_base_learners()

    for i, (name, model) in enumerate(base_models):
        model.fit(X_train_final, y_train)
        meta_train[:, i] = model.predict_proba(X_train_final)[:,1]
        meta_val[:, i] = model.predict_proba(X_val_final)[:,1]
    meta_learner = LogisticRegression(max_iter=400)
    meta_learner.fit(meta_train, y_train)
    val_meta_pred = meta_learner.predict_proba(meta_val)[:,1]
    auc = roc_auc_score(y_val, val_meta_pred)
    ll = log_loss(y_val, val_meta_pred)
    st.info(f"Stacked Ensemble Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # Calibrate
    st.write("Calibrating meta-ensemble (isotonic regression)...")
    ir = IsotonicRegression(out_of_bounds="clip")
    val_meta_pred_cal = ir.fit_transform(val_meta_pred, y_val)

    # Predict for today
    meta_today = np.zeros((X_today_final.shape[0], 6))
    for i, (name, model) in enumerate(base_models):
        meta_today[:, i] = model.predict_proba(X_today_final)[:,1]
    y_today_pred = meta_learner.predict_proba(meta_today)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    st.write("Prediction distribution (today):")
    st.write(today_df['hr_probability'].describe())

    # Overlay scoring
    today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)

    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "overlay_multiplier", "final_hr_probability"]

    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)

    top_n = 30  # <---- HARD CODED: Top 30 leaderboard
    st.markdown(f"### 🏆 **Top {top_n} Precision HR Leaderboard (Deep Calibrated)**")
    leaderboard_top = leaderboard.head(top_n)
    st.dataframe(leaderboard_top, use_container_width=True)

    # Confidence gap: drop-off between last included and next
    if len(leaderboard) > top_n:
        gap = leaderboard.loc[top_n - 1, "final_hr_probability"] - leaderboard.loc[top_n, "final_hr_probability"]
        st.markdown(f"**Confidence gap between #{top_n}/{top_n + 1}:** `{gap:.4f}`")
    else:
        st.markdown(f"**Confidence gap:** (less than {top_n+1} players in leaderboard)")

    # Download full leaderboard and prediction CSVs
    st.download_button(
        f"⬇️ Download Full Prediction CSV",
        data=today_df.to_csv(index=False),
        file_name="today_hr_predictions.csv"
    )
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
