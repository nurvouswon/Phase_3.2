import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH + GAME DAY OVERLAYS]", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather Score [DEEP RESEARCH + GAME DAY OVERLAYS]")

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

def drop_high_na_low_var(df, thresh_na=1.0, thresh_var=0.0):
    cols_to_drop = []
    na_frac = df.isnull().mean()
    low_var_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).std() < thresh_var]
    for c in df.columns:
        if na_frac.get(c, 0) > thresh_na:
            cols_to_drop.append(c)
        elif c in low_var_cols:
            cols_to_drop.append(c)
    df2 = df.drop(columns=cols_to_drop, errors="ignore")
    return df2, cols_to_drop

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

# ==== GAME DAY OVERLAY MULTIPLIERS ====
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

# ==== META FEATURE CLUSTERING ====
def build_cluster_meta_features(df, cluster_corr=0.92):
    # Agglomerative: features with abs(corr) > cluster_corr are grouped, mean as meta-feature
    feature_cols = get_valid_feature_cols(df)
    X_full = df[feature_cols].fillna(0)
    corr_matrix = X_full.corr().abs()
    clusters = []
    used = set()
    for i, col in enumerate(corr_matrix.columns):
        if col in used:
            continue
        group = [col]
        for j, col2 in enumerate(corr_matrix.columns):
            if col != col2 and corr_matrix.loc[col, col2] > cluster_corr:
                group.append(col2)
        clusters.append(group)
        used.update(group)
    # Build new meta-features as the mean of each group
    meta_X = pd.DataFrame(index=df.index)
    for idx, group in enumerate(clusters):
        meta_X[f'cluster_{idx}'] = X_full[group].mean(axis=1)
    return meta_X

# ==== UI ====
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files (1-2 min, be patient)..."):
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

    # === Cluster method 1: One-feature-per-cluster (corr>0.97) ===
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols_1 = sorted(list(feat_cols_train & feat_cols_today))
    X_full = event_df[feature_cols_1].fillna(0)
    corr_matrix = X_full.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]
    feature_cols_1 = [col for col in feature_cols_1 if col not in to_drop]

    X1 = clean_X(event_df[feature_cols_1])
    y = event_df[target_col]
    X1_today = clean_X(today_df[feature_cols_1], train_cols=X1.columns)
    X1 = downcast_df(X1)
    X1_today = downcast_df(X1_today)
    nan_inf_check(X1, "X1 features")
    nan_inf_check(X1_today, "X1_today features")

    # === Cluster method 2: Meta-feature clusters (corr>0.92) ===
    meta_X2 = build_cluster_meta_features(event_df, cluster_corr=0.92)
    meta_X2_today = build_cluster_meta_features(today_df, cluster_corr=0.92)
    nan_inf_check(meta_X2, "Meta X2 features")
    nan_inf_check(meta_X2_today, "Meta X2 today features")

    # ==== SCALE, SPLIT, FIT ENSEMBLES ====
    X1_train, X1_val, y_train, y_val = train_test_split(
        X1, y, test_size=0.2, random_state=42, stratify=y
    )
    X2_train, X2_val, _, _ = train_test_split(
        meta_X2, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1_train_scaled = scaler1.fit_transform(X1_train)
    X1_val_scaled = scaler1.transform(X1_val)
    X1_today_scaled = scaler1.transform(X1_today)
    X2_train_scaled = scaler2.fit_transform(X2_train)
    X2_val_scaled = scaler2.transform(X2_val)
    meta_X2_today_scaled = scaler2.transform(meta_X2_today)

    # ---- Base models for both ensembles (identical config) ----
    def make_ensemble():
        return VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier(
                n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss',
                n_jobs=1, verbosity=1, tree_method='hist'
            )),
            ('lgb', lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)),
            ('cat', cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)),
            ('gb', GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)),
            ('lr', LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)),
        ], voting='soft', n_jobs=1)

    ensemble1 = make_ensemble()
    ensemble2 = make_ensemble()
    ensemble1.fit(X1_train_scaled, y_train)
    ensemble2.fit(X2_train_scaled, y_train)

    # Calibration for each
    ir1 = IsotonicRegression(out_of_bounds="clip")
    ir2 = IsotonicRegression(out_of_bounds="clip")
    y1_val_pred = ensemble1.predict_proba(X1_val_scaled)[:,1]
    y2_val_pred = ensemble2.predict_proba(X2_val_scaled)[:,1]
    y1_val_pred_cal = ir1.fit_transform(y1_val_pred, y_val)
    y2_val_pred_cal = ir2.fit_transform(y2_val_pred, y_val)
    y1_today_pred = ensemble1.predict_proba(X1_today_scaled)[:,1]
    y2_today_pred = ensemble2.predict_proba(meta_X2_today_scaled)[:,1]
    y1_today_pred_cal = ir1.transform(y1_today_pred)
    y2_today_pred_cal = ir2.transform(y2_today_pred)

    # Add all probabilities to the today_df
    today_df['hr_prob_cluster1'] = y1_today_pred_cal
    today_df['hr_prob_cluster2'] = y2_today_pred_cal
    today_df['hr_probability'] = (y1_today_pred_cal + y2_today_pred_cal) / 2  # Blended

    # ==== APPLY OVERLAY SCORING ====
    today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)
    today_df['final_hr_prob_cluster1'] = (today_df['hr_prob_cluster1'] * today_df['overlay_multiplier']).clip(0, 1)
    today_df['final_hr_prob_cluster2'] = (today_df['hr_prob_cluster2'] * today_df['overlay_multiplier']).clip(0, 1)

    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += [
        "hr_prob_cluster1", "hr_prob_cluster2", "hr_probability",
        "overlay_multiplier", "final_hr_prob_cluster1", "final_hr_prob_cluster2", "final_hr_probability"
    ]

    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard_top = leaderboard.head(30)

    st.markdown("### 🏆 **Top 30 HR Leaderboard (Blended/Hybrid)**")
    st.dataframe(leaderboard_top, use_container_width=True)

    st.markdown("### 🏆 **Cluster 1 Only**")
    st.dataframe(
        leaderboard.sort_values("final_hr_prob_cluster1", ascending=False).head(30),
        use_container_width=True
    )

    st.markdown("### 🏆 **Cluster 2 Only**")
    st.dataframe(
        leaderboard.sort_values("final_hr_prob_cluster2", ascending=False).head(30),
        use_container_width=True
    )

    # Downloads
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")
    st.download_button("⬇️ Download Top 30 Blended Leaderboard CSV", data=leaderboard_top.to_csv(index=False), file_name="top30_blended_leaderboard.csv")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
