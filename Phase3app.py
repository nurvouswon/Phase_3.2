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

    # === Deep Research Feature Engineering ===
    # event_df = feature_engineering(event_df)
    # today_df = feature_engineering(today_df)

    # Only keep features present in BOTH event and today sets (intersection)
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features in both event/today: {len(feature_cols)}")
    st.write(f"Features being used: {feature_cols}")

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # === CLUSTERING UPGRADE (meta-feature engineering) ===
    st.markdown("## 🧩 Feature Clustering & Meta-Features")

    def cluster_and_build_meta_features(X, X_today, num_clusters=20):
        # Drop constant/all-NaN/non-finite columns before clustering
        drop_cols = []
        for col in X.columns:
            if X[col].isnull().all() or X[col].std() == 0 or not np.isfinite(X[col]).all():
                drop_cols.append(col)
        if drop_cols:
            st.warning(f"Dropping {len(drop_cols)} constant/all-NaN/non-finite features before clustering: {drop_cols[:10]}{'...' if len(drop_cols) > 10 else ''}")
            X = X.drop(columns=drop_cols)
            X_today = X_today.drop(columns=drop_cols, errors="ignore")
        # Compute abs correlation matrix and cluster
        corr = X.corr().abs()
        distance_matrix = 1 - corr
        distance_matrix = distance_matrix.fillna(0)
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster
        dist_array = squareform(distance_matrix.values, checks=False)
        num_clusters = min(num_clusters, len(X.columns))
        Z = linkage(dist_array, method='average')
        cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
        cluster_map = dict(zip(X.columns, cluster_labels))
        # Create meta-features: mean per cluster
        meta_X = pd.DataFrame(index=X.index)
        meta_X_today = pd.DataFrame(index=X_today.index)
        for k in np.unique(cluster_labels):
            cluster_features = [col for col, lab in cluster_map.items() if lab == k]
            meta_X[f"cluster_{k}_mean"] = X[cluster_features].mean(axis=1)
            meta_X_today[f"cluster_{k}_mean"] = X_today[cluster_features].mean(axis=1)
        st.write(f"Feature clusters created: {meta_X.shape[1]}")
        return meta_X, meta_X_today, cluster_map

    num_clusters = min(20, len(feature_cols))
    meta_X, meta_X_today, cluster_map = cluster_and_build_meta_features(X, X_today, num_clusters=num_clusters)
    X = meta_X
    X_today = meta_X_today
    st.write(f"Meta-feature columns for modeling: {list(X.columns)}")

    st.write("Splitting for validation and scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # =========== DEEP RESEARCH ENSEMBLE (SOFT VOTING) ===========
    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss',
        n_jobs=1, verbosity=1, tree_method='hist'
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)
    lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)

    model_status = []
    models_for_ensemble = []
    importances = {}
    try:
        xgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('xgb', xgb_clf))
        model_status.append('XGB OK')
        importances['XGB'] = xgb_clf.feature_importances_
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")
    try:
        lgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lgb', lgb_clf))
        model_status.append('LGB OK')
        importances['LGB'] = lgb_clf.feature_importances_
    except Exception as e:
        st.warning(f"LightGBM failed: {e}")
    try:
        cat_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('cat', cat_clf))
        model_status.append('CatBoost OK')
        importances['CatBoost'] = cat_clf.feature_importances_
    except Exception as e:
        st.warning(f"CatBoost failed: {e}")
    try:
        rf_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('rf', rf_clf))
        model_status.append('RF OK')
        importances['RF'] = rf_clf.feature_importances_
    except Exception as e:
        st.warning(f"RandomForest failed: {e}")
    try:
        gb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('gb', gb_clf))
        model_status.append('GB OK')
        importances['GB'] = gb_clf.feature_importances_
    except Exception as e:
        st.warning(f"GBM failed: {e}")
    try:
        lr_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lr', lr_clf))
        model_status.append('LR OK')
        importances['LR'] = np.abs(lr_clf.coef_[0])
    except Exception as e:
        st.warning(f"LogReg failed: {e}")

    st.info("Model training status: " + ', '.join(model_status))
    if not models_for_ensemble:
        st.error("All models failed to train! Try reducing features or rows.")
        st.stop()

    st.write("Fitting ensemble (soft voting)...")
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_scaled, y_train)

    # =========== FEATURE IMPORTANCE DIAGNOSTICS ===========
    st.markdown("## 🔍 Feature Importances (Mean of Tree Models)")
    tree_keys = [k for k in importances.keys() if k in ("XGB", "LGB", "CatBoost", "RF", "GB")]
    if tree_keys:
        tree_importances = np.mean([importances[k] for k in tree_keys], axis=0)
        import_df = pd.DataFrame({
            "feature": X.columns,
            "importance": tree_importances
        }).sort_values("importance", ascending=False)
        st.dataframe(import_df.head(30), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.barh(import_df.head(20)["feature"][::-1], import_df.head(20)["importance"][::-1])
        ax.set_title("Top 20 Feature Importances (Avg of Tree Models)")
        st.pyplot(fig)
    else:
        st.warning("Tree model feature importances not available.")

    # =========== VALIDATION ===========
    st.write("Validating (out-of-fold, not test-leak)...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # =========== CALIBRATION (Isotonic Regression) ===========
    st.write("Calibrating prediction probabilities (isotonic regression, deep research)...")
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    # =========== PREDICT ===========
    st.write("Predicting HR probability for today (calibrated)...")
    y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # ==== APPLY OVERLAY SCORING ====
    st.write("Applying post-prediction game day overlay scoring (weather, park, etc)...")
    if 'hr_probability' in today_df.columns:
        today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
        today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)
    else:
        today_df['final_hr_probability'] = today_df['hr_probability']

    # ==== TOP N PRECISION LEADERBOARD WITH CONFIDENCE GAP ====
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "overlay_multiplier", "final_hr_probability"]

    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)

    # Show debug: Print stats for 3 players at every step
    debug_names = ["Agustin Ramirez", "Matt Wallner", "Trenton Brooks"]

    st.markdown("## 🐞 Debug: Raw Stats from today_df")
    for name in debug_names:
        st.write(f"{name} - Raw features:", today_df[today_df['player_name'] == name])

    st.markdown("## 🐞 Debug: Model Input Vector (X_today)")
    for name in debug_names:
        ix = today_df["player_name"] == name
        st.write(f"{name} - X_today input:", pd.DataFrame(X_today[ix], columns=X_today.columns if hasattr(X_today, "columns") else X.columns))

    st.markdown("## 🐞 Debug: Model Output Probabilities")
    for name in debug_names:
        prob = today_df.loc[today_df['player_name'] == name, "hr_probability"].values
        st.write(f"{name} - hr_probability:", prob)

    # Change this value for Top 10 or Top 30 leaderboard
    top_n = 30

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
