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
from xgboost import XGBRegressor

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

# ==== FEATURE ENGINEERING BLOCK ====
def feature_engineering(df):
    # DELTA/DIFFERENCE FEATURES
    windows_short = [3, 5, 7]
    windows_long = [14, 20, 30, 60]
    stat_bases = [
        'b_hr_per_pa', 'b_barrel_rate', 'b_hard_hit_rate', 'b_fb_rate', 'b_slg', 'b_avg_exit_velo',
        'p_hr_per_pa', 'p_barrel_rate', 'p_hard_hit_rate', 'p_fb_rate', 'p_slg', 'p_avg_exit_velo'
    ]
    for stat in stat_bases:
        for w_s in windows_short:
            for w_l in windows_long:
                col_short = f"{stat}_{w_s}"
                col_long = f"{stat}_{w_l}"
                if col_short in df.columns and col_long in df.columns:
                    df[f"delta_{stat}_{w_s}_{w_l}"] = df[col_short] - df[col_long]
    # HOT STREAK FLAGS
    for stat in stat_bases:
        for w_s in windows_short:
            for w_l in windows_long:
                col_short = f"{stat}_{w_s}"
                col_long = f"{stat}_{w_l}"
                if col_short in df.columns and col_long in df.columns:
                    df[f"is_hot_{stat}_{w_s}_{w_l}"] = (df[col_short] > df[col_long]).astype(int)
    # INTERACTION FEATURES
    for bstat in ['b_hr_per_pa_3', 'b_barrel_rate_3', 'b_slg_3', 'b_avg_exit_velo_3']:
        for pstat in ['p_hr_per_pa_3', 'p_barrel_rate_3', 'p_slg_3', 'p_avg_exit_velo_3']:
            if bstat in df.columns and pstat in df.columns:
                df[f"{bstat}_X_{pstat}"] = df[bstat] * df[pstat]
    # OUTLIER CLIPPING
    for col in df.columns:
        if 'hr_per_pa' in col:
            df[col] = df[col].clip(0, 0.25)
        if 'avg_exit_velo' in col:
            df[col] = df[col].clip(70, 120)
        if 'barrel_rate' in col or 'hard_hit_rate' in col or 'fb_rate' in col or 'pull_rate' in col:
            df[col] = df[col].clip(0, 1)
    return df

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

    # Feature engineering
    event_df = feature_engineering(event_df)
    today_df = feature_engineering(today_df)

    # Keep only intersection of features
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

    # ==== TOP N PRECISION LEADERBOARD ====
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "overlay_multiplier", "final_hr_probability"]

    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)

# --- Top-N Pool for Meta-Modeling ---
top_n = 30
leaderboard_top = leaderboard.head(top_n).copy()

# ======== AUTO-SELECT TOP META FEATURES ========
# Use top 15 most important features from the model (from the import_df above if available)
n_meta_feats = 15
# Attempt to grab top features from earlier importance DF if available
if "import_df" in locals():
    meta_features = import_df.head(n_meta_feats)["feature"].tolist()
else:
    meta_features = [c for c in leaderboard_top.columns if c not in ("player_name", "hr_probability", "overlay_multiplier", "final_hr_probability") and leaderboard_top[c].dtype != "O"][:n_meta_feats]
# Fallback if not enough features
if len(meta_features) < n_meta_feats:
    more_feats = [c for c in leaderboard_top.columns if c not in meta_features and leaderboard_top[c].dtype != "O"]
    meta_features += more_feats[:n_meta_feats - len(meta_features)]
meta_features = [f for f in meta_features if f in leaderboard_top.columns]
st.write("Meta-model selected features:", meta_features)

# ======== META-STACKED MODEL: XGBRegressor with KFold CV ========
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

meta_X = leaderboard_top[meta_features].copy()
meta_y = 1.0 - (leaderboard_top["final_hr_probability"].rank(method="first", ascending=False) - 1) / (len(leaderboard_top) - 1)
meta_pred = np.zeros(len(meta_X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(meta_X):
    xtr, xte = meta_X.iloc[train_idx], meta_X.iloc[test_idx]
    ytr = meta_y.iloc[train_idx]
    meta_model = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.14, subsample=0.85, colsample_bytree=0.9, random_state=42)
    meta_model.fit(xtr, ytr)
    meta_pred[test_idx] = meta_model.predict(xte)
leaderboard_top["meta_refined_prob"] = meta_pred

# Fit final meta-model for feature importances
meta_model_full = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.14, subsample=0.85, colsample_bytree=0.9, random_state=42)
meta_model_full.fit(meta_X, meta_y)

# Sort by meta-model result
refined_leaderboard = leaderboard_top.sort_values("meta_refined_prob", ascending=False).reset_index(drop=True)

st.markdown("### 🏅 **Refined Top 10 (Meta-Model Stacking, XGBRegressor, Deep Research)**")
st.dataframe(refined_leaderboard.head(10), use_container_width=True)
st.markdown("### 🏅 **Refined Top 15**")
st.dataframe(refined_leaderboard.head(15), use_container_width=True)
st.markdown("### 🏅 **Refined Top 30**")
st.dataframe(refined_leaderboard, use_container_width=True)

st.markdown("#### ⬇️ Download Refined Leaderboards")
st.download_button(
    "Download Refined Top 10 Leaderboard CSV",
    data=refined_leaderboard.head(10).to_csv(index=False),
    file_name="refined_top10_leaderboard.csv"
)
st.download_button(
    "Download Refined Top 15 Leaderboard CSV",
    data=refined_leaderboard.head(15).to_csv(index=False),
    file_name="refined_top15_leaderboard.csv"
)
st.download_button(
    "Download Refined Top 30 Leaderboard CSV",
    data=refined_leaderboard.to_csv(index=False),
    file_name="refined_top30_leaderboard.csv"
)

# Show meta-model feature importances
with st.expander("Meta-Model (XGBRegressor) Feature Importances"):
    meta_imp_df = pd.DataFrame({
        "feature": meta_features,
        "importance": meta_model_full.feature_importances_
    }).sort_values("importance", ascending=False)
    st.dataframe(meta_imp_df)

# Also show the classic top_n leaderboard for comparison
st.markdown(f"### 🏆 **Top {top_n} Precision HR Leaderboard (Deep Calibrated)**")
st.dataframe(leaderboard_top, use_container_width=True)

if len(leaderboard) > top_n:
    gap = leaderboard.loc[top_n - 1, "final_hr_probability"] - leaderboard.loc[top_n, "final_hr_probability"]
    st.markdown(f"**Confidence gap between #{top_n}/{top_n + 1}:** `{gap:.4f}`")
else:
    st.markdown(f"**Confidence gap:** (less than {top_n+1} players in leaderboard)")

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
