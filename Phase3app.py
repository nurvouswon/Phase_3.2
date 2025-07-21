import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import timedelta
import shap
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb
import catboost
from betacal import BetaCalibration
import joblib
import os
import matplotlib.pyplot as plt

# --- Set random seeds for reproducibility ---
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

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

def dedup_columns(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status', 'team_code', 'time'}
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
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status','team_code','time'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def nan_inf_check(X, name):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
        nans = X_num.isna().sum().sum()
        infs = np.isinf(X_num.to_numpy(dtype=np.float64, copy=False)).sum()
    else:
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def feature_debug(X):
    st.write("🛡️ **Feature Debugging:**")
    st.write("Data types:", X.dtypes.value_counts())
    st.write("Columns with object dtype:", X.select_dtypes('O').columns.tolist())
    for col in X.columns:
        if X[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            st.write(f"Column {col} is {X[col].dtype}, unique values: {X[col].unique()[:8]}")
    st.write("Missing values per column (top 10):", X.isna().sum().sort_values(ascending=False).head(10))

def overlay_multiplier(row):
    """
    Upgraded overlay multiplier using:
    - Batted ball pull/oppo/fb/gb/air for both batter and pitcher.
    - Full wind direction parsing.
    - Handedness logic.
    - Amplified, but reasonable, effects for correct context.
    """
    edge = 1.0

    # --- Get relevant values safely ---
    wind = row.get("wind_mph", np.nan)
    wind_dir = str(row.get("wind_dir_string", "")).lower().strip()
    temp = row.get("temp", np.nan)
    humidity = row.get("humidity", np.nan)
    park_hr_col = 'park_hr_rate'

    # Batter
    b_hand = str(row.get('stand', row.get('batter_hand', 'R'))).upper() or "R"
    b_pull = row.get("pull_rate", np.nan)
    b_oppo = row.get("oppo_rate", np.nan)
    b_fb = row.get("fb_rate", np.nan)
    b_air = row.get("air_rate", np.nan)
    b_ld = row.get("ld_rate", np.nan)
    b_pu = row.get("pu_rate", np.nan)
    b_hot = row.get("b_hr_per_pa_7", np.nan)  # rolling HR/PA
    
    # Pitcher
    p_hand = str(row.get("pitcher_hand", "")).upper() or "R"
    p_fb = row.get("p_fb_rate", row.get("fb_rate", np.nan))
    p_gb = row.get("p_gb_rate", row.get("gb_rate", np.nan))
    p_air = row.get("p_air_rate", row.get("air_rate", np.nan))
    p_ld = row.get("p_ld_rate", row.get("ld_rate", np.nan))
    p_pu = row.get("p_pu_rate", row.get("pu_rate", np.nan))

    # --- Wind logic: Amplified/Smart ---
    wind_factor = 1.0
    if wind is not None and pd.notnull(wind) and wind >= 7 and wind_dir and wind_dir != "nan":
        # Strong outfield wind: boost for right context
        for field, out_bonus, in_bonus, field_side in [
            ("rf", 1.19, 0.85, ("R", "oppo")),  # RHH oppo or LHH pull to RF
            ("lf", 1.19, 0.85, ("R", "pull")),  # RHH pull or LHH oppo to LF
            ("cf", 1.11, 0.90, ("ANY", "fb")),  # Any FB to CF
        ]:
            if field in wind_dir:
                if "out" in wind_dir or "o" in wind_dir:
                    if field == "rf":
                        if (b_hand == "R" and b_oppo > 0.26) or (b_hand == "L" and b_pull > 0.35):
                            wind_factor *= out_bonus
                    elif field == "lf":
                        if (b_hand == "R" and b_pull > 0.35) or (b_hand == "L" and b_oppo > 0.26):
                            wind_factor *= out_bonus
                    elif field == "cf":
                        if b_fb > 0.21 or b_air > 0.34:
                            wind_factor *= out_bonus
                if "in" in wind_dir or "i" in wind_dir:
                    if field == "rf":
                        if (b_hand == "R" and b_oppo > 0.26) or (b_hand == "L" and b_pull > 0.35):
                            wind_factor *= in_bonus
                    elif field == "lf":
                        if (b_hand == "R" and b_pull > 0.35) or (b_hand == "L" and b_oppo > 0.26):
                            wind_factor *= in_bonus
                    elif field == "cf":
                        if b_fb > 0.21 or b_air > 0.34:
                            wind_factor *= in_bonus

        # Add extra boost/fade for high-flyball hitters facing high-flyball pitchers
        if p_fb is not np.nan and p_fb > 0.25 and (b_fb > 0.23 or b_air > 0.36):
            if "out" in wind_dir or "o" in wind_dir:
                wind_factor *= 1.09
            elif "in" in wind_dir or "i" in wind_dir:
                wind_factor *= 0.94
        # Fade for extreme groundball pitchers
        if p_gb is not np.nan and p_gb > 0.53:
            wind_factor *= 0.93

    # --- Hot streak logic (recent HR/PA) ---
    if b_hot is not np.nan and b_hot > 0.09:
        edge *= 1.04
    elif b_hot is not np.nan and b_hot < 0.025:
        edge *= 0.97

    # --- Weather logic ---
    if temp is not None and pd.notnull(temp):
        edge *= 1.036 ** ((temp - 70) / 10)
    if humidity is not None and pd.notnull(humidity):
        if humidity > 65: edge *= 1.02
        elif humidity < 35: edge *= 0.98

    # --- Park HR Rate logic ---
    if park_hr_col in row and pd.notnull(row[park_hr_col]):
        pf = max(0.80, min(1.22, float(row[park_hr_col])))
        edge *= pf

    # --- Multiply wind after everything else, so it can be "amplifying" ---
    edge *= wind_factor

    # --- Clamp for sanity ---
    return float(np.clip(edge, 0.70, 1.36))

def rate_weather(row):
    ratings = {}
    temp = row.get("temp", np.nan)
    if pd.isna(temp):
        ratings["temp_rating"] = "?"
    elif 68 <= temp <= 85:
        ratings["temp_rating"] = "Excellent"
    elif 60 <= temp < 68 or 85 < temp <= 92:
        ratings["temp_rating"] = "Good"
    elif 50 <= temp < 60 or 92 < temp <= 98:
        ratings["temp_rating"] = "Fair"
    else:
        ratings["temp_rating"] = "Poor"
    humidity = row.get("humidity", np.nan)
    if pd.isna(humidity):
        ratings["humidity_rating"] = "?"
    elif 45 <= humidity <= 65:
        ratings["humidity_rating"] = "Excellent"
    elif 30 <= humidity < 45 or 65 < humidity <= 80:
        ratings["humidity_rating"] = "Good"
    elif 15 <= humidity < 30 or 80 < humidity <= 90:
        ratings["humidity_rating"] = "Fair"
    else:
        ratings["humidity_rating"] = "Poor"
    wind = row.get("wind_mph", np.nan)
    wind_dir = str(row.get("wind_dir_string", "")).lower()
    if pd.isna(wind):
        ratings["wind_rating"] = "?"
    elif wind < 6:
        ratings["wind_rating"] = "Excellent"
    elif 6 <= wind < 12:
        ratings["wind_rating"] = "Good"
    elif 12 <= wind < 18:
        if "out" in wind_dir:
            ratings["wind_rating"] = "Good"
        elif "in" in wind_dir:
            ratings["wind_rating"] = "Fair"
        else:
            ratings["wind_rating"] = "Fair"
    else:
        if "out" in wind_dir:
            ratings["wind_rating"] = "Fair"
        elif "in" in wind_dir:
            ratings["wind_rating"] = "Poor"
        else:
            ratings["wind_rating"] = "Poor"
    condition = str(row.get("condition", "")).lower()
    if "clear" in condition or "sun" in condition or "outdoor" in condition:
        ratings["condition_rating"] = "Excellent"
    elif "cloud" in condition or "partly" in condition:
        ratings["condition_rating"] = "Good"
    elif "rain" in condition or "fog" in condition:
        ratings["condition_rating"] = "Poor"
    else:
        ratings["condition_rating"] = "Fair"
    return pd.Series(ratings)

def drift_check(train, today, n=5):
    drifted = []
    for c in train.columns:
        if c not in today.columns: continue
        tmean = np.nanmean(train[c])
        tstd = np.nanstd(train[c])
        dmean = np.nanmean(today[c])
        if tstd > 0 and abs(tmean - dmean) / tstd > n:
            drifted.append(c)
    return drifted

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)
    for col in X.columns:
        lower = X[col].quantile(limits[0])
        upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def stickiness_rank_boost(df, top_k=10, stickiness_boost=0.18, prev_rank_col=None, hr_col='hr_probability'):
    stick = df[hr_col].copy()
    if prev_rank_col and prev_rank_col in df.columns:
        prev_rank = df[prev_rank_col].rank(method='min', ascending=False)
        stick = stick + stickiness_boost * (prev_rank <= top_k)
    else:
        stick.iloc[:top_k] += stickiness_boost
    return stick

def auto_feature_crosses(X, max_cross=24, template_cols=None):
    cross_names = []
    if template_cols is not None:
        for name in template_cols:
            f1, f2 = name.split('*')
            X[name] = X[f1] * X[f2]
            cross_names.append(name)
        X = X.copy()
        return X, cross_names
    means = X.mean()
    var_scores = {}
    cols = list(X.columns)
    for i, f1 in enumerate(cols):
        for j, f2 in enumerate(cols):
            if i >= j: continue
            cross = X[f1] * X[f2]
            var_scores[(f1, f2)] = cross.var()
    top_pairs = sorted(var_scores.items(), key=lambda kv: -kv[1])[:max_cross]
    for (f1, f2), _ in top_pairs:
        name = f"{f1}*{f2}"
        X[name] = X[f1] * X[f2]
        cross_names.append(name)
    X = X.copy()
    return X, cross_names

# --------- ADVANCED OUTLIER ENSEMBLE ---------
def remove_outliers_ensemble(X, y, contamination=0.012):
    mask1 = IsolationForest(contamination=contamination, random_state=SEED).fit_predict(X) == 1
    try:
        from sklearn.neighbors import LocalOutlierFactor
        mask2 = LocalOutlierFactor(contamination=contamination).fit_predict(X) == 1
    except ImportError:
        mask2 = mask1
    mask = mask1 & mask2  # keep only samples not flagged by both
    return X[mask], y[mask]

def smooth_labels(y, smoothing=0.02):
    y = np.asarray(y)
    y_smooth = y.copy().astype(float)
    y_smooth[y == 1] = 1 - smoothing
    y_smooth[y == 0] = smoothing
    return y_smooth

# --------- MODEL STACKING/ENSEMBLE ---------
def get_base_models():
    models = [
        ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.07, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0, random_state=SEED)),
        ("lgb", lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.07, n_jobs=1, random_state=SEED)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=1)),
        ("cat", catboost.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.07, verbose=0, random_seed=SEED)),
        ("lr", LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED))
    ]
    return models

def get_meta_model():
    return Ridge(random_state=SEED)

# ---- APP START ----

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload Today's File (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    event_df = pd.read_csv(event_file) if str(event_file.name).endswith('.csv') else pd.read_parquet(event_file)
    today_df = pd.read_csv(today_file) if str(today_file.name).endswith('.csv') else pd.read_parquet(today_file)

    st.write(f"event_df shape: {event_df.shape}, today_df shape: {today_df.shape}")
    st.write(f"event_df memory usage (MB): {event_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    st.write(f"today_df memory usage (MB): {today_df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # ---- Feature Filtering ----
    feature_cols = sorted(list(set(get_valid_feature_cols(event_df)) & set(get_valid_feature_cols(today_df))))
    st.write(f"Feature count before filtering: {len(feature_cols)}")
    X = clean_X(event_df[feature_cols])
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    feature_debug(X)

    # ---- Winsorize ----
    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)

    # ---- Feature Crosses ----
    X, cross_names = auto_feature_crosses(X, max_cross=12)
    X_today, _ = auto_feature_crosses(X_today, max_cross=12, template_cols=cross_names)
    st.write(f"Cross features created: {cross_names}")
    st.write(f"After cross sync: X cols {X.shape[1]}, X_today cols {X_today.shape[1]}")

    # ========== OUTLIER REMOVAL UPGRADE ==========
    y = event_df[target_col].astype(int)
    X, y = remove_outliers_ensemble(X, y, contamination=0.012)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)
    st.write(f"Rows after outlier removal: {X.shape[0]}")

    # ========== OOS SPLIT ==========
    OOS_ROWS = 5000
    X_train, X_oos = X.iloc[:-OOS_ROWS].copy(), X.iloc[-OOS_ROWS:].copy()
    y_train, y_oos = y.iloc[:-OOS_ROWS].copy(), y.iloc[-OOS_ROWS:].copy()
    st.write(f"🔒 Automatically reserving last {OOS_ROWS} rows for Out-of-Sample (OOS) test. Using first {X_train.shape[0]} for training.")

    # ========== MODEL STACKING/ENSEMBLE ==========
    n_splits, n_repeats = 5, 1
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    scaler = StandardScaler()
    test_fold_probas = np.zeros((X_today.shape[0], 5))  # for each model
    val_fold_probas = np.zeros((X_train.shape[0], 5))
    fold_times = []

    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train)):
        t_fold_start = time.time()
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        sc = scaler.fit(X_tr)
        X_tr_scaled = sc.transform(X_tr)
        X_va_scaled = sc.transform(X_va)
        X_today_scaled = sc.transform(X_today)

        for i, (model_name, model) in enumerate(get_base_models()):
            m = model.fit(X_tr_scaled, y_tr)
            val_fold_probas[va_idx, i] = m.predict_proba(X_va_scaled)[:, 1]
            test_fold_probas[:, i] += m.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * ((n_splits * n_repeats) - (fold + 1))
        st.write(f"Fold {fold + 1} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    # Meta-model stacking
    meta_model = get_meta_model()
    meta_model.fit(val_fold_probas, y_train)
    y_val_stack = meta_model.predict(val_fold_probas)
    y_today_stack = meta_model.predict(test_fold_probas)

    # ========== PERMUTATION IMPORTANCE POST-TRAINING ==========
    pi = permutation_importance(meta_model, val_fold_probas, y_train, n_repeats=3, random_state=SEED)
    low_importance = [i for i, imp in enumerate(pi.importances_mean) if imp <= 0]
    if low_importance:
        st.write(f"Removing {len(low_importance)} meta-features with <=0 importance")
        val_fold_probas = np.delete(val_fold_probas, low_importance, axis=1)
        test_fold_probas = np.delete(test_fold_probas, low_importance, axis=1)

    # ========== CALIBRATION ==========
    bc = BetaCalibration(parameters="abm")
    bc.fit(y_val_stack.reshape(-1, 1), y_train)
    y_val_beta = bc.predict(y_val_stack.reshape(-1, 1))
    y_today_beta = bc.predict(y_today_stack.reshape(-1, 1))
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_iso = ir.fit_transform(y_val_stack, y_train)
    y_today_iso = ir.transform(y_today_stack)
    y_val_blend = 0.5 * y_val_beta + 0.5 * y_val_iso
    y_today_blend = 0.5 * y_today_beta + 0.5 * y_today_iso

    # ========== SHAP EXPORT ==========
    show_shap = st.checkbox("Show SHAP Feature Importance (slow, only for small datasets)", value=False)
    if show_shap:
        with st.spinner("Computing SHAP values for XGB (validation set)..."):
            explainer = shap.TreeExplainer(get_base_models()[0][1])
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, show=False)
            st.pyplot(bbox_inches='tight')
            plt.clf()
            joblib.dump(shap_values, "shap_values.joblib")

    # ========== OOS DRIFT LOGGING ==========
    oos_auc = roc_auc_score(y_oos, y_today_blend[:len(y_oos)])
    oos_logloss = log_loss(y_oos, y_today_blend[:len(y_oos)])
    with open("oos_drift_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},OOS_AUC={oos_auc},OOS_LogLoss={oos_logloss}\n")

    st.success(f"OOS AUC: {oos_auc:.4f} | OOS LogLoss: {oos_logloss:.4f}")

    # ========== DRIFT CHECK ==========
    drifted = drift_check(X, X_today, n=6)
    if drifted:
        st.markdown("#### ⚡ **Feature Drift Diagnostics**")
        st.write("These features have unusual mean/std changes between training and today, check if input context shifted:", drifted)

    # ========== OVERLAY MULTIPLIER & LEADERBOARD ==========
    today_df = today_df.copy()
    ratings_df = today_df.apply(rate_weather, axis=1)
    for col in ratings_df.columns:
        today_df[col] = ratings_df[col]
    hr_probs = y_today_blend
    today_df['hr_probability'] = hr_probs
    if any([k in today_df.columns for k in ["wind_mph", "temp", "humidity", "park_hr_rate"]]):
        today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
        today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)
        sort_col = "final_hr_probability"
    else:
        today_df['final_hr_probability'] = today_df['hr_probability']
        sort_col = "final_hr_probability"
    leaderboard_cols = [c for c in ["player_name", "team_code", "time", "hr_probability", "final_hr_probability", "temp", "temp_rating", "humidity", "humidity_rating", "wind_mph", "wind_rating", "wind_dir_string", "condition", "condition_rating", "overlay_multiplier"] if c in today_df.columns]
    leaderboard = today_df[leaderboard_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
    leaderboard['hr_probability'] = leaderboard['hr_probability'].round(4)
    leaderboard['final_hr_probability'] = leaderboard['final_hr_probability'].round(4)
    if 'overlay_multiplier' in leaderboard.columns:
        leaderboard['overlay_multiplier'] = leaderboard['overlay_multiplier'].round(3)
    st.dataframe(leaderboard.head(30))

    # Prediction distribution plot
    st.subheader(f"📊 HR Probability Distribution (Top 30, Blend)")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(leaderboard["player_name"].astype(str)[:30], leaderboard[sort_col][:30], color='royalblue')
    ax.invert_yaxis()
    ax.set_xlabel('Predicted HR Probability')
    ax.set_ylabel('Player')
    st.pyplot(fig)

    st.subheader(f"Prediction Probability Distribution (all predictions, Blend)")
    plt.figure(figsize=(8, 3))
    plt.hist(leaderboard[sort_col], bins=30, color='orange', alpha=0.7)
    plt.xlabel("Final HR Probability")
    st.pyplot(plt.gcf())
