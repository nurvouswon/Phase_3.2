import streamlit as st
import pandas as pd
import numpy as np
import gc
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

st.set_page_config("🏆 MLB Home Run Predictor — State of the Art, Full Phase 1", layout="wide")
st.title("🏆 MLB Home Run Predictor — State of the Art, Full Phase 1")

@st.cache_data(show_spinner=False, max_entries=2)
def safe_read_cached(path):
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

def remove_outliers(X, y, method="iforest", contamination=0.012):
    if method == "iforest":
        mask = IsolationForest(contamination=contamination, random_state=42).fit_predict(X) == 1
    else:
        mask = LocalOutlierFactor(contamination=contamination).fit_predict(X) == 1
    return X[mask], y[mask]

# ---- APP START ----

event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)
        event_df = event_df.dropna(axis=1, how='all')
        today_df = today_df.dropna(axis=1, how='all')
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.reset_index(drop=True)
        today_df = today_df.reset_index(drop=True)
        if find_duplicate_columns(event_df):
            st.error(f"Duplicate columns in event file")
            st.stop()
        if find_duplicate_columns(today_df):
            st.error(f"Duplicate columns in today file")
            st.stop()
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)
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

    nan_thresh = 0.3
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > nan_thresh].index.tolist()
    if drop_cols:
        st.warning(f"Dropping {len(drop_cols)} features with >30% NaNs: {drop_cols[:20]}")
        X = X.drop(columns=drop_cols)
        X_today = X_today.drop(columns=drop_cols, errors='ignore')

    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        st.warning(f"Dropping {len(nzv_cols)} near-constant features.")
        X = X.drop(columns=nzv_cols)
        X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        st.warning(f"Dropping {len(to_drop)} highly correlated features.")
        X = X.drop(columns=to_drop)
        X_today = X_today.drop(columns=to_drop, errors='ignore')

    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)

    # ======= LIMIT TO 200 FEATURES BY VARIANCE =======
    max_feats = 200
    variances = X.var().sort_values(ascending=False)
    top_feat_names = variances.head(max_feats).index.tolist()
    X = X[top_feat_names]
    X_today = X_today[top_feat_names]
    st.success(f"Final number of features after auto-filtering: {X.shape[1]}")

    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # ===== PHASE 1: Feature Crosses & Outlier Removal (sync crosses!) =====
    X, cross_names = auto_feature_crosses(X, max_cross=24)
    X_today, _ = auto_feature_crosses(X_today, max_cross=24, template_cols=cross_names)
    st.write(f"Cross features created: {cross_names}")
    st.write(f"After cross sync: X cols {X.shape[1]}, X_today cols {X_today.shape[1]}")
    # Outlier removal (train only)
    y = event_df[target_col].astype(int)
    X, y = remove_outliers(X, y, method="iforest", contamination=0.012)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)
    st.write(f"Rows after outlier removal: {X.shape[0]}")
    st.write(f"X index: {X.index.min()}-{X.index.max()}, y index: {y.index.min()}-{y.index.max()}, X shape: {X.shape}, y shape: {y.shape}")

    # ===== Sampling for Streamlit Cloud =====
    if X.shape[0] > 20000:
        st.warning(f"Training limited to 20000 rows for memory (full dataset was {X.shape[0]} rows).")
        X = X.iloc[:20000].copy()
        y = y.iloc[:20000].copy()

    # ---- KFold Setup ----
    n_splits = 3
    n_repeats = 2
    st.write(f"Preparing KFold splits: X {X.shape}, y {y.shape}, X_today {X_today.shape}")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    val_preds = np.zeros((len(y), n_splits * n_repeats))
    test_preds = []
    scaler = StandardScaler()
    fold_times = []
    t_start = time.time()

    progress = st.progress(0, text="Starting model folds... (please wait, can take several minutes)")

    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X, y)):
        t_fold_start = time.time()
        progress.progress((fold + 1) / (n_splits * n_repeats), text=f"Training fold {fold+1}/{n_splits*n_repeats} ...")

        # Diagnostic
        if fold == 0:
            st.write(f"First fold train idx: {tr_idx[:5]}... val idx: {va_idx[:5]}...")
            st.write(f"Train shape: {X.iloc[tr_idx].shape}, Val shape: {X.iloc[va_idx].shape}")

        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Standard scaling
        sc = scaler.fit(X_tr)
        X_tr_scaled = sc.transform(X_tr)
        X_va_scaled = sc.transform(X_va)
        X_today_scaled = sc.transform(X_today)

        xgb_clf = xgb.XGBClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
        lgb_clf = lgb.LGBMClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, n_jobs=1)
        cat_clf = cb.CatBoostClassifier(iterations=90, depth=7, learning_rate=0.08, verbose=0, thread_count=1)
        rf_clf = RandomForestClassifier(n_estimators=80, max_depth=8, n_jobs=1)
        gb_clf = GradientBoostingClassifier(n_estimators=80, max_depth=7, learning_rate=0.08)
        lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)

        models_for_ensemble = [
            ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf), ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
        ]
        ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)

        for name, model in models_for_ensemble:
            try:
                model.fit(X_tr_scaled, y_tr)
            except Exception as e:
                st.warning(f"{name} training failed in fold {fold+1}: {e}")

        try:
            ensemble.fit(X_tr_scaled, y_tr)
            val_preds[va_idx, fold] = ensemble.predict_proba(X_va_scaled)[:, 1]
            test_preds.append(ensemble.predict_proba(X_today_scaled)[:, 1])
        except Exception as e:
            st.error(f"Ensemble failed in fold {fold+1}: {e}")
            break

        # Release memory aggressively
        del X_tr, X_va, X_tr_scaled, X_va_scaled
        gc.collect()

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * ((n_splits*n_repeats) - (fold+1))
        st.write(f"Fold {fold+1} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    progress.progress(1.0, text="All folds complete!")

    # Bagged predictions
    y_val_bag = val_preds.mean(axis=1)
    y_today_bag = np.mean(np.column_stack(test_preds), axis=1)

    # ===== Calibration =====
    st.write("Calibrating probabilities (BetaCalibration & Isotonic)...")
    bc = BetaCalibration(parameters="abm")
    bc.fit(y_val_bag.reshape(-1,1), y)
    y_val_beta = bc.predict(y_val_bag.reshape(-1,1))
    y_today_beta = bc.predict(y_today_bag.reshape(-1,1))
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_iso = ir.fit_transform(y_val_bag, y)
    y_today_iso = ir.transform(y_today_bag)

    # Use the best calibration method based on logloss
    val_logloss_beta = log_loss(y, y_val_beta)
    val_logloss_iso = log_loss(y, y_val_iso)
    if val_logloss_beta < val_logloss_iso:
        st.success(f"BetaCalibration used (logloss={val_logloss_beta:.4f})")
        hr_probs = y_today_beta
    else:
        st.success(f"Isotonic used (logloss={val_logloss_iso:.4f})")
        hr_probs = y_today_iso

    today_df['hr_probability'] = hr_probs

    # =========== STICKY, META-BOOSTED LEADERBOARD UPGRADES ===========
    st.write("Sticky meta-learning leaderboard upgrades (supercharging for Top 5/10/30 accuracy)...")
    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    today_df['hr_base_rank'] = today_df['hr_probability'].rank(method='min', ascending=False)
    today_df['sticky_hr_boost'] = stickiness_rank_boost(today_df, top_k=10, stickiness_boost=0.19, prev_rank_col=None, hr_col='hr_probability')
    today_df['prob_gap_prev'] = today_df['hr_probability'].diff().fillna(0)
    today_df['prob_gap_next'] = today_df['hr_probability'].shift(-1) - today_df['hr_probability']
    today_df['prob_gap_next'] = today_df['prob_gap_next'].fillna(0)
    today_df['is_top_10_pred'] = (today_df.index < 10).astype(int)
    meta_pseudo_y = today_df['sticky_hr_boost'].copy()
    meta_pseudo_y.iloc[:10] = meta_pseudo_y.iloc[:10] + 0.18
    meta_features = ['sticky_hr_boost', 'prob_gap_prev', 'prob_gap_next', 'is_top_10_pred']
    from sklearn.ensemble import GradientBoostingRegressor
    meta_booster = GradientBoostingRegressor(n_estimators=90, max_depth=3, learning_rate=0.13)
    X_meta = today_df[meta_features].values
    y_meta = meta_pseudo_y.values
    meta_booster.fit(X_meta, y_meta)
    today_df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
    today_df = today_df.sort_values("meta_hr_rank_score", ascending=False).reset_index(drop=True)

    # Overlay: contextual multipliers
    if any([k in today_df.columns for k in ["wind_mph", "temp", "humidity", "park_hr_rate"]]):
        today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
        today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)
        sort_col = "final_hr_probability"
    else:
        today_df['final_hr_probability'] = today_df['hr_probability']
        sort_col = "final_hr_probability"

    # =========== Leaderboard Output ===========
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "final_hr_probability", "meta_hr_rank_score"]
    if 'overlay_multiplier' in today_df.columns:
        leaderboard_cols.append('overlay_multiplier')
    leaderboard = today_df[leaderboard_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    leaderboard["meta_hr_rank_score"] = leaderboard["meta_hr_rank_score"].round(4)
    if 'overlay_multiplier' in leaderboard.columns:
        leaderboard['overlay_multiplier'] = leaderboard['overlay_multiplier'].round(3)

    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} Supercharged HR Leaderboard**")
    leaderboard_top = leaderboard.head(top_n)
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )

    # Download full predictions (with diagnostics)
    st.download_button(
        "⬇️ Download Full Prediction CSV",
        data=today_df.to_csv(index=False),
        file_name="today_hr_predictions_full.csv"
    )

    # HR Probability Distribution (Top 30)
    if "player_name" in leaderboard_top.columns:
        st.subheader("📊 HR Probability Distribution (Top 30)")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(leaderboard_top["player_name"].astype(str), leaderboard_top[sort_col], color='royalblue')
        ax.invert_yaxis()
        ax.set_xlabel('Predicted HR Probability')
        ax.set_ylabel('Player')
        st.pyplot(fig)

    # Meta-Ranker Feature Importance
    st.subheader("🔎 Meta-Ranker Feature Importance")
    importances = pd.Series(meta_booster.feature_importances_, index=meta_features)
    st.dataframe(importances.sort_values(ascending=False).to_frame("importance"))

    if leaderboard_top.isna().any().any():
        st.warning("⚠️ NaNs detected in leaderboard! Double-check the input features.")
    if (leaderboard_top[sort_col] < 0).any() or (leaderboard_top[sort_col] > 1).any():
        st.warning("⚠️ Some final probabilities are out of [0,1] range!")

    # Display drifted features if any
    drifted = drift_check(X, X_today, n=6)
    if drifted:
        st.markdown("#### ⚡ **Feature Drift Diagnostics**")
        st.write("These features have unusual mean/std changes between training and today, check if input context shifted:", drifted)

    # Extra: Show prediction histogram (full today set)
    st.subheader("Prediction Probability Distribution (all predictions)")
    plt.figure(figsize=(8, 3))
    plt.hist(today_df[sort_col], bins=30, color='orange', alpha=0.7)
    plt.xlabel("Final HR Probability")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.close()

    # Memory cleanup
    del X, X_today, y, val_preds, test_preds
    gc.collect()

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
