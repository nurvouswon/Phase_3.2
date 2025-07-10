import streamlit as st
import pandas as pd
import numpy as np
import gc
import time
from datetime import timedelta
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import shap

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

def smooth_labels(y, smoothing=0.02):
    y = np.asarray(y)
    y_smooth = y.copy().astype(float)
    y_smooth[y == 1] = 1 - smoothing
    y_smooth[y == 0] = smoothing
    return y_smooth

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

    # Outlier removal (train only, before smoothing!)
    y = event_df[target_col].astype(int)
    X, y = remove_outliers(X, y, method="iforest", contamination=0.012)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)
    st.write(f"Rows after outlier removal: {X.shape[0]}")

    # ========== OOS TEST =============
    OOS_ROWS = 2000
    X_train, X_oos = X.iloc[:-OOS_ROWS].copy(), X.iloc[-OOS_ROWS:].copy()
    y_train, y_oos = y.iloc[:-OOS_ROWS].copy(), y.iloc[-OOS_ROWS:].copy()
    st.write(f"🔒 Automatically reserving last {OOS_ROWS} rows for Out-of-Sample (OOS) test. Using first 15000 for training.")

    # ===== Sampling for Streamlit Cloud =====
    max_rows = 15000
    if X_train.shape[0] > max_rows:
        st.warning(f"Training limited to {max_rows} rows for memory (full dataset was {X_train.shape[0]} rows).")
        X_train = X_train.iloc[:max_rows].copy()
        y_train = y_train.iloc[:max_rows].copy()

    # ---- KFold Setup ----
    n_splits = 3
    n_repeats = 2
    st.write(f"Preparing KFold splits: X {X_train.shape}, y {y_train.shape}, X_today {X_today.shape}")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    val_fold_probas = np.zeros((len(y_train), 6))
    test_fold_probas = np.zeros((X_today.shape[0], 6))
    scaler = StandardScaler()
    fold_times = []
    show_shap = st.checkbox("Show SHAP Feature Importance (slow, only for small datasets)", value=False)

    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train)):
        t_fold_start = time.time()
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        sc = scaler.fit(X_tr)
        X_tr_scaled = sc.transform(X_tr)
        X_va_scaled = sc.transform(X_va)
        X_today_scaled = sc.transform(X_today)

        # Tree models
        xgb_clf = xgb.XGBClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
        lgb_clf = lgb.LGBMClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, n_jobs=1)
        cat_clf = cb.CatBoostClassifier(iterations=90, depth=7, learning_rate=0.08, verbose=0, thread_count=1)
        gb_clf = GradientBoostingClassifier(n_estimators=80, max_depth=7, learning_rate=0.08)
        rf_clf = RandomForestClassifier(n_estimators=80, max_depth=8, n_jobs=1)
        lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)

        # Fit all with hard labels (classification)
        xgb_clf.fit(X_tr_scaled, y_tr)
        lgb_clf.fit(X_tr_scaled, y_tr)
        cat_clf.fit(X_tr_scaled, y_tr)
        gb_clf.fit(X_tr_scaled, y_tr)
        rf_clf.fit(X_tr_scaled, y_tr)
        lr_clf.fit(X_tr_scaled, y_tr)

        val_fold_probas[va_idx, 0] = xgb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 1] = lgb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 2] = cat_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 3] = gb_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 4] = rf_clf.predict_proba(X_va_scaled)[:, 1]
        val_fold_probas[va_idx, 5] = lr_clf.predict_proba(X_va_scaled)[:, 1]

        test_fold_probas[:, 0] += xgb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 1] += lgb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 2] += cat_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 3] += gb_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 4] += rf_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)
        test_fold_probas[:, 5] += lr_clf.predict_proba(X_today_scaled)[:, 1] / (n_splits * n_repeats)

        # Optional: SHAP for XGB (first fold only)
        if fold == 0 and show_shap:
            with st.spinner("Computing SHAP values (this can be slow)..."):
                explainer = shap.TreeExplainer(xgb_clf)
                shap_values = explainer.shap_values(X_va_scaled)
                st.write("Top SHAP Features (XGB, validation set):")
                shap.summary_plot(shap_values, pd.DataFrame(X_va_scaled, columns=X_tr.columns), show=False)
                st.pyplot(bbox_inches='tight')
                plt.clf()

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * ((n_splits * n_repeats) - (fold + 1))
        st.write(f"Fold {fold + 1} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    # Blend all model outputs (simple mean for robust ensemble)
    y_val_bag = val_fold_probas.mean(axis=1)
    y_today_bag = test_fold_probas.mean(axis=1)

    # ========== OOS TEST ==========
    with st.spinner("🔍 Running Out-Of-Sample (OOS) test on last 2,000 rows..."):
        scaler_oos = StandardScaler()
        X_oos_train_scaled = scaler_oos.fit_transform(X_train)
        X_oos_scaled = scaler_oos.transform(X_oos)
        oos_preds = []
        # Use all base models as above
        models = [
            xgb.XGBClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0),
            lgb.LGBMClassifier(n_estimators=90, max_depth=7, learning_rate=0.08, n_jobs=1),
            cb.CatBoostClassifier(iterations=90, depth=7, learning_rate=0.08, verbose=0, thread_count=1),
            GradientBoostingClassifier(n_estimators=80, max_depth=7, learning_rate=0.08),
            RandomForestClassifier(n_estimators=80, max_depth=8, n_jobs=1),
            LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)
        ]
        for model in models:
            model.fit(X_oos_train_scaled, y_train)
            oos_preds.append(model.predict_proba(X_oos_scaled)[:, 1])
        oos_probs = np.mean(np.column_stack(oos_preds), axis=1)
        oos_auc = roc_auc_score(y_oos, oos_probs)
        oos_logloss = log_loss(y_oos, oos_probs)
        st.success(f"OOS AUC: {oos_auc:.4f} | OOS LogLoss: {oos_logloss:.4f}")

    # ===== Calibration =====
    st.write("Calibrating probabilities (BetaCalibration & Isotonic)...")
    bc = BetaCalibration(parameters="abm")
    bc.fit(y_val_bag.reshape(-1,1), y_train)
    y_val_beta = bc.predict(y_val_bag.reshape(-1,1))
    y_today_beta = bc.predict(y_today_bag.reshape(-1,1))
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_iso = ir.fit_transform(y_val_bag, y_train)
    y_today_iso = ir.transform(y_today_bag)

    # --- Leaderboard logic for BOTH calibrations ---
    def build_leaderboard(df, hr_probs, label="calibrated_hr_probability"):
        df = df.copy()
        df[label] = hr_probs
        df = df.sort_values(label, ascending=False).reset_index(drop=True)
        df['hr_base_rank'] = df[label].rank(method='min', ascending=False)
        df['sticky_hr_boost'] = stickiness_rank_boost(df, top_k=10, stickiness_boost=0.19, prev_rank_col=None, hr_col=label)
        df['prob_gap_prev'] = df[label].diff().fillna(0)
        df['prob_gap_next'] = df[label].shift(-1) - df[label]
        df['prob_gap_next'] = df['prob_gap_next'].fillna(0)
        df['is_top_10_pred'] = (df.index < 10).astype(int)
        meta_pseudo_y = df['sticky_hr_boost'].copy()
        meta_pseudo_y.iloc[:10] = meta_pseudo_y.iloc[:10] + 0.18
        meta_features = ['sticky_hr_boost', 'prob_gap_prev', 'prob_gap_next', 'is_top_10_pred']
        from sklearn.ensemble import GradientBoostingRegressor
        meta_booster = GradientBoostingRegressor(n_estimators=90, max_depth=3, learning_rate=0.13)
        X_meta = df[meta_features].values
        y_meta = meta_pseudo_y.values
        meta_booster.fit(X_meta, y_meta)
        df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
        df = df.sort_values("meta_hr_rank_score", ascending=False).reset_index(drop=True)
        if any([k in df.columns for k in ["wind_mph", "temp", "humidity", "park_hr_rate"]]):
            df['overlay_multiplier'] = df.apply(overlay_multiplier, axis=1)
            df['final_hr_probability'] = (df[label] * df['overlay_multiplier']).clip(0, 1)
            sort_col = "final_hr_probability"
        else:
            df['final_hr_probability'] = df[label]
            sort_col = "final_hr_probability"
        leaderboard_cols = []
        if "player_name" in df.columns:
            leaderboard_cols.append("player_name")
        leaderboard_cols += [label, "final_hr_probability", "meta_hr_rank_score"]
        if 'overlay_multiplier' in df.columns:
            leaderboard_cols.append('overlay_multiplier')
        leaderboard = df[leaderboard_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
        leaderboard[label] = leaderboard[label].round(4)
        leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
        leaderboard["meta_hr_rank_score"] = leaderboard["meta_hr_rank_score"].round(4)
        if 'overlay_multiplier' in leaderboard.columns:
            leaderboard['overlay_multiplier'] = leaderboard['overlay_multiplier'].round(3)
        return leaderboard, sort_col, meta_features, meta_booster

    leaderboard_beta, sort_col_beta, meta_features, meta_booster_beta = build_leaderboard(today_df, y_today_beta, "hr_probability_beta")
    leaderboard_iso, sort_col_iso, meta_features, meta_booster_iso = build_leaderboard(today_df, y_today_iso, "hr_probability_iso")

    # Download and display for BOTH leaderboards
    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (BetaCalibration)**")
    leaderboard_top_beta = leaderboard_beta.head(top_n)
    st.dataframe(leaderboard_top_beta, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard (BetaCalibration) CSV",
        data=leaderboard_top_beta.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard_beta.csv"
    )

    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Isotonic)**")
    leaderboard_top_iso = leaderboard_iso.head(top_n)
    st.dataframe(leaderboard_top_iso, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard (Isotonic) CSV",
        data=leaderboard_top_iso.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard_iso.csv"
    )

    # Download full predictions
    st.download_button(
        "⬇️ Download Full Prediction CSV (BetaCalibration)",
        data=leaderboard_beta.to_csv(index=False),
        file_name="today_hr_predictions_full_beta.csv"
    )
    st.download_button(
        "⬇️ Download Full Prediction CSV (Isotonic)",
        data=leaderboard_iso.to_csv(index=False),
        file_name="today_hr_predictions_full_iso.csv"
    )

    # Leaderboard plots
    for name, lb, sort_col in [("BetaCalibration", leaderboard_top_beta, sort_col_beta), ("Isotonic", leaderboard_top_iso, sort_col_iso)]:
        if "player_name" in lb.columns:
            st.subheader(f"📊 HR Probability Distribution (Top 30, {name})")
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(lb["player_name"].astype(str), lb[sort_col], color='royalblue')
            ax.invert_yaxis()
            ax.set_xlabel('Predicted HR Probability')
            ax.set_ylabel('Player')
            st.pyplot(fig)

        # Meta-Ranker Feature Importance
        st.subheader(f"🔎 Meta-Ranker Feature Importance ({name})")
        importances = pd.Series(meta_booster_beta.feature_importances_, index=meta_features) if name=="BetaCalibration" else pd.Series(meta_booster_iso.feature_importances_, index=meta_features)
        st.dataframe(importances.sort_values(ascending=False).to_frame("importance"))

        if lb.isna().any().any():
            st.warning("⚠️ NaNs detected in leaderboard! Double-check the input features.")
        if (lb[sort_col] < 0).any() or (lb[sort_col] > 1).any():
            st.warning("⚠️ Some final probabilities are out of [0,1] range!")

    # Drift diagnostics
    drifted = drift_check(X, X_today, n=6)
    if drifted:
        st.markdown("#### ⚡ **Feature Drift Diagnostics**")
        st.write("These features have unusual mean/std changes between training and today, check if input context shifted:", drifted)

    # Extra: Show prediction histogram (full today set, both calibrations)
    for name, lb, sort_col in [("BetaCalibration", leaderboard_beta, sort_col_beta), ("Isotonic", leaderboard_iso, sort_col_iso)]:
        st.subheader(f"Prediction Probability Distribution (all predictions, {name})")
        plt.figure(figsize=(8, 3))
        plt.hist(lb[sort_col], bins=30, color='orange', alpha=0.7)
        plt.xlabel("Final HR Probability")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.close()

    # Memory cleanup
    del X, X_today, y, val_fold_probas, test_fold_probas, y_train, y_oos, X_train, X_oos
    gc.collect()

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
