import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from scipy.stats import zscore
from itertools import combinations

st.set_page_config("🏆 MLB Home Run Predictor — Supercharged Edition", layout="wide")
st.title("🏆 MLB Home Run Predictor — State of the Art, Supercharged, Debug-First")

# =========== BASE UTILITY FUNCTIONS ===========
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
        dstd = np.nanstd(today[c])
        if tstd > 0 and abs(tmean - dmean) / tstd > n:
            drifted.append(c)
    return drifted

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)  # <-- fixes masked int bug
    for col in X.columns:
        lower = X[col].quantile(limits[0])
        upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def stickiness_rank_boost(df, top_k=10, stickiness_boost=0.18, prev_rank_col=None, hr_col='hr_probability'):
    """Apply a sticky rank meta-boost so HRs stick to the top N spots more often."""
    # Use prior leaderboard rank or previous model (if available) to add persistence
    stick = df[hr_col].copy()
    if prev_rank_col and prev_rank_col in df.columns:
        # Boost those who ranked high before
        prev_rank = df[prev_rank_col].rank(method='min', ascending=False)
        stick = stick + stickiness_boost * (prev_rank <= top_k)
    else:
        # If no prior info, just boost the current top K
        stick.iloc[:top_k] += stickiness_boost
    return stick

# ---- APP START ----
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

    # --- Drop columns with >30% NaNs for safety
    nan_thresh = 0.3
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > nan_thresh].index.tolist()
    if drop_cols:
        st.warning(f"Dropping {len(drop_cols)} features with >30% NaNs: {drop_cols[:20]}")
        X = X.drop(columns=drop_cols)
        X_today = X_today.drop(columns=drop_cols, errors='ignore')

    # --- Drop near-zero variance features
    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        st.warning(f"Dropping {len(nzv_cols)} near-constant features.")
        X = X.drop(columns=nzv_cols)
        X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    # --- Remove perfectly correlated features (keep only one in each group)
    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        st.warning(f"Dropping {len(to_drop)} highly correlated features.")
        X = X.drop(columns=to_drop)
        X_today = X_today.drop(columns=to_drop, errors='ignore')

    # --- Winsorize/Clip
    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)
     # ===> LIMIT TO 200 FEATURES BY VARIANCE (Add this section!) <===
    max_feats = 200
    variances = X.var().sort_values(ascending=False)
    top_feat_names = variances.head(max_feats).index.tolist()
    X = X[top_feat_names]
    X_today = X_today[top_feat_names]
    st.success(f"Final number of features after auto-filtering: {X.shape[1]}")

    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # =========== Split & Scale ===========
    y = event_df[target_col].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # ====== Drift Detection ======
    drifted = drift_check(X_train, X_today, n=6)
    if drifted:
        st.warning(f"⚠️ Drift detected in these features: {drifted[:12]} (distribution changed vs training!)")

    # =========== ML Training ===========
    st.write("Training state-of-the-art ensemble (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(n_estimators=120, max_depth=8, learning_rate=0.06, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    lgb_clf = lgb.LGBMClassifier(n_estimators=120, max_depth=8, learning_rate=0.06, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=120, depth=8, learning_rate=0.06, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=8, learning_rate=0.07)
    lr_clf = LogisticRegression(max_iter=800, solver='lbfgs', n_jobs=1)
    models_for_ensemble = [
        ('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf), ('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)
    ]
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    for name, model in models_for_ensemble:
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            st.warning(f"{name} training failed: {e}")
    ensemble.fit(X_train_scaled, y_train)

    # =========== Probability Calibration ===========
    st.write("Calibrating probabilities with isotonic regression...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # =========== STICKY, META-BOOSTED LEADERBOARD UPGRADES ===========
    st.write("Sticky meta-learning leaderboard upgrades (supercharging for Top 5/10/30 accuracy)...")
    # Save prior probability as base
    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    today_df['hr_base_rank'] = today_df['hr_probability'].rank(method='min', ascending=False)
    # Sticky boost: prefer HRs that have been ranked highly before
    today_df['sticky_hr_boost'] = stickiness_rank_boost(today_df, top_k=10, stickiness_boost=0.19, prev_rank_col=None, hr_col='hr_probability')
    # Additional: gap-based meta feature
    today_df['prob_gap_prev'] = today_df['hr_probability'].diff().fillna(0)
    today_df['prob_gap_next'] = today_df['hr_probability'].shift(-1) - today_df['hr_probability']
    today_df['prob_gap_next'] = today_df['prob_gap_next'].fillna(0)
    today_df['is_top_10_pred'] = (today_df.index < 10).astype(int)
    meta_pseudo_y = today_df['sticky_hr_boost'].copy()
    meta_pseudo_y.iloc[:10] = meta_pseudo_y.iloc[:10] + 0.18  # Extra stick for top 10
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
    meta_imp = pd.Series(meta_booster.feature_importances_, index=meta_features)
    st.dataframe(meta_imp.sort_values(ascending=False).to_frame("importance"))

    # Quick check for any NaNs or impossible values in leaderboard
    if leaderboard_top.isna().any().any():
        st.warning("⚠️ NaNs detected in leaderboard! Double-check the input features.")
    if (leaderboard_top[sort_col] < 0).any() or (leaderboard_top[sort_col] > 1).any():
        st.warning("⚠️ Some final probabilities are out of [0,1] range!")

    # Display drifted features if any
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

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
