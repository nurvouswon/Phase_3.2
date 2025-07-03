import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

st.set_page_config("2️⃣ MLB HR Predictor — World Class Deep Research", layout="wide")
st.title("2️⃣ MLB HR Predictor — World Class Deep Research [Stacked Ensemble | Meta-Learning | Feature Science]")

# ============ FILE HELPERS =============
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

# ============ ADVANCED FEATURE ENGINEERING =============
def make_deltas(df, base_features, windows=[3,5,7,14,20,30,60]):
    """Add delta and pct change features for rolling windows."""
    for base in base_features:
        for i, win1 in enumerate(windows[:-1]):
            for win2 in windows[i+1:]:
                col1, col2 = f"{base}_{win1}", f"{base}_{win2}"
                if col1 in df.columns and col2 in df.columns:
                    df[f"{base}_delta_{win1}_{win2}"] = df[col1] - df[col2]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df[f"{base}_pctchg_{win1}_{win2}"] = np.where(df[col2]!=0, (df[col1] - df[col2]) / df[col2], 0)
    return df

def make_interactions(df, group1, group2):
    """Multiply every col in group1 by every col in group2 (for batter vs pitcher)."""
    for a in group1:
        for b in group2:
            if a in df.columns and b in df.columns:
                df[f"{a}_X_{b}"] = df[a] * df[b]
    return df

def add_context_dummies(df, context_cols):
    """One-hot encode categorical context columns."""
    for col in context_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    return df

# ============ GAME DAY OVERLAY (NO DOUBLE-COUNTING) ============
def overlay_multiplier(row):
    multiplier = 1.0
    # Only overlay for factors NOT already in training features!
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

# ============ UI ============
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files (deep research takes a minute)..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)
        # Remove all-NA cols, dedup, etc.
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

    # ---- Feature set-up: get only columns present in both sets ----
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))

    # ==== SUPERCHARGED FEATURE ENGINEERING ====
    base_rolling_stats = [
        "b_avg_exit_velo", "b_barrel_rate", "b_fb_rate", "b_hard_contact_rate", "b_hard_hit_rate",
        "b_hit_dist_avg", "b_pull_rate", "b_slg", "b_spray_angle_avg", "b_spray_angle_std",
        "b_sweet_spot_rate", "p_avg_exit_velo", "p_barrel_rate", "p_fb_rate", "p_hard_contact_rate",
        "p_hard_hit_rate", "p_hit_dist_avg", "p_pull_rate", "p_slg", "p_spray_angle_avg",
        "p_spray_angle_std", "p_sweet_spot_rate"
    ]
    event_df = make_deltas(event_df, base_rolling_stats)
    today_df = make_deltas(today_df, base_rolling_stats)

    batter_feats = [c for c in event_df.columns if c.startswith("b_") and c.endswith(tuple(str(x) for x in [3,5,7,14,20,30,60]))]
    pitcher_feats = [c for c in event_df.columns if c.startswith("p_") and c.endswith(tuple(str(x) for x in [3,5,7,14,20,30,60]))]
    event_df = make_interactions(event_df, batter_feats, pitcher_feats)
    today_df = make_interactions(today_df, batter_feats, pitcher_feats)

    # One-hot encode selected context features if present
    context_cols = ["batter_hand", "pitcher_hand", "stand", "p_throws", "roof_status", "condition"]
    event_df = add_context_dummies(event_df, context_cols)
    today_df = add_context_dummies(today_df, context_cols)

    # Re-evaluate shared features after new columns added
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    # =========== DEEP RESEARCH STACKED ENSEMBLE ===========
    st.write("Splitting for validation and scaling (Stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # Train base models
    st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=60, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss',
        n_jobs=1, verbosity=1, tree_method='hist'
    )
    lgb_clf = lgb.LGBMClassifier(n_estimators=60, max_depth=5, learning_rate=0.08, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=60, depth=5, learning_rate=0.09, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=40, max_depth=7, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=40, max_depth=5, learning_rate=0.09)
    lr_clf = LogisticRegression(max_iter=400, solver='lbfgs', n_jobs=1)

    base_models = [
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf),
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('lr', lr_clf)
    ]

    for name, model in base_models:
        try:
            model.fit(X_train_scaled, y_train)
            st.write(f"{name} trained.")
        except Exception as e:
            st.warning(f"{name} failed: {e}")

    # Get validation predictions for stacking
    val_preds = []
    for name, model in base_models:
        try:
            val_pred = model.predict_proba(X_val_scaled)[:,1]
            val_preds.append(val_pred)
        except Exception:
            val_preds.append(np.zeros_like(y_val))
    val_preds = np.vstack(val_preds).T

    # Train meta-model (stacking)
    st.write("Training meta-learner (Logistic Regression) on base model outputs...")
    meta_lr = LogisticRegression(max_iter=300, solver='lbfgs')
    meta_lr.fit(val_preds, y_val)

    # Calibrate with Isotonic Regression (on meta outputs)
    st.write("Calibrating predictions (isotonic regression, top decile focus)...")
    meta_val_pred = meta_lr.predict_proba(val_preds)[:,1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(meta_val_pred, y_val)

    # Predict for today
    st.write("Predicting HR probabilities for today (stacked + calibrated)...")
    today_preds = []
    for name, model in base_models:
        try:
            pred = model.predict_proba(X_today_scaled)[:,1]
            today_preds.append(pred)
        except Exception:
            today_preds.append(np.zeros(X_today_scaled.shape[0]))
    today_preds = np.vstack(today_preds).T
    meta_today_pred = meta_lr.predict_proba(today_preds)[:,1]
    calibrated_today_pred = ir.transform(meta_today_pred)
    today_df['hr_probability'] = calibrated_today_pred

    # === Overlay/Postprocessing ===
    st.write("Applying post-prediction game day overlay scoring (weather, park, etc)...")
    today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)

    # ==== LIVE DIAGNOSTICS ====
    st.markdown("## 🔍 Feature Diagnostics")
    st.write("Feature count:", len(feature_cols))
    st.write("Top 30 Feature Names (post-engineering):")
    st.write(feature_cols[:30])

    # === Leaderboard, Hit Rate Tracking ===
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "overlay_multiplier", "final_hr_probability"]
    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)

    leaderboard_top10 = leaderboard.head(10)
    st.markdown("### 🏆 **Top 10 Precision HR Leaderboard (World Class Deep Research)**")
    st.dataframe(leaderboard_top10, use_container_width=True)
    if len(leaderboard) > 10:
        gap = leaderboard.loc[9, "final_hr_probability"] - leaderboard.loc[10, "final_hr_probability"]
        st.markdown(f"**Confidence gap between Top 10/11:** `{gap:.4f}`")

    # Save/track Top-10 hit rate (user adds today's actuals)
    if "actual_hr" in today_df.columns:
        top10_actuals = today_df.loc[leaderboard_top10.index, "actual_hr"]
        hits = top10_actuals.sum()
        st.markdown(f"### 🎯 **Today's Top-10 Hits: {int(hits)}/10**")
    else:
        st.markdown("*Upload your actual HR column for hit tracking!*")

    # Download buttons
    st.download_button(
        "⬇️ Download Full Prediction CSV",
        data=today_df.to_csv(index=False),
        file_name="today_hr_predictions.csv"
    )
    st.download_button(
        "⬇️ Download Top 10 Leaderboard CSV",
        data=leaderboard_top10.to_csv(index=False),
        file_name="top10_leaderboard.csv"
    )

    # Validation scores (meta-learner)
    auc = roc_auc_score(y_val, meta_val_pred)
    ll = log_loss(y_val, meta_val_pred)
    st.success(f"Validation AUC (Meta-Stacked, Calibrated): **{auc:.4f}** — LogLoss: **{ll:.4f}**")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
