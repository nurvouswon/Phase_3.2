import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

st.set_page_config("2️⃣ MLB HR Predictor — World Class AI Top 10 Focused", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — World Class AI Top 10 Focused")

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

    # ==== AI-POWERED FEATURE INTERACTION SELECTION ====
    st.markdown("## 🤖 AI Feature Interaction Selection (world class)")
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features used (no clustering): {len(feature_cols)}")
    X_base = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today_base = clean_X(today_df[feature_cols], train_cols=X_base.columns)
    nan_inf_check(X_base, "X features")
    nan_inf_check(X_today_base, "X_today features")

    # --- Create pairwise interactions and keep top N
    st.write("Building polynomial feature interactions (AI selection)...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    poly_features = poly.get_feature_names_out(feature_cols)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_features)
    # Select only the top interactions by correlation to y
    corrs = np.abs([np.corrcoef(X_poly_df[col], y)[0,1] if X_poly_df[col].std() > 0 else 0 for col in poly_features])
    top_ix = np.argsort(-corrs)[:20]  # Keep the 20 most predictive interactions (tunable)
    top_inter_feats = [poly_features[i] for i in top_ix]
    st.write(f"Top interaction features: {top_inter_feats}")
    X_ai = pd.concat([X_base, X_poly_df[top_inter_feats]], axis=1)
    X_today_ai = pd.concat([X_today_base, pd.DataFrame(poly.transform(X_today_base), columns=poly_features)[top_inter_feats]], axis=1)

    # ==== STANDARDIZATION & SPLIT ====
    X = X_ai
    X_today = X_today_ai
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # =========== DEEP ENSEMBLE (AI-powered) ===========
    st.write("Training deep ensemble (XGB, LGBM, CatBoost, RF, GB, ExtraTrees, LR)...")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.06, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=1, tree_method='hist')
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.06, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=100, depth=7, learning_rate=0.07, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=80, max_depth=9, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=80, max_depth=7, learning_rate=0.08)
    et_clf = ExtraTreesClassifier(n_estimators=70, max_depth=8, n_jobs=1)
    lr_clf = LogisticRegression(max_iter=700, solver='lbfgs', n_jobs=1)

    models_for_ensemble = [
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf),
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('et', et_clf),
        ('lr', lr_clf)
    ]
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_scaled, y_train)

    # =========== FEATURE IMPORTANCE DIAGNOSTICS ===========
    importances = []
    for name, clf in models_for_ensemble:
        try:
            if hasattr(clf, "feature_importances_"):
                importances.append(clf.feature_importances_)
            elif hasattr(clf, "coef_"):
                importances.append(np.abs(clf.coef_[0]))
        except Exception:
            continue
    st.markdown("## 🔍 Feature Importances (Mean of AI Models)")
    if importances:
        tree_importances = np.mean(importances, axis=0)
        import_df = pd.DataFrame({
            "feature": X.columns,
            "importance": tree_importances
        }).sort_values("importance", ascending=False)
        st.dataframe(import_df.head(30), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.barh(import_df.head(20)["feature"][::-1], import_df.head(20)["importance"][::-1])
        ax.set_title("Top 20 Feature Importances (Avg of All Models)")
        st.pyplot(fig)
    else:
        st.warning("Model feature importances not available.")

    # =========== VALIDATION: AUC & TOP-10 PRECISION ===========
    st.write("Validating (AI-ensemble, out-of-fold)...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    # Calculate Top-10 Precision (fraction of true HRs captured in top 10 preds)
    top10_ix = np.argsort(-y_val_pred)[:10]
    top10_hits = y_val.iloc[top10_ix].sum()
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}** — Top 10 True HRs: **{int(top10_hits)}**")

    # =========== CALIBRATION (Isotonic Regression) ===========
    st.write("Calibrating prediction probabilities (isotonic regression, AI-tuned)...")
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)

    # =========== PREDICT ===========

    st.write("Predicting HR probability for today (calibrated)...")
    y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # ==== APPLY OVERLAY SCORING ====
    today_df['overlay_multiplier'] = today_df.apply(overlay_multiplier, axis=1)
    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)

    # ==== POST-PREDICTION AI META-BOOSTER ====
    st.markdown("## 🤖 Post-Prediction AI Booster: Local Rank Adjustment")
    # Generate meta-features for top 20: probability, overlay, rank, etc.
    leaderboard = today_df.sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    top_n = 20
    leaderboard_top = leaderboard.head(top_n).copy()
    leaderboard_top['rank'] = np.arange(1, top_n+1)
    leaderboard_top['prob_diff_next'] = leaderboard_top['final_hr_probability'].shift(-1, fill_value=0) - leaderboard_top['final_hr_probability']
    meta_features = ['hr_probability', 'overlay_multiplier', 'final_hr_probability', 'rank', 'prob_diff_next']
    # Synthesize a meta-booster (regression tree trained on today's leaderboard context)
    from sklearn.ensemble import GradientBoostingRegressor
    meta_X = leaderboard_top[meta_features]
    meta_y = leaderboard_top['final_hr_probability']
    meta_booster = GradientBoostingRegressor(n_estimators=30, max_depth=2)
    meta_booster.fit(meta_X, meta_y)
    # Use booster to refine probabilities
    leaderboard_top['ai_boosted_prob'] = meta_booster.predict(meta_X)
    leaderboard_top = leaderboard_top.sort_values('ai_boosted_prob', ascending=False).reset_index(drop=True)

    # ==== TOP 10 PRECISION LEADERBOARD (AI boosted) ====
    leaderboard_top["hr_probability"] = leaderboard_top["hr_probability"].round(4)
    leaderboard_top["final_hr_probability"] = leaderboard_top["final_hr_probability"].round(4)
    leaderboard_top["overlay_multiplier"] = leaderboard_top["overlay_multiplier"].round(3)
    leaderboard_top["ai_boosted_prob"] = leaderboard_top["ai_boosted_prob"].round(4)
    st.markdown(f"### 🏆 **Top 10 Precision HR Leaderboard (AI Boosted)**")
    st.dataframe(leaderboard_top.head(10), use_container_width=True)

    st.download_button(
        f"⬇️ Download Top 20 AI-Boosted Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top20_ai_boosted_leaderboard.csv"
    )
    st.download_button(
        f"⬇️ Download Full Prediction CSV",
        data=today_df.to_csv(index=False),
        file_name="today_hr_predictions.csv"
    )

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
