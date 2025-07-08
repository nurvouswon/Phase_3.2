import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
import optuna

st.set_page_config("🏆 MLB HR Predictor — Top-5 Precision AI", layout="wide")
st.title("🏆 MLB Home Run Predictor — AI Top-5 Precision Booster (World Class)")

# === File Uploader ===
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training", type=['csv', 'parquet'])
today_file = st.file_uploader("Upload TODAY CSV for Prediction", type=['csv', 'parquet'])
actual_file = st.file_uploader("OPTIONAL: Upload Actual Outcomes CSV (for leaderboard validation)", type=['csv', 'parquet'])

# === Helper functions ===
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

def get_valid_feature_cols(df):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status'])
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
    arr = np.asarray(X)
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def add_meta_features(df):
    df = df.copy()
    df = df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    df['prob_gap_prev'] = df['hr_probability'].diff().fillna(0)
    df['prob_gap_next'] = df['hr_probability'].shift(-1) - df['hr_probability']
    df['prob_gap_next'] = df['prob_gap_next'].fillna(0)
    df['is_top_10_pred'] = (df.index < 10).astype(int)
    # (add more meta-features as you like)
    return df

def topk_precision(pred_df, actual_list, k=5, player_col="player_name"):
    pred_topk = set(pred_df.sort_values("final_hr_probability", ascending=False).head(k)[player_col].values)
    actual_set = set(actual_list)
    if len(actual_set) == 0: return 0
    return len(pred_topk & actual_set) / min(len(pred_topk), len(actual_set))

# === Main Logic ===
if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.dropna(axis=1, how='all').reset_index(drop=True)
        today_df = today_df.dropna(axis=1, how='all').reset_index(drop=True)
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)
        if find_duplicate_columns(event_df):
            st.error(f"Duplicate columns in event file")
            st.stop()
        if find_duplicate_columns(today_df):
            st.error(f"Duplicate columns in today file")
            st.stop()

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features in play: {len(feature_cols)}")

    # === Smart Feature Reduction: XGB + Permutation (to top 80 features) ===
    X = clean_X(event_df[feature_cols])
    y = event_df[target_col].astype(int)
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # Model-based feature ranking
    st.write("⚡ Selecting features for top-5 leaderboard accuracy...")
    xgb_fs = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    xgb_fs.fit(X, y)
    feature_importances = pd.Series(xgb_fs.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(80).index.tolist()

    # Permutation importance (pick intersection for most stable)
    X_train, X_val, y_train, y_val = train_test_split(X[top_features], y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today[top_features])

    perm_imp = permutation_importance(xgb_fs, X_val_scaled, y_val, n_repeats=7, random_state=42)
    pi_scores = pd.Series(perm_imp.importances_mean, index=top_features)
    pi_features = pi_scores.sort_values(ascending=False).head(50).index.tolist()

    # Final features: intersection
    final_features = [f for f in pi_features if f in top_features][:40]
    st.write(f"Final features in use: {final_features}")

    X_train_final = X_train[final_features]
    X_val_final = X_val[final_features]
    X_today_final = X_today[final_features]

    # === Ensemble Training ===
    st.write("🧠 Training high-precision ensemble for Top-5 ranking...")
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    lgb_clf = lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)
    lr_clf = LogisticRegression(max_iter=800, solver='lbfgs', n_jobs=1)
    models_for_ensemble = [
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf),
        ('rf', rf_clf),
        ('lr', lr_clf),
    ]
    for name, model in models_for_ensemble:
        try:
            model.fit(X_train_final, y_train)
        except Exception as e:
            st.warning(f"{name} training failed: {e}")
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_final, y_train)

    y_val_pred = ensemble.predict_proba(X_val_final)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    auc = roc_auc_score(y_val, y_val_pred)
    top5_acc = np.mean(np.isin(np.argsort(y_val_pred)[-5:], np.where(y_val == 1)[0]))
    st.info(f"Validation AUC: **{auc:.4f}** — Top-5 accuracy: **{top5_acc:.3f}**")

    y_today_pred = ensemble.predict_proba(X_today_final)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # === Meta-Ranker (for top 20) ===
    st.write("⚡ Meta-ranking top 20 for leaderboard optimization...")
    today_df = add_meta_features(today_df)
    from sklearn.ensemble import GradientBoostingRegressor
    meta_features = ['hr_probability', 'prob_gap_prev', 'prob_gap_next', 'is_top_10_pred']
    X_meta = today_df[meta_features].values
    meta_pseudo_y = today_df['hr_probability'].copy()
    meta_pseudo_y[:5] += 0.12   # Slight boost for top-5 (optional)
    meta_booster = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.08)
    meta_booster.fit(X_meta, meta_pseudo_y)
    today_df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
    today_df['final_hr_probability'] = today_df['meta_hr_rank_score']

    # === LEADERBOARD OUTPUT ===
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "final_hr_probability"]
    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    top_n = 10
    leaderboard_top = leaderboard.head(top_n)
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Precision Booster)**")
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )

    # === LEADERBOARD VALIDATION (if actual outcomes provided) ===
    if actual_file is not None:
        actual_df = safe_read(actual_file)
        if 'player_name' not in actual_df.columns:
            st.error("Actuals CSV must have a 'player_name' column for validation!")
        else:
            actual_players = set(actual_df['player_name'])
            precision_at_5 = topk_precision(today_df, actual_players, k=5)
            st.markdown(f"#### **Precision@5 for Today: `{precision_at_5:.2%}`**")
else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
