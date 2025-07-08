import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

st.set_page_config("🏆 MLB HR Predictor — AI World Class", layout="wide")
st.title("🏆 MLB Home Run Predictor — AI-Planet Crash-Proof Edition")

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
        # Force conversion to float64 for isinf
        infs = np.isinf(X_num.astype(np.float64).values).sum()
    else:
        arr = np.asarray(X, dtype=np.float64)
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

# --- Feature Filter: Variance, Correlation, Cap ---
def filter_features(X, X_today, max_feats=200, corr_threshold=0.95):
    # Variance threshold
    selector = VarianceThreshold(threshold=0.0001)
    selector.fit(X)
    low_var_cols = X.columns[~selector.get_support()]
    if len(low_var_cols) > 0:
        st.info(f"Removed {len(low_var_cols)} low-variance features.")
        X = X.loc[:, selector.get_support()]
        X_today = X_today.loc[:, selector.get_support()]

    # Correlation filter
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    if len(drop_corr) > 0:
        st.info(f"Removed {len(drop_corr)} highly correlated features.")
        X = X.drop(columns=drop_corr)
        X_today = X_today.drop(columns=drop_corr)

    # Cap features for ML selection
    if X.shape[1] > max_feats:
        st.info(f"Capped feature count to top {max_feats} (first {max_feats} features)")
        keep_cols = X.columns[:max_feats]
        X = X[keep_cols]
        X_today = X_today[keep_cols]
    return X, X_today

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

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")
    st.write("Feature count before filtering:", len(event_df.columns))

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Initial number of numeric features in play: {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col].astype(int)
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

    # ---- Crash-Proof Feature Filtering ----
    X, X_today = filter_features(X, X_today, max_feats=200, corr_threshold=0.95)
    st.success(f"Final number of features after filtering: {X.shape[1]}")

    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # --- XGBoost Feature Importance Selection ---
    st.write("⚡ Running XGBoost for feature importances...")
    xgb_fs = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0, tree_method='hist')
    xgb_fs.fit(X, y)
    feature_importances = pd.Series(xgb_fs.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(40).index.tolist()
    st.write(f"Top 40 features by XGB importance: {top_features}")

    # --- Permutation Importances ---
    st.write("⚡ Calculating permutation importances (validation set)...")
    X_train, X_val, y_train, y_val = train_test_split(X[top_features], y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today[top_features])
    xgb_fs.fit(X_train_scaled, y_train)
    perm_imp = permutation_importance(xgb_fs, X_val_scaled, y_val, n_repeats=7, random_state=42)
    pi_scores = pd.Series(perm_imp.importances_mean, index=top_features)
    pi_features = pi_scores.sort_values(ascending=False).head(30).index.tolist()
    st.write(f"Top 30 features by permutation importance: {pi_features}")

    # === Final features: intersection of XGB and Permutation top features ===
    final_features = [f for f in pi_features if f in top_features]
    st.write(f"Features used in final model: {final_features}")

    X_train_final = X_train[final_features]
    X_val_final = X_val[final_features]
    X_today_final = X_today[final_features]

    # --- Meta-Feature Interactions (top 10 only) ---
    st.write("⚡ Creating meta-feature interactions for top features...")
    from itertools import combinations
    interaction_features = []
    top10 = final_features[:10]
    for f1, f2 in combinations(top10, 2):
        X_train_final[f'{f1}_x_{f2}'] = X_train_final[f1] * X_train_final[f2]
        X_val_final[f'{f1}_x_{f2}'] = X_val_final[f1] * X_val_final[f2]
        X_today_final[f'{f1}_x_{f2}'] = X_today_final[f1] * X_today_final[f2]
        interaction_features.append(f'{f1}_x_{f2}')
    st.write(f"Added {len(interaction_features)} meta interaction features.")

    # --- Stacked Ensemble (Crash Proof) ---
    st.write("🧠 Training stacked ensemble (XGB, LGBM, CatBoost, RF, GB, LR)...")
    def make_base_learners():
        return [
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0, tree_method='hist')),
            ('lgb', lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)),
            ('cat', cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)),
            ('gb', GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)),
            ('lr', LogisticRegression(max_iter=900, solver='lbfgs', n_jobs=1))
        ]
    base_models = make_base_learners()
    meta_train = np.zeros((X_train_final.shape[0], len(base_models)))
    meta_val = np.zeros((X_val_final.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        model.fit(X_train_final, y_train)
        meta_train[:, i] = model.predict_proba(X_train_final)[:,1]
        meta_val[:, i] = model.predict_proba(X_val_final)[:,1]
    meta_learner = LogisticRegression(max_iter=600)
    meta_learner.fit(meta_train, y_train)
    val_meta_pred = meta_learner.predict_proba(meta_val)[:,1]
    auc = roc_auc_score(y_val, val_meta_pred)
    ll = log_loss(y_val, val_meta_pred)
    st.info(f"Stacked Ensemble Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # --- Calibration ---
    st.write("Calibrating meta-ensemble (isotonic regression)...")
    ir = IsotonicRegression(out_of_bounds="clip")
    val_meta_pred_cal = ir.fit_transform(val_meta_pred, y_val)

    # --- Predict for today ---
    meta_today = np.zeros((X_today_final.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        meta_today[:, i] = model.predict_proba(X_today_final)[:,1]
    y_today_pred = meta_learner.predict_proba(meta_today)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    today_df['prob_gap_prev'] = today_df['hr_probability'].diff().fillna(0)
    today_df['prob_gap_next'] = today_df['hr_probability'].shift(-1) - today_df['hr_probability']
    today_df['prob_gap_next'] = today_df['prob_gap_next'].fillna(0)
    today_df['is_top_10_pred'] = (today_df.index < 10).astype(int)
    meta_pseudo_y = today_df['hr_probability'].copy()
    meta_pseudo_y.iloc[:10] = meta_pseudo_y.iloc[:10] + 0.15
    meta_features = ['hr_probability', 'prob_gap_prev', 'prob_gap_next', 'is_top_10_pred']
    X_meta = today_df[meta_features].values
    y_meta = meta_pseudo_y.values
    from sklearn.ensemble import GradientBoostingRegressor
    meta_booster = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.1)
    meta_booster.fit(X_meta, y_meta)
    today_df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
    today_df = today_df.sort_values("meta_hr_rank_score", ascending=False).reset_index(drop=True)

    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "meta_hr_rank_score"]

    leaderboard = today_df[leaderboard_cols]
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["meta_hr_rank_score"] = leaderboard["meta_hr_rank_score"].round(4)
    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Crash-Proof, AI-Planet)**")
    leaderboard_top = leaderboard.head(top_n)
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )

    # HR Probability Distribution
    if "hr_probability" in leaderboard_top.columns:
        st.subheader("📊 HR Probability Distribution (Top 30)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(leaderboard_top["player_name"].astype(str), leaderboard_top["hr_probability"], color='dodgerblue')
        ax.invert_yaxis()
        ax.set_xlabel('HR Probability')
        ax.set_ylabel('Player')
        st.pyplot(fig)

    # Meta-Ranker Feature Importance
    if 'meta_booster' in locals() and 'meta_features' in locals():
        st.subheader("🔎 Meta-Ranker Feature Importance")
        meta_imp = pd.Series(meta_booster.feature_importances_, index=meta_features)
        st.dataframe(meta_imp.sort_values(ascending=False).to_frame("importance"))

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
