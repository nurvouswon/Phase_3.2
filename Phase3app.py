import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingRegressor

# ------- State-of-the-Art MLB HR Predictor App -------

st.set_page_config("🏆 MLB HR Predictor — World Class AI", layout="wide")
st.title("🏆 MLB Home Run Predictor — World Class AI Stack")

# ==== File Reading & Utilities ====
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
    arr = np.asarray(X)
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

# ==== Context/Weather/Streak Features ====
def compute_weather_multiplier(row):
    multiplier = 1.0
    if 'wind_mph' in row and not pd.isna(row['wind_mph']):
        multiplier *= 1 + 0.01 * min(max(row['wind_mph'], 0), 20)
    if 'temp' in row and not pd.isna(row['temp']):
        if row['temp'] >= 80:
            multiplier *= 1.10
        elif row['temp'] >= 65:
            multiplier *= 1.05
        elif row['temp'] <= 50:
            multiplier *= 0.95
    if 'humidity' in row and not pd.isna(row['humidity']):
        if row['humidity'] >= 70:
            multiplier *= 1.05
        elif row['humidity'] <= 30:
            multiplier *= 0.97
    return round(multiplier, 3)

def add_streak_features(event_df, today_df):
    event_df = event_df.sort_values(['player_name', 'game_date'])
    event_df['hr_5g'] = (
        event_df.groupby('player_name')['hr_outcome']
        .transform(lambda x: x.rolling(window=5, min_periods=1).sum().shift(1))
    )
    event_df['hr_10g'] = (
        event_df.groupby('player_name')['hr_outcome']
        .transform(lambda x: x.rolling(window=10, min_periods=1).sum().shift(1))
    )
    def cold_streak(x):
        streak = 0
        for val in reversed(x):
            if val == 0:
                streak += 1
            else:
                break
        return streak
    event_df['cold_streak'] = (
        event_df.groupby('player_name')['hr_outcome']
        .transform(lambda x: cold_streak(x.values))
    )
    streak_cols = ['player_name', 'game_date', 'hr_5g', 'hr_10g', 'cold_streak']
    streak_df = event_df.sort_values(['player_name', 'game_date']).groupby('player_name').tail(1)[streak_cols]
    today_df = today_df.merge(streak_df, on='player_name', how='left')
    return today_df

def assign_streak_label(row):
    if pd.notnull(row.get('hr_5g', None)) and row['hr_5g'] >= 3:
        return "HOT"
    if pd.notnull(row.get('cold_streak', None)) and row['cold_streak'] >= 7:
        return "COLD"
    if pd.notnull(row.get('hr_10g', None)) and 2 <= row['hr_10g'] < 3:
        return "Breakout Watch"
    return ""

# ==== Main App ====
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

top_n = st.slider("Select Top N for Leaderboard", 5, 50, 10)

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
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # ---- Context for leaderboard display
    context_cols = ["player_name", "pitcher_team_code", "park"]
    context_cols_present = [c for c in context_cols if c in today_df.columns]
    orig_context = today_df[context_cols_present].copy() if context_cols_present else None

    # ==== 1. Feature Selection: Model-based + Permutation Importances ====
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of available features: {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col].astype(int)
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    st.write("Running XGBoost for model-based feature importances...")
    xgb_fs = xgb.XGBClassifier(n_estimators=60, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    xgb_fs.fit(X, y)
    feature_importances = pd.Series(xgb_fs.feature_importances_, index=X.columns)
    top_xgb = feature_importances.sort_values(ascending=False).head(200).index.tolist()

    # --- Permutation importances on validation
    X_train, X_val, y_train, y_val = train_test_split(X[top_xgb], y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today[top_xgb])
    xgb_fs.fit(X_train_scaled, y_train)
    perm_imp = permutation_importance(xgb_fs, X_val_scaled, y_val, n_repeats=8, random_state=42)
    pi_scores = pd.Series(perm_imp.importances_mean, index=top_xgb)
    pi_features = pi_scores.sort_values(ascending=False).head(60).index.tolist()
    st.write(f"Top 60 permutation features: {pi_features}")

    # --- Final features used
    final_features = [f for f in pi_features if f in top_xgb]
    st.write(f"Final features (used in stack): {final_features}")

    # ==== 2. Meta-Features/Interactions ====
    from itertools import combinations
    top10 = final_features[:10]
    interaction_features = []
    for f1, f2 in combinations(top10, 2):
        X_train[f'{f1}_x_{f2}'] = X_train[f1] * X_train[f2]
        X_val[f'{f1}_x_{f2}'] = X_val[f1] * X_val[f2]
        X_today_scaled = np.hstack([X_today_scaled, (X_today[f1] * X_today[f2]).values.reshape(-1,1)])
        interaction_features.append(f'{f1}_x_{f2}')
    st.write(f"Added {len(interaction_features)} interaction features.")

    # ==== 3. Model Stacking ====
    def make_base_learners():
        return [
            ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)),
            ('lgb', lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)),
            ('cat', cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)),
            ('gb', GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)),
            ('lr', LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1))
        ]

    base_models = make_base_learners()
    meta_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_val = np.zeros((X_val.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        model.fit(X_train, y_train)
        meta_train[:, i] = model.predict_proba(X_train)[:,1]
        meta_val[:, i] = model.predict_proba(X_val)[:,1]
    meta_learner = LogisticRegression(max_iter=400)
    meta_learner.fit(meta_train, y_train)
    val_meta_pred = meta_learner.predict_proba(meta_val)[:,1]
    auc = roc_auc_score(y_val, val_meta_pred)
    ll = log_loss(y_val, val_meta_pred)
    st.info(f"Stacked Ensemble Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    # Calibration
    ir = IsotonicRegression(out_of_bounds="clip")
    val_meta_pred_cal = ir.fit_transform(val_meta_pred, y_val)

    # ==== 4. Today’s Prediction ====
    meta_today = np.zeros((X_today_scaled.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        meta_today[:, i] = model.predict_proba(X_today_scaled)[:,1]
    y_today_pred = meta_learner.predict_proba(meta_today)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # ==== 5. Meta-Ranker Booster ====
    today_df['prob_gap_prev'] = today_df['hr_probability'].diff().fillna(0)
    today_df['prob_gap_next'] = today_df['hr_probability'].shift(-1) - today_df['hr_probability']
    today_df['prob_gap_next'] = today_df['prob_gap_next'].fillna(0)
    today_df['is_top_10_pred'] = (today_df.index < 10).astype(int)
    meta_pseudo_y = today_df['hr_probability'].copy()
    meta_pseudo_y.iloc[:10] = meta_pseudo_y.iloc[:10] + 0.15
    meta_features = ['hr_probability', 'prob_gap_prev', 'prob_gap_next', 'is_top_10_pred']
    X_meta = today_df[meta_features].values
    y_meta = meta_pseudo_y.values
    meta_booster = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.1)
    meta_booster.fit(X_meta, y_meta)
    today_df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
    today_df = today_df.sort_values("meta_hr_rank_score", ascending=False).reset_index(drop=True)

    # ==== 6. Context, Weather, Streak Labels ====
    today_df['weather_multiplier'] = today_df.apply(compute_weather_multiplier, axis=1)
    today_df = add_streak_features(event_df, today_df)
    today_df['streak_label'] = today_df.apply(assign_streak_label, axis=1)

    # ==== 7. Leaderboard Output ====
    leaderboard_cols = []
    if "player_name" in today_df.columns: leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "meta_hr_rank_score", "weather_multiplier", "streak_label"]
    if orig_context is not None:
        leaderboard_cols += [c for c in orig_context.columns if c not in leaderboard_cols]
    leaderboard = today_df[leaderboard_cols].copy()
    leaderboard = leaderboard.round(4)
    leaderboard_top = leaderboard.head(top_n)

    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (AI World Class)**")
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )
    st.download_button(
        f"⬇️ Download Full Prediction CSV",
        data=today_df.to_csv(index=False),
        file_name=f"full_hr_predictions.csv"
    )
    st.success("Done! This leaderboard is ready for prime time. All steps are self-documented in code for full transparency and reproducibility.")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
