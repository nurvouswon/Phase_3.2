import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression

st.set_page_config("🏆 MLB HR Predictor — AI World Class", layout="wide")
st.title("🏆 MLB Home Run Predictor — AI-Powered Best in the World")

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

# ==== UI ====
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
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features in play: {len(feature_cols)}")

    # === CRASHPROOF: Numeric-only, de-duplication, feature cap ===
    X = event_df[feature_cols].copy()
    X_today = today_df[feature_cols].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    X_today = X_today.loc[:, ~X_today.columns.duplicated()]
    # Only numerics!
    num_cols = X.select_dtypes(include=[np.number]).columns
    X = X[num_cols]
    X_today = X_today[num_cols]
    st.write(f"Numeric features after cleanup: {X.shape[1]}")
    # Cap at 200 for speed/safety
    MAX_FEATS = 200
    if X.shape[1] > MAX_FEATS:
        st.warning(f"Capping to {MAX_FEATS} features for stability.")
        X = X.iloc[:, :MAX_FEATS]
        X_today = X_today[X.columns]

    y = event_df[target_col].astype(int)
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)
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

    st.write("Training ensemble models (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=1)
    lgb_clf = lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)
    lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)

    models_for_ensemble = []
    for clf, name in zip([xgb_clf, lgb_clf, cat_clf, rf_clf, gb_clf, lr_clf], ['XGB','LGB','CatBoost','RF','GB','LR']):
        try:
            clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append((name, clf))
        except Exception as e:
            st.warning(f"{name} failed: {e}")

    if not models_for_ensemble:
        st.error("No models could be trained. Try reducing features or data size.")
        st.stop()

    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_scaled, y_train)

    st.write("Calibrating probabilities with isotonic regression...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # Add weather and streak features
    today_df['weather_multiplier'] = today_df.apply(compute_weather_multiplier, axis=1)
    today_df = add_streak_features(event_df, today_df)
    today_df['streak_label'] = today_df.apply(assign_streak_label, axis=1)

    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "weather_multiplier", "streak_label"]

    leaderboard = today_df[leaderboard_cols].copy()
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["weather_multiplier"] = leaderboard["weather_multiplier"].round(3)

    top_n = 30
    leaderboard_top = leaderboard.head(top_n)
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard**")
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )
else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
