import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

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

def weather_rating(multiplier):
    if multiplier < 0.98:
        return "Poor"
    elif multiplier < 1.03:
        return "Average"
    elif multiplier < 1.09:
        return "Good"
    else:
        return "Excellent"

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

def combine_weather_labels(row):
    parts = [
        row.get('wind_speed_rating', ""),
        row.get('humidity_rating', ""),
        row.get('temperature_rating', ""),
        row.get('park_rating', ""),
        row.get('wind_dir_rating', "")
    ]
    return ", ".join([p for p in parts if p])

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
        st.write(f"Event: {event_df.shape}, Today: {today_df.shape}")
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
    st.dataframe(event_df[target_col].value_counts(dropna=False).reset_index(), use_container_width=True)

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features (before selection): {len(feature_cols)}")
    if len(feature_cols) > 200:
        feature_cols = feature_cols[:200]
        st.warning("Limiting features to first 200 for stability.")

    context_cols = ["player_name", "pitcher_team_code", "park"]
    context_cols_present = [c for c in context_cols if c in today_df.columns]
    if len(context_cols_present) < len(context_cols):
        missing = list(set(context_cols) - set(context_cols_present))
        st.warning(f"Some expected context columns are missing from today's file: {missing}")
    if not context_cols_present:
        st.error("No context columns available in today's file. Please check your input!")
        st.stop()
    orig_context = today_df[context_cols_present].copy()

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col].astype(int)
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    st.write("Splitting for validation and scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train = np.asarray(y_train, dtype=np.int32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.int32).reshape(-1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # --- Feature importance selection using XGB + Permutation (top 40->30) ---
    st.write("Running XGBoost for model-based feature importances...")
    xgb_fs = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    xgb_fs.fit(X, y)
    feature_importances = pd.Series(xgb_fs.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(40).index.tolist()
    st.write(f"Top 40 features by XGB importance: {top_features}")

    # Permutation importance
    st.write("Calculating permutation importances (validation set)...")
    perm_imp = permutation_importance(xgb_fs, X_val_scaled, y_val, n_repeats=5, random_state=42)
    pi_scores = pd.Series(perm_imp.importances_mean, index=top_features)
    pi_features = pi_scores.sort_values(ascending=False).head(30).index.tolist()
    st.write(f"Top 30 features by permutation importance: {pi_features}")

    # Final features: intersection
    final_features = [f for f in pi_features if f in top_features]
    st.write(f"Final features used in model: {final_features}")

    # Reduce feature set for modeling
    X_train_final = pd.DataFrame(X_train, columns=X.columns)[final_features].copy()
    X_val_final = pd.DataFrame(X_val, columns=X.columns)[final_features].copy()
    X_today_final = pd.DataFrame(X_today, columns=X.columns)[final_features].copy()

    # Meta-feature creation (pairwise for top 10)
    from itertools import combinations
    interaction_features = []
    top10 = final_features[:10]
    for f1, f2 in combinations(top10, 2):
        for dfset, df in zip(['train','val','today'], [X_train_final, X_val_final, X_today_final]):
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
            df[f'{f1}_div_{f2}'] = np.where(df[f2]!=0, df[f1]/df[f2], 0)
        interaction_features.extend([f'{f1}_x_{f2}', f'{f1}_div_{f2}'])
    st.write(f"Added {len(interaction_features)} meta-feature interactions.")

    # Robust ensemble
    st.write("Training ensemble (XGB, LGBM, CatBoost, RF, GB, LR)...")
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss', n_jobs=1, verbosity=0)
    lgb_clf = lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)
    cat_clf = cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)
    rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)
    gb_clf = GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)
    lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)
    models_for_ensemble = [
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('cat', cat_clf),
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('lr', lr_clf),
    ]
    for name, model in models_for_ensemble:
        try:
            model.fit(X_train_final, y_train)
            st.write(f"{name} fit success.")
        except Exception as e:
            st.warning(f"{name} training failed: {e}")
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_final, y_train)

    # Calibrate
    st.write("Calibrating probabilities (isotonic regression)...")
    y_val_pred = ensemble.predict_proba(X_val_final)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    y_today_pred = ensemble.predict_proba(X_today_final)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # Add weather & streaks
    today_df['weather_multiplier'] = today_df.apply(compute_weather_multiplier, axis=1)
    today_df['weather_rating'] = today_df['weather_multiplier'].apply(weather_rating)
    today_df = add_streak_features(event_df, today_df)
    today_df['streak_label'] = today_df.apply(assign_streak_label, axis=1)

    # Leaderboard
    today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    top_n = 30
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "weather_multiplier", "weather_rating", "streak_label"]
    leaderboard = today_df[leaderboard_cols].copy()
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["weather_multiplier"] = leaderboard["weather_multiplier"].round(3)
    leaderboard_top = leaderboard.head(top_n)

    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (AI, Weather & Streak Context)**")
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )
    st.write(f"Number of features used for modeling: {len(final_features) + len(interaction_features)}")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
