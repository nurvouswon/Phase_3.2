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
import matplotlib.pyplot as plt

st.set_page_config("🏆 MLB Home Run Predictor — AI World Class", layout="wide")
st.title("🏆 MLB Home Run Predictor — AI-Powered Best in the World")

# ========== Utility Functions ==========
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
    # NaN handling is below, do NOT fillna here
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
        infs = np.isinf(X_num.values).sum()
    else:
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
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

def rate_wind_speed(ws):
    if pd.isnull(ws): return ""
    if ws >= 15: return "Extreme Wind"
    if ws >= 10: return "Strong Wind"
    if ws >= 5:  return "Mild Wind"
    return "Calm"

def rate_humidity(hum):
    if pd.isnull(hum): return ""
    if hum >= 70: return "Very Humid"
    if hum >= 50: return "Humid"
    if hum <= 30: return "Dry"
    return "Moderate"

def rate_temperature(temp):
    if pd.isnull(temp): return ""
    if temp >= 90: return "Very Hot"
    if temp >= 75: return "Hot"
    if temp <= 50: return "Cold"
    return "Mild"

def rate_park(park):
    if pd.isnull(park): return ""
    s = str(park).upper()
    if "COORS" in s: return "HR Paradise"
    if "PETCO" in s: return "Pitcher's Park"
    if "DODGER" in s or "ORACLE" in s: return "Neutral Park"
    return "Neutral Park"

def rate_wind_dir(wd):
    if pd.isnull(wd): return ""
    wd_str = str(wd).lower()
    if "out" in wd_str: return "HR Wind"
    if "in" in wd_str:  return "HR Suppressing"
    if "left" in wd_str or "right" in wd_str: return "Crosswind"
    return "Neutral"

def combine_weather_labels(row):
    parts = [
        row.get('wind_speed_rating', ""),
        row.get('humidity_rating', ""),
        row.get('temperature_rating', ""),
        row.get('park_rating', ""),
        row.get('wind_dir_rating', "")
    ]
    return ", ".join([p for p in parts if p])

# ========== Streamlit App ==========
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
    st.dataframe(event_df[target_col].value_counts(dropna=False).reset_index(), use_container_width=True)

    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Feature count before filtering: {len(feature_cols)}")

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
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

    st.write(f"Initial number of numeric features in play: {X.shape[1]}")

    # === NaN Filtering Logic ===
    nan_ratio = X.isna().mean()
    bad_cols = nan_ratio[nan_ratio > 0.3].index
    if len(bad_cols) > 0:
        st.warning(f"Dropping {len(bad_cols)} features with >30% NaNs: {list(bad_cols)}")
        X = X.drop(columns=bad_cols)
        X_today = X_today.drop(columns=bad_cols, errors='ignore')

    st.write(f"NaNs in X after dropping bad columns: {X.isna().sum().sum()}")
    st.write(f"NaNs in X_today after dropping bad columns: {X_today.isna().sum().sum()}")

    # Fill remaining NaNs with -1
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)

    st.write(f"Final NaN count (should be 0): X={X.isna().sum().sum()}, X_today={X_today.isna().sum().sum()}")
    nan_inf_check(X, "X features")
    nan_inf_check(X_today, "X_today features")

    # ===== Limit features to 200 for speed/robustness =====
    if X.shape[1] > 200:
        st.info("Reducing to top 200 features by variance for speed/robustness.")
        var_rank = X.var().sort_values(ascending=False)
        keep_cols = var_rank.index[:200]
        X = X[keep_cols]
        X_today = X_today[keep_cols]

    st.success(f"Final number of features after filtering: {X.shape[1]}")

    # ========== Main ML Pipeline ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # Train models (robust)
    st.write("Training ensemble models...")
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
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    model_status = []
    for name, model in models_for_ensemble:
        try:
            model.fit(X_train_scaled, y_train)
            model_status.append(f"{name} OK")
        except Exception as e:
            st.warning(f"{name} training failed: {e}")
    ensemble.fit(X_train_scaled, y_train)
    st.info("Model status: " + ", ".join(model_status))

    # Calibrate
    st.write("Calibrating probabilities (isotonic regression)...")
    y_val_pred = ensemble.predict_proba(X_val_scaled)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
    y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
    y_today_pred_cal = ir.transform(y_today_pred)
    today_df['hr_probability'] = y_today_pred_cal

    # Add context features back
    today_df['weather_multiplier'] = today_df.apply(compute_weather_multiplier, axis=1)
    today_df['weather_rating'] = today_df['weather_multiplier'].apply(weather_rating)
    today_df = add_streak_features(event_df, today_df)
    today_df['streak_label'] = today_df.apply(assign_streak_label, axis=1)
    today_df['wind_speed_rating'] = today_df['wind_mph'].apply(rate_wind_speed) if 'wind_mph' in today_df else ""
    today_df['humidity_rating'] = today_df['humidity'].apply(rate_humidity) if 'humidity' in today_df else ""
    today_df['temperature_rating'] = today_df['temp'].apply(rate_temperature) if 'temp' in today_df else ""
    today_df['park_rating'] = today_df['park'].apply(rate_park) if 'park' in today_df else ""
    today_df['wind_dir_rating'] = today_df['wind_dir_string'].apply(rate_wind_dir) if 'wind_dir_string' in today_df else ""
    today_df['weather_labels'] = today_df.apply(combine_weather_labels, axis=1)

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
    meta_booster = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.1)
    meta_booster.fit(X_meta, y_meta)
    today_df['meta_hr_rank_score'] = meta_booster.predict(X_meta)
    today_df = today_df.sort_values("meta_hr_rank_score", ascending=False).reset_index(drop=True)
    today_df = today_df.merge(orig_context, on="player_name", how="left")

    desired_cols = [
        "player_name", "pitcher_team_code", "park",
        "hr_probability", "meta_hr_rank_score", "weather_multiplier", "weather_rating",
        "streak_label", "wind_speed_rating", "humidity_rating", "temperature_rating",
        "park_rating", "wind_dir_rating", "weather_labels"
    ]
    cols = [c for c in desired_cols if c in today_df.columns]
    leaderboard = today_df[cols].copy()

    # Round key columns for clean display
    for col, n in [("hr_probability", 4), ("meta_hr_rank_score", 4), ("weather_multiplier", 3)]:
        if col in leaderboard.columns:
            leaderboard[col] = leaderboard[col].round(n)

    st.write("Columns in leaderboard:", leaderboard.columns.tolist())  # For debug

    top_n = 30
    leaderboard_top = leaderboard.head(top_n)
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (AI, Weather & Streak Context)**")
    st.dataframe(leaderboard_top, use_container_width=True)
    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard.csv"
    )

    # Visual: HR Probability Distribution (Top 30)
    if "hr_probability" in leaderboard_top.columns:
        st.subheader("📊 HR Probability Distribution (Top 30)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(leaderboard_top["player_name"].astype(str), leaderboard_top["hr_probability"], color='dodgerblue')
        ax.invert_yaxis()
        ax.set_xlabel('HR Probability')
        ax.set_ylabel('Player')
        st.pyplot(fig)

    # Visual: Meta-Ranker Feature Importance
    if 'meta_booster' in locals() and 'meta_features' in locals():
        st.subheader("🔎 Meta-Ranker Feature Importance")
        meta_imp = pd.Series(meta_booster.feature_importances_, index=meta_features)
        st.dataframe(meta_imp.sort_values(ascending=False).to_frame("importance"))

    # Download full predictions file
    st.download_button(
        "⬇️ Download Full Predictions CSV",
        data=today_df.to_csv(index=False),
        file_name="today_full_predictions.csv"
    )

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
