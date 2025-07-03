import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

st.set_page_config("2️⃣ MLB HR Predictor — Deep Ensemble + Weather, Power, Platoon Overlays", layout="wide")
st.title("2️⃣ MLB Home Run Predictor — Deep Ensemble + Weather, Power, Platoon Overlays [2025 DEEP RESEARCH]")
# ===================== CONTEXT MAPS & RATES =====================
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82,
    'wrigley_field': 1.12, 'great_american_ball_park': 1.26, 'american_family_field': 1.17,
    'pnc_park': 0.87, 'busch_stadium': 0.87, 'truist_park': 1.06, 'loan_depot_park': 0.86,
    'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05, 'petco_park': 0.85,
    'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12, 'target_field': 1.05
}
park_altitude_map = {
    'coors_field': 5280, 'chase_field': 1100, 'dodger_stadium': 338, 'minute_maid_park': 50,
    'fenway_park': 19, 'wrigley_field': 594, 'great_american_ball_park': 489, 'oracle_park': 10,
    'petco_park': 62, 'yankee_stadium': 55, 'citizens_bank_park': 30, 'kauffman_stadium': 750,
    'guaranteed_rate_field': 600, 'progressive_field': 650, 'busch_stadium': 466, 'camden_yards': 40,
    'rogers_centre': 250, 'angel_stadium': 160, 'tropicana_field': 3, 'citi_field': 3,
    'oakland_coliseum': 50, 'globe_life_field': 560, 'pnc_park': 725, 'loan_depot_park': 7,
    'loandepot_park': 7, 'nationals_park': 25, 'american_family_field': 633, 'sutter_health_park': 20,
    'target_field': 830
}
roof_status_map = {
    'rogers_centre': 'closed', 'chase_field': 'open', 'minute_maid_park': 'open',
    'loan_depot_park': 'closed', 'loandepot_park': 'closed', 'globe_life_field': 'open',
    'tropicana_field': 'closed', 'american_family_field': 'open'
}
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field',
    'BOS': 'fenway_park', 'NYY': 'yankee_stadium', 'CHC': 'wrigley_field',
    'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park', 'ATH': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park',
    'MIA': 'loandepot_park', 'TB': 'tropicana_field', 'MIL': 'american_family_field',
    'SD': 'petco_park', 'SF': 'oracle_park', 'TOR': 'rogers_centre', 'CLE': 'progressive_field',
    'MIN': 'target_field', 'KC': 'kauffman_stadium', 'CWS': 'guaranteed_rate_field',
    'CHW': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park',
    'TEX': 'globe_life_field', 'ARI': 'chase_field', 'AZ': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park',
    'STL': 'busch_stadium', 'BAL': 'camden_yards', 'WSH': 'nationals_park', 'WAS': 'nationals_park'
}
mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'AZ': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'CHW': 'Chicago', 'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim',
    'LAD': 'Los Angeles', 'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York',
    'NYY': 'New York', 'OAK': 'Oakland', 'ATH': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
    'SD': 'San Diego', 'SEA': 'Seattle', 'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg',
    'TEX': 'Arlington', 'TOR': 'Toronto', 'WSH': 'Washington', 'WAS': 'Washington'
}
park_hand_hr_rate_map = {
    'angels_stadium': {'L': 1.09, 'R': 1.02}, 'angel_stadium': {'L': 1.09, 'R': 1.02},
    'minute_maid_park': {'L': 1.13, 'R': 1.06}, 'coors_field': {'L': 1.38, 'R': 1.24},
    'yankee_stadium': {'L': 1.47, 'R': 0.98}, 'fenway_park': {'L': 1.04, 'R': 0.97},
    'rogers_centre': {'L': 1.08, 'R': 1.12}, 'tropicana_field': {'L': 0.84, 'R': 0.89},
    'camden_yards': {'L': 0.98, 'R': 1.27}, 'guaranteed_rate_field': {'L': 1.25, 'R': 1.11},
    'progressive_field': {'L': 0.99, 'R': 1.02}, 'comerica_park': {'L': 1.10, 'R': 0.91},
    'kauffman_stadium': {'L': 0.90, 'R': 1.03}, 'globe_life_field': {'L': 1.01, 'R': 0.98},
    'dodger_stadium': {'L': 1.02, 'R': 1.18}, 'oakland_coliseum': {'L': 0.81, 'R': 0.85},
    't-mobile_park': {'L': 0.81, 'R': 0.92}, 'tmobile_park': {'L': 0.81, 'R': 0.92},
    'oracle_park': {'L': 0.67, 'R': 0.99}, 'wrigley_field': {'L': 1.10, 'R': 1.16},
    'great_american_ball_park': {'L': 1.30, 'R': 1.23}, 'american_family_field': {'L': 1.25, 'R': 1.13},
    'pnc_park': {'L': 0.76, 'R': 0.92}, 'busch_stadium': {'L': 0.78, 'R': 0.91},
    'truist_park': {'L': 1.00, 'R': 1.09}, 'loan_depot_park': {'L': 0.83, 'R': 0.91},
    'loandepot_park': {'L': 0.83, 'R': 0.91}, 'citi_field': {'L': 1.11, 'R': 0.98},
    'nationals_park': {'L': 1.04, 'R': 1.06}, 'petco_park': {'L': 0.90, 'R': 0.88},
    'chase_field': {'L': 1.16, 'R': 1.05}, 'citizens_bank_park': {'L': 1.22, 'R': 1.20},
    'sutter_health_park': {'L': 1.12, 'R': 1.12}, 'target_field': {'L': 1.09, 'R': 1.01}
}
# ========== DEEP RESEARCH HR MULTIPLIERS: BATTER SIDE ===============
park_hr_percent_map_all = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CWS': 1.25,
    'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09,
    'WAS': 1.00, 'WSH': 1.00
}

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
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status', 'batting_order'}
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
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status','batting_order'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def drop_high_na_low_var(df, thresh_na=0.25, thresh_var=1e-7):
    cols_to_drop = []
    na_frac = df.isnull().mean()
    low_var_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).std() < thresh_var]
    for c in df.columns:
        if na_frac.get(c, 0) > thresh_na:
            cols_to_drop.append(c)
        elif c in low_var_cols:
            cols_to_drop.append(c)
    df2 = df.drop(columns=cols_to_drop, errors="ignore")
    return df2, cols_to_drop

def cluster_select_features(df, threshold=0.95):
    corr = df.corr().abs()
    clusters = []
    selected = []
    dropped = []
    visited = set()
    for col in corr.columns:
        if col in visited:
            continue
        cluster = [col]
        visited.add(col)
        for other in corr.columns:
            if other != col and other not in visited and corr.loc[col, other] >= threshold:
                cluster.append(other)
                visited.add(other)
        clusters.append(cluster)
        selected.append(cluster[0])
        dropped.extend(cluster[1:])
    return selected, clusters, dropped

def downcast_df(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int', 'int64', 'int32'])
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# ==== OVERLAY: DEEP RESEARCH SCORING LOGIC ====
def overlay_multiplier(row, detailed=False):
    """
    Ultra-robust overlay: weather, park, SZN power, hot streak, platoon, lineup, capping.
    Shows all notes if detailed=True.
    """
    multiplier = 1.0
    notes = []

    # Wind (game day only)
    wind = row.get("wind_mph", None)
    wind_dir = str(row.get("wind_dir_string", "")).lower()
    if pd.notnull(wind) and wind >= 10:
        if "out" in wind_dir:
            multiplier *= 1.08
            notes.append("Wind Out +8%")
        elif "in" in wind_dir:
            multiplier *= 0.93
            notes.append("Wind In -7%")

    # Temperature
    temp = row.get("temp", None)
    if pd.notnull(temp):
        delta = temp - 70
        temp_mult = 1.03 ** (delta / 10)
        multiplier *= temp_mult
        if abs(delta) >= 5:
            notes.append(f"Temp Adj {temp_mult:.2f}x")

    # Humidity
    hum = row.get("humidity", None)
    if pd.notnull(hum):
        if hum > 60:
            multiplier *= 1.02
            notes.append("High Humidity +2%")
        elif hum < 40:
            multiplier *= 0.98
            notes.append("Low Humidity -2%")

    # Park HR Factor
    pf = row.get("park_hr_rate", 1.0)
    if pd.notnull(pf):
        pf = max(0.85, min(1.20, float(pf)))
        multiplier *= pf
        if pf != 1.0:
            notes.append(f"Park {pf:.2f}x")

    # Season-to-date Power: use best available power column (e.g. barrel, HR, hard hit)
    power_cols = [c for c in row.index if ("barrel_rate_60" in c or "hr_rate_60" in c or "hard_hit_rate_60" in c)]
    power_boost = 1.0
    if power_cols:
        base_val = np.nanmean([row[c] for c in power_cols if pd.notnull(row[c])])
        if not np.isnan(base_val) and base_val > 0.12:  # above MLB average
            power_boost = 1.07 + 0.07 * (base_val - 0.12) / 0.08
            power_boost = min(power_boost, 1.18)
            multiplier *= power_boost
            notes.append(f"SZN Power {power_boost:.2f}x")
    # Momentum / hot streak (b_barrel_rate_7, b_hr_rate_7, b_hard_hit_rate_7, etc)
    moment_cols = [c for c in row.index if ("barrel_rate_7" in c or "hr_rate_7" in c or "hard_hit_rate_7" in c)]
    moment_boost = 1.0
    if moment_cols:
        hot_val = np.nanmean([row[c] for c in moment_cols if pd.notnull(row[c])])
        if not np.isnan(hot_val) and hot_val > 0.18:
            moment_boost = 1.08 + 0.10 * (hot_val - 0.18) / 0.08
            moment_boost = min(moment_boost, 1.22)
            multiplier *= moment_boost
            notes.append(f"Hot Streak {moment_boost:.2f}x")

    # Platoon advantage
    bat_hand = str(row.get("batter_hand", "")).upper()
    pit_hand = str(row.get("pitcher_hand", "")).upper()
    platoon_mult = 1.0
    if bat_hand in ("L", "R") and pit_hand in ("L", "R"):
        if (bat_hand == "L" and pit_hand == "R") or (bat_hand == "R" and pit_hand == "L"):
            platoon_mult = 1.08
            multiplier *= platoon_mult
            notes.append("Platoon Adv +8%")
        else:
            platoon_mult = 0.97
            multiplier *= platoon_mult
            notes.append("No Platoon -3%")

    # Lineup order effect (if available)
    order = row.get("batting_order", None)
    if pd.notnull(order) and isinstance(order, (int, float)):
        try:
            if order <= 3:
                order_mult = 1.10
                multiplier *= order_mult
                notes.append("Top Order +10%")
            elif order <= 6:
                order_mult = 1.04
                multiplier *= order_mult
                notes.append("Middle Order +4%")
            elif order >= 8:
                order_mult = 0.93
                multiplier *= order_mult
                notes.append("Bottom Order -7%")
        except Exception:
            pass

    # Capping/flooring for sanity
    overlay_cap = 1.22
    overlay_floor = 0.85
    multiplier = min(max(multiplier, overlay_floor), overlay_cap)
    notes.append(f"Capped/Floored ({overlay_floor:.2f}-{overlay_cap:.2f})")

    if detailed:
        return multiplier, notes
    else:
        return multiplier

# ==== UI ====
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files (1-2 min, be patient)..."):
        event_df = safe_read(event_file)
        today_df = safe_read(today_file)

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

        st.write(f"Loaded event-level: {getattr(event_file, 'name', 'event_file')} shape {event_df.shape}")
        st.write(f"Loaded today: {getattr(today_file, 'name', 'today_file')} shape {today_df.shape}")

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

    st.write("Dropping columns with >25% missing or near-zero variance...")
    event_df, event_dropped = drop_high_na_low_var(event_df, thresh_na=0.25, thresh_var=1e-7)
    today_df, today_dropped = drop_high_na_low_var(today_df, thresh_na=0.25, thresh_var=1e-7)
    st.write("Dropped columns from event-level data:")
    st.write(event_dropped)
    st.write("Dropped columns from today data:")
    st.write(today_dropped)

    # === CLUSTER-BASED FEATURE SELECTION ===
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    X_for_cluster = event_df[feature_cols]
    selected_features, clusters, cluster_dropped = cluster_select_features(X_for_cluster, threshold=0.95)
    st.write(f"Feature clusters (threshold 0.95):")
    for i, cluster in enumerate(clusters):
        st.write(f"Cluster {i+1}: {cluster}")
    st.write("Selected features from clusters:")
    st.write(selected_features)
    st.write("Dropped features from clusters:")
    st.write(cluster_dropped)

    # === FINAL PREP ===
    X = clean_X(event_df[selected_features])
    y = event_df[target_col]
    X_today = clean_X(today_df[selected_features], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

    st.write("DEBUG: X shape:", X.shape)
    st.write("DEBUG: y shape:", y.shape)

    st.write("Splitting for validation and scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_today_scaled = scaler.transform(X_today)

    # =========== DEEP RESEARCH ENSEMBLE (SOFT VOTING) ===========
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

    model_status = []
    models_for_ensemble = []
    try:
        xgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('xgb', xgb_clf))
        model_status.append('XGB OK')
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")
    try:
        lgb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lgb', lgb_clf))
        model_status.append('LGB OK')
    except Exception as e:
        st.warning(f"LightGBM failed: {e}")
    try:
        cat_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('cat', cat_clf))
        model_status.append('CatBoost OK')
    except Exception as e:
        st.warning(f"CatBoost failed: {e}")
    try:
        rf_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('rf', rf_clf))
        model_status.append('RF OK')
    except Exception as e:
        st.warning(f"RandomForest failed: {e}")
    try:
        gb_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('gb', gb_clf))
        model_status.append('GB OK')
    except Exception as e:
        st.warning(f"GBM failed: {e}")
    try:
        lr_clf.fit(X_train_scaled, y_train)
        models_for_ensemble.append(('lr', lr_clf))
        model_status.append('LR OK')
    except Exception as e:
        st.warning(f"LogReg failed: {e}")

    st.info("Model training status: " + ', '.join(model_status))
    if not models_for_ensemble:
        st.error("All models failed to train! Try reducing features or rows.")
        st.stop()

    # =========== CALIBRATION (ISOTONIC REGRESSION) ===========
    st.write("Fitting ensemble (soft voting)...")
    ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
    ensemble.fit(X_train_scaled, y_train)
    st.write("Fitting isotonic regression calibration...")
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv='prefit')
    calibrated_ensemble.fit(X_val_scaled, y_val)

    # =========== VALIDATION ===========
    st.write("Validating...")
    y_val_pred = calibrated_ensemble.predict_proba(X_val_scaled)[:,1]
    auc = roc_auc_score(y_val, y_val_pred)
    ll = log_loss(y_val, y_val_pred)
    st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

    with st.expander("Show Calibration Curve (Validation Set)"):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_pred, n_bins=20)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
        ax.plot([0,1],[0,1], "--", color="k")
        ax.set_xlabel("Predicted HR Probability")
        ax.set_ylabel("Observed HR Rate")
        ax.legend()
        ax.set_title("Calibration Curve — Validation Set")
        st.pyplot(fig)

    # =========== PREDICT ===========
    st.write("Predicting HR probability for today...")
    today_df['hr_probability'] = calibrated_ensemble.predict_proba(X_today_scaled)[:,1]

    # ==== APPLY ADVANCED OVERLAY SCORING ====
    st.write("Applying post-prediction overlay (weather, park, power, platoon, order, streaks)...")
    overlays = today_df.apply(lambda row: overlay_multiplier(row, detailed=True), axis=1, result_type="expand")
    today_df['overlay_multiplier'] = overlays[0]
    today_df['overlay_notes'] = overlays[1].apply(lambda l: " | ".join(l) if isinstance(l, list) else "")

    today_df['final_hr_probability'] = (today_df['hr_probability'] * today_df['overlay_multiplier']).clip(0, 1)

    # ==== SHOW LEADERBOARD TOP 15 (with overlay factors) ====
    leaderboard_cols = []
    if "player_name" in today_df.columns:
        leaderboard_cols.append("player_name")
    leaderboard_cols += ["hr_probability", "overlay_multiplier", "final_hr_probability", "overlay_notes"]
    leaderboard = today_df[leaderboard_cols].sort_values("final_hr_probability", ascending=False).reset_index(drop=True).head(15)
    leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
    leaderboard["final_hr_probability"] = leaderboard["final_hr_probability"].round(4)
    leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)

    st.markdown("### 🏆 **Today's HR Probabilities & Overlay Multipliers — Top 15**")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button("⬇️ Download Full Prediction CSV", data=today_df.to_csv(index=False), file_name="today_hr_predictions.csv")
    st.download_button("⬇️ Download Overlay Factors CSV", data=leaderboard.to_csv(index=False), file_name="today_overlay_factors.csv")

    # ==== SHOW ADVANCED DIAGNOSTICS: Top 5 Overlay Scoring Breakdown ====
    N_DIAG = 5  # Change to see more or fewer players

    st.markdown(f"### 🔬 Overlay Diagnostics — Top {N_DIAG} Players")
    diag_cols = ["player_name", "hr_probability", "overlay_multiplier", "final_hr_probability", "overlay_notes"]
    top_diag = leaderboard.head(N_DIAG).copy()
    st.dataframe(top_diag[diag_cols], use_container_width=True)

    # Expanders for detailed notes for each player
    for idx, row in top_diag.iterrows():
        with st.expander(f"Scoring breakdown for {row['player_name']}"):
            st.markdown(f"**Raw HR Probability:** {row['hr_probability']:.4f}")
            st.markdown(f"**Overlay Multiplier:** {row['overlay_multiplier']:.3f}")
            st.markdown(f"**Final HR Probability:** {row['final_hr_probability']:.4f}")
            st.markdown(f"**Overlay Notes:** {row['overlay_notes']}")

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
