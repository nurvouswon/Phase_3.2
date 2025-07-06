import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import shap
import optuna
from scipy.stats import ks_2samp
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config("🏆 MLB HR Predictor — AI Top 10 World Class", layout="wide")
st.title("🏆 MLB Home Run Predictor — AI-Powered Top 10 Precision Booster")

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

def downcast_df(df):
    float_cols = df.select_dtypes(include=['float'])
    int_cols = df.select_dtypes(include=['int', 'int64', 'int32'])
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def nan_inf_check(df, name):
    numeric_df = df.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors='coerce')
    arr = numeric_df.to_numpy(dtype=np.float64, copy=False)
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

# ==== UI ====
st.sidebar.header("⚡️ AI Mode")
use_world_class = st.sidebar.checkbox("World Class Explainer (Optuna/Stacking/SHAP)", value=False)

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

    # ==== ALL MUTUAL FEATURES ====
    feat_cols_train = set(get_valid_feature_cols(event_df))
    feat_cols_today = set(get_valid_feature_cols(today_df))
    feature_cols = sorted(list(feat_cols_train & feat_cols_today))
    st.write(f"Number of features (no deduplication): {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    y = event_df[target_col]
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)
    X = downcast_df(X)
    X_today = downcast_df(X_today)

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

    if not use_world_class:
        # =========== CLASSIC AI ENSEMBLE (SOFT VOTING) ===========
        st.write("Training base models (XGB, LGBM, CatBoost, RF, GB, LR)...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=80, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss',
            n_jobs=1, verbosity=1, tree_method='hist'
        )
        lgb_clf = lgb.LGBMClassifier(n_estimators=80, max_depth=6, learning_rate=0.07, n_jobs=1)
        cat_clf = cb.CatBoostClassifier(iterations=80, depth=6, learning_rate=0.08, verbose=0, thread_count=1)
        rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, n_jobs=1)
        gb_clf = GradientBoostingClassifier(n_estimators=60, max_depth=6, learning_rate=0.08)
        lr_clf = LogisticRegression(max_iter=600, solver='lbfgs', n_jobs=1)

        model_status = []
        models_for_ensemble = []
        importances = {}
        try:
            xgb_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('xgb', xgb_clf))
            model_status.append('XGB OK')
            importances['XGB'] = xgb_clf.feature_importances_
        except Exception as e:
            st.warning(f"XGBoost failed: {e}")
        try:
            lgb_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('lgb', lgb_clf))
            model_status.append('LGB OK')
            importances['LGB'] = lgb_clf.feature_importances_
        except Exception as e:
            st.warning(f"LightGBM failed: {e}")
        try:
            cat_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('cat', cat_clf))
            model_status.append('CatBoost OK')
            importances['CatBoost'] = cat_clf.feature_importances_
        except Exception as e:
            st.warning(f"CatBoost failed: {e}")
        try:
            rf_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('rf', rf_clf))
            model_status.append('RF OK')
            importances['RF'] = rf_clf.feature_importances_
        except Exception as e:
            st.warning(f"RandomForest failed: {e}")
        try:
            gb_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('gb', gb_clf))
            model_status.append('GB OK')
            importances['GB'] = gb_clf.feature_importances_
        except Exception as e:
            st.warning(f"GBM failed: {e}")
        try:
            lr_clf.fit(X_train_scaled, y_train)
            models_for_ensemble.append(('lr', lr_clf))
            model_status.append('LR OK')
            importances['LR'] = np.abs(lr_clf.coef_[0])
        except Exception as e:
            st.warning(f"LogReg failed: {e}")

        st.info("Model training status: " + ', '.join(model_status))
        if not models_for_ensemble:
            st.error("All models failed to train! Try reducing features or rows.")
            st.stop()

        st.write("Fitting ensemble (soft voting)...")
        ensemble = VotingClassifier(estimators=models_for_ensemble, voting='soft', n_jobs=1)
        ensemble.fit(X_train_scaled, y_train)

        # =========== FEATURE IMPORTANCE DIAGNOSTICS ===========
        st.markdown("## 🔍 Feature Importances (Mean of Tree Models)")
        tree_keys = [k for k in importances.keys() if k in ("XGB", "LGB", "CatBoost", "RF", "GB")]
        if tree_keys:
            tree_importances = np.mean([importances[k] for k in tree_keys], axis=0)
            import_df = pd.DataFrame({
                "feature": X.columns,
                "importance": tree_importances
            }).sort_values("importance", ascending=False)
            st.dataframe(import_df.head(30), use_container_width=True)
        else:
            st.warning("Tree model feature importances not available.")

        # =========== VALIDATION ===========
        st.write("Validating (out-of-fold, not test-leak)...")
        y_val_pred = ensemble.predict_proba(X_val_scaled)[:,1]
        auc = roc_auc_score(y_val, y_val_pred)
        ll = log_loss(y_val, y_val_pred)
        st.info(f"Validation AUC: **{auc:.4f}** — LogLoss: **{ll:.4f}**")

        # =========== CALIBRATION (Isotonic Regression) ===========
        st.write("Calibrating prediction probabilities (isotonic regression, deep research)...")
        ir = IsotonicRegression(out_of_bounds="clip")
        y_val_pred_cal = ir.fit_transform(y_val_pred, y_val)
        # =========== PREDICT ===========
        st.write("Predicting HR probability for today (calibrated)...")
        y_today_pred = ensemble.predict_proba(X_today_scaled)[:, 1]
        y_today_pred_cal = ir.transform(y_today_pred)
        today_df['hr_probability'] = y_today_pred_cal

        # =========== POST-PREDICTION RANK BOOSTER ===========
        st.write("🔮 Post-Prediction Top 10 Booster (AI meta-prediction)...")
        today_df = today_df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
        topN = 15
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

        # ==== TOP 10 LEADERBOARD AND MULTIPLIER PLOT ====
        desired_cols = ["player_name", "team", "game_time", "opposing_pitcher", "hr_probability", "meta_hr_rank_score"]
        cols = [c for c in desired_cols if c in today_df.columns]
        leaderboard = today_df[cols].copy()
        leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
        leaderboard["meta_hr_rank_score"] = leaderboard["meta_hr_rank_score"].round(4)
        top_n = 10
        leaderboard_top = leaderboard.head(top_n)
        st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (AI Top 10 Booster)**")
        st.dataframe(leaderboard_top, use_container_width=True)
        st.download_button(
            f"⬇️ Download Top {top_n} Leaderboard CSV",
            data=leaderboard_top.to_csv(index=False),
            file_name=f"top{top_n}_leaderboard.csv"
        )

        st.subheader("💸 Outlay Multiplier Strength (Top 10)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            leaderboard_top["player_name"].astype(str),
            leaderboard_top["meta_hr_rank_score"],
            color="mediumseagreen"
        )
        ax.set_xlabel("Meta HR Rank Score (Multiplier Strength)")
        ax.set_ylabel("Player")
        ax.invert_yaxis()
        for i, v in enumerate(leaderboard_top["meta_hr_rank_score"]):
            ax.text(v, i, f"{v:.2f}", va='center')
        st.pyplot(fig)

    else:
        # ==== WORLD CLASS: OPTUNA + STACKING + SHAP EXPLAINER ====
        st.markdown("## 🚀 **World Class Stacking AI (Optuna, SHAP, Platt/Isotonic Calib, Advanced Meta)**")
        def drift_check(train, test):
            p_values = [ks_2samp(train[:, i], test[:, i]).pvalue for i in range(train.shape[1])]
            return np.any(np.array(p_values) < 0.05)
        if drift_check(X_train_scaled, X_today_scaled):
            st.warning("⚠️ Drift detected: Consider retraining your model.")

        def opt_objective(trial):
            clf = xgb.XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 150),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('lr', 0.01, 0.2),
                eval_metric='logloss',
                use_label_encoder=False
            )
            clf.fit(X_train_scaled, y_train)
            return roc_auc_score(y_val, clf.predict_proba(X_val_scaled)[:, 1])
        study = optuna.create_study(direction='maximize')
        study.optimize(opt_objective, n_trials=20)
        xgb_clf = xgb.XGBClassifier(**study.best_params, eval_metric='logloss', use_label_encoder=False)
        lgb_clf = lgb.LGBMClassifier(n_estimators=100)
        cat_clf = cb.CatBoostClassifier(iterations=100, verbose=0)
        rf_clf = RandomForestClassifier(n_estimators=80)
        stack = StackingClassifier(
            estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf), ('rf', rf_clf)],
            final_estimator=LogisticRegression(),
            cv=StratifiedKFold(5)
        )
        stack.fit(X_train_scaled, y_train)
        iso = IsotonicRegression().fit(stack.predict_proba(X_val_scaled)[:, 1], y_val)
        platt_calib = CalibratedClassifierCV(stack, method='sigmoid', cv='prefit').fit(X_val_scaled, y_val)
        preds_iso = iso.transform(stack.predict_proba(X_today_scaled)[:, 1])
        preds_platt = platt_calib.predict_proba(X_today_scaled)[:, 1]
        today_df['hr_probability'] = (preds_iso + preds_platt) / 2

        today_df['Momentum'] = today_df['hr_probability'].diff().fillna(0)
        today_df['Fatigue'] = today_df['hr_probability'].rolling(window=3).mean().fillna(method='bfill')
        meta_feats = today_df[['hr_probability', 'Momentum', 'Fatigue']]
        meta_y = today_df['hr_probability'].copy()
        meta_y.iloc[:10] += 0.1
        meta_boost = GradientBoostingRegressor(n_estimators=120)
        meta_boost.fit(meta_feats, meta_y)
        today_df['meta_hr_rank_score'] = meta_boost.predict(meta_feats)
        today_df = today_df.sort_values('meta_hr_rank_score', ascending=False).reset_index(drop=True)

        # ==== TOP 10 LEADERBOARD AND MULTIPLIER PLOT ====
        desired_cols = ["player_name", "team", "game_time", "opposing_pitcher", "hr_probability", "meta_hr_rank_score"]
        cols = [c for c in desired_cols if c in today_df.columns]
        leaderboard = today_df[cols].copy()
        leaderboard["hr_probability"] = leaderboard["hr_probability"].round(4)
        leaderboard["meta_hr_rank_score"] = leaderboard["meta_hr_rank_score"].round(4)
        top_n = 10
        leaderboard_top = leaderboard.head(top_n)
        st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (World Class Booster)**")
        st.dataframe(leaderboard_top, use_container_width=True)
        st.download_button(
            f"⬇️ Download Top {top_n} Leaderboard CSV",
            data=leaderboard_top.to_csv(index=False),
            file_name=f"top{top_n}_worldclass_leaderboard.csv"
        )

        st.subheader("💸 Outlay Multiplier Strength (Top 10)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            leaderboard_top["player_name"].astype(str),
            leaderboard_top["meta_hr_rank_score"],
            color="mediumseagreen"
        )
        ax.set_xlabel("Meta HR Rank Score (Multiplier Strength)")
        ax.set_ylabel("Player")
        ax.invert_yaxis()
        for i, v in enumerate(leaderboard_top["meta_hr_rank_score"]):
            ax.text(v, i, f"{v:.2f}", va='center')
        st.pyplot(fig)

        # ==== SHAP Feature Impact Plot ====
        st.subheader("📈 SHAP Feature Impact (XGB)")
        try:
            explainer = shap.TreeExplainer(xgb_clf.fit(X_train_scaled, y_train))
            shap_values = explainer.shap_values(X_today_scaled)
            shap.summary_plot(shap_values, X_today, plot_type="dot", show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.warning(f"SHAP summary plot error: {e}")

        # ==== HR Probability Distribution Plot ====
        st.subheader("📊 HR Probability Distribution (Top 10)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(leaderboard_top["player_name"].astype(str), leaderboard_top["hr_probability"], color='dodgerblue')
        ax.invert_yaxis()
        ax.set_xlabel('HR Probability')
        ax.set_ylabel('Player')
        st.pyplot(fig)

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
