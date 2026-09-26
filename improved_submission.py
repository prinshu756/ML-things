import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

CATEGORICAL_FEATURES = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level"
]

DATA_DIR = "D:/MachineLearning/Random_shit/playground-series-s6e9"
REFERENCE_PATH = f"{DATA_DIR}/sample_submission.csv"

def build_row_local_features(frame):
    out = frame.copy()
    income = out["Annual_Income_USD"].to_numpy(dtype=np.int64)
    commute = out["Daily_Commute_km"].to_numpy(dtype=float)
    commute_x10 = np.round(commute * 10).astype(np.int64)
    age = out["Age"].to_numpy(dtype=np.int64)

    # Income digit features
    out["inc_d1"] = (income % 10).astype("int8")
    out["inc_d2"] = (income // 10 % 10).astype("int8")
    out["inc_d3"] = (income // 100 % 10).astype("int8")
    out["inc_mod100"] = (income % 100).astype("int16")
    out["inc_mod1000"] = (income % 1000).astype("int16")
    
    # Commute digit features
    out["km_d1"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int8")

    # Quantile bins
    for d in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{d}"] = (income // d).astype("int32")
    for d in (5, 10, 25, 50):
        out[f"km_q{d}"] = (commute_x10 // d).astype("int32")
    for d in (10, 20, 30, 40, 50):
        out[f"age_q{d}"] = (age // d).astype("int32")

    # Special indicators
    out["is_30k_spike"] = (income == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income >= 170537).astype("int8")
    out["is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype("int8")
    out["is_env_hater"] = (out["Environmental_Concern_Level"] == 1).astype("int8")
    out["is_high_income"] = (income >= 150000).astype("int8")
    out["is_low_income"] = (income <= 20000).astype("int8")
    out["is_long_commute"] = (commute >= 50).astype("int8")
    out["is_short_commute"] = (commute <= 5).astype("int8")

    # Interaction features
    out["inc_per_age"] = (income / (age + 1)).astype(np.float32)
    out["commute_per_income"] = (commute / (income + 1)).astype(np.float32)
    out["income_x_env"] = income * out["Environmental_Concern_Level"].to_numpy()
    out["age_x_env"] = age * out["Environmental_Concern_Level"].to_numpy()
    out["cars_x_charging_home"] = out["Number_of_Cars_Owned"] * out["Charging_Stations_Near_Home"]
    out["cars_x_charging_work"] = out["Number_of_Cars_Owned"] * out["Charging_Stations_Near_Work"]
    
    # Polynomial features for key numerics
    out["income_sq"] = (income ** 2).astype(np.float32)
    out["commute_sq"] = (commute ** 2).astype(np.float32)
    out["age_sq"] = (age ** 2).astype(np.float32)
    out["income_log"] = np.log1p(income).astype(np.float32)
    out["commute_log"] = np.log1p(commute).astype(np.float32)

    # Keys for target encoding
    keys = pd.DataFrame(index=out.index)
    keys["k_inc_exact"] = income.astype(str)
    keys["k_inc100"] = (income // 100).astype(str)
    keys["k_inc1000"] = (income // 1000).astype(str)
    keys["k_km_int"] = (commute_x10 // 10).astype(str)
    keys["k_age"] = age.astype(str)
    keys["k_inc_age"] = pd.Series(income).astype(str) + "_" + pd.Series(age).astype(str)
    keys["k_inc_km"] = pd.Series(income).astype(str) + "_" + pd.Series(commute_x10).astype(str)
    keys["k_age_km"] = pd.Series(age).astype(str) + "_" + pd.Series(commute_x10).astype(str)

    for col in CATEGORICAL_FEATURES + [
        "Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work", "Environmental_Concern_Level"
    ]:
        keys[f"k_{col}"] = out[col].astype(str).to_numpy()

    return out.reset_index(drop=True), keys.reset_index(drop=True)

def map_zero(values, mapping):
    return values.map(mapping).fillna(0.0).astype(np.float32).to_numpy()

def prepare_full_features(X_train, train_keys, X_test, test_keys, y):
    X_train = X_train.reset_index(drop=True).copy()
    X_test = X_test.reset_index(drop=True).copy()
    train_keys = train_keys.reset_index(drop=True).copy()
    test_keys = test_keys.reset_index(drop=True).copy()

    # Frequency encoding for income and commute
    income_train = pd.Series(X_train["Annual_Income_USD"].to_numpy(dtype=np.int64))
    income_test = pd.Series(X_test["Annual_Income_USD"].to_numpy(dtype=np.int64))
    income_counts = income_train.value_counts(dropna=False)
    X_train["fq_inc"] = map_zero(income_train, income_counts)
    X_test["fq_inc"] = map_zero(income_test, income_counts)

    commute_train = pd.Series(np.round(X_train["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64))
    commute_test = pd.Series(np.round(X_test["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64))
    commute_counts = commute_train.value_counts(dropna=False)
    X_train["fq_km"] = map_zero(commute_train, commute_counts)
    X_test["fq_km"] = map_zero(commute_test, commute_counts)

    # Frequency encoding for all keys
    for key in train_keys.columns:
        freq = train_keys[key].value_counts(normalize=True, dropna=False)
        name = f"{key}_fe"
        X_train[name] = map_zero(train_keys[key], freq)
        X_test[name] = map_zero(test_keys[key], freq)

    # Categorical encoding
    for col in CATEGORICAL_FEATURES:
        train_strings = X_train[col].astype(str).reset_index(drop=True)
        test_strings = X_test[col].astype(str).reset_index(drop=True)
        categories = sorted(train_strings.unique().tolist())
        unseen_test = set(test_strings.unique()) - set(categories)
        if unseen_test:
            # Map unseen to most frequent
            test_strings = test_strings.map(lambda x: x if x in categories else categories[0])
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        X_train[col] = train_strings.astype(dtype)
        X_test[col] = test_strings.astype(dtype)

    # Target encoding with different smoothing
    from sklearn.model_selection import KFold
    for smooth, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100"), (1000.0, "1000")):
        cv = StratifiedKFold(5, shuffle=True, random_state=42) if smooth == "auto" else KFold(5, shuffle=True, random_state=42)
        encoder = TargetEncoder(cv=cv, smooth=smooth, random_state=42)
        encoded_train = encoder.fit_transform(train_keys, y)
        encoded_test = encoder.transform(test_keys)

        for i, key in enumerate(train_keys.columns):
            name = f"{key}_te{tag}"
            X_train[name] = encoded_train[:, i].astype(np.float32)
            X_test[name] = encoded_test[:, i].astype(np.float32)

    return X_train, X_test

def bin_statistics(codes, y_fit, n_bins, prior, smooth):
    sums = np.bincount(codes, weights=y_fit, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)

    left_sums, left_counts = np.r_[0.0, sums[:-1]], np.r_[0.0, counts[:-1]]
    right_sums, right_counts = np.r_[sums[1:], 0.0], np.r_[counts[1:], 0.0]

    left = (left_sums + smooth * prior) / (left_counts + smooth)
    right = (right_sums + smooth * prior) / (right_counts + smooth)

    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    neighbor_sums = np.convolve(sums, kernel, mode="same")
    neighbor_counts = np.convolve(counts, kernel, mode="same")
    symmetric = (neighbor_sums + smooth * kernel.sum() * prior) / (
        neighbor_counts + smooth * kernel.sum()
    )

    slope = right - left
    curvature = central - 0.5 * (left + right)

    return np.column_stack([
        central, symmetric, left, right, slope, curvature, np.log1p(counts)
    ]).astype(np.float32)

def income_asymmetric_full(train_income, y, test_income, q, smooth=10.0):
    train_income = np.asarray(train_income, np.float64)
    test_income = np.asarray(test_income, np.float64)
    y = np.asarray(y, np.uint8)

    prior = float(y.mean())
    edges = np.linspace(float(train_income.min()), float(train_income.max()), q + 1)

    train_codes = np.searchsorted(edges[1:-1], train_income)
    test_codes = np.searchsorted(edges[1:-1], test_income)

    n_bins = len(edges)
    position_scale = max(n_bins - 2, 1)
    train_features = np.zeros((len(train_income), 8), dtype=np.float32)

    inner = StratifiedKFold(5, shuffle=True, random_state=17)

    for fit_idx, valid_idx in inner.split(train_codes, y):
        stats = bin_statistics(
            train_codes[fit_idx],
            y[fit_idx],
            n_bins,
            float(y[fit_idx].mean()),
            smooth
        )

        train_features[valid_idx, 0] = train_codes[valid_idx] / position_scale
        train_features[valid_idx, 1:] = stats[train_codes[valid_idx]]

    stats = bin_statistics(train_codes, y, n_bins, prior, smooth)

    test_features = np.column_stack([
        test_codes / position_scale,
        stats[test_codes]
    ]).astype(np.float32)

    return train_features, test_features

def get_models():
    """Return dictionary of models to ensemble"""
    models = {
        'lgbm_1': LGBMClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=4, num_leaves=31,
            min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.35,
            reg_alpha=0.05, reg_lambda=1.5, max_bin=255, feature_pre_filter=False,
            random_state=42, n_jobs=6, verbose=-1, metric="auc"
        ),
        'lgbm_2': LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=5, num_leaves=63,
            min_child_samples=20, subsample=0.75, subsample_freq=1, colsample_bytree=0.4,
            reg_alpha=0.1, reg_lambda=2.0, max_bin=255, feature_pre_filter=False,
            random_state=123, n_jobs=6, verbose=-1, metric="auc"
        ),
        'lgbm_3': LGBMClassifier(
            n_estimators=600, learning_rate=0.04, max_depth=3, num_leaves=15,
            min_child_samples=10, subsample=0.85, subsample_freq=1, colsample_bytree=0.3,
            reg_alpha=0.01, reg_lambda=1.0, max_bin=255, feature_pre_filter=False,
            random_state=456, n_jobs=6, verbose=-1, metric="auc"
        ),
        'xgb_1': XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.35,
            reg_alpha=0.05, reg_lambda=1.5, max_bin=256,
            random_state=42, n_jobs=6, verbosity=0, eval_metric="auc",
            tree_method="hist", enable_categorical=True
        ),
        'xgb_2': XGBClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=5,
            min_child_weight=10, subsample=0.75, colsample_bytree=0.4,
            reg_alpha=0.1, reg_lambda=2.0, max_bin=256,
            random_state=123, n_jobs=6, verbosity=0, eval_metric="auc",
            tree_method="hist", enable_categorical=True
        ),
        'cat_1': CatBoostClassifier(
            iterations=800, learning_rate=0.03, depth=4,
            l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.35,
            random_state=42, verbose=False, thread_count=6,
            eval_metric="AUC"
        ),
        'cat_2': CatBoostClassifier(
            iterations=1000, learning_rate=0.02, depth=5,
            l2_leaf_reg=5, subsample=0.75, colsample_bylevel=0.4,
            random_state=123, verbose=False, thread_count=6,
            eval_metric="AUC"
        ),
    }
    return models

def train_ensemble(X_train, y, X_test, n_folds=5):
    """Train ensemble with cross-validation and return predictions"""
    models = get_models()
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
    
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    test_preds = {name: np.zeros(len(X_test)) for name in models}
    cv_scores = {}
    
    print(f"\nTraining {len(models)} models with {n_folds}-fold CV...")
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            if 'cat' in name:
                cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == 'category']
                model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
            
            # Predict
            val_pred = model.predict_proba(X_val)[:, 1]
            oof_preds[name][val_idx] = val_pred
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            
            test_preds[name] += model.predict_proba(X_test)[:, 1] / n_folds
        
        cv_scores[name] = np.mean(fold_scores)
        print(f"    CV AUC: {cv_scores[name]:.6f} (folds: {[f'{s:.6f}' for s in fold_scores]})")
    
    # Stack OOF predictions for meta-learner
    oof_stack = np.column_stack([oof_preds[name] for name in models])
    test_stack = np.column_stack([test_preds[name] for name in models])
    
    # Train meta-learner (Logistic Regression)
    meta = LogisticRegression(C=0.1, max_iter=1000, random_state=42, n_jobs=6)
    meta.fit(oof_stack, y)
    meta_oof = meta.predict_proba(oof_stack)[:, 1]
    meta_test = meta.predict_proba(test_stack)[:, 1]
    
    meta_auc = roc_auc_score(y, meta_oof)
    print(f"\n  Meta-learner CV AUC: {meta_auc:.6f}")
    
    # Weighted average of all models (including meta)
    all_oof = np.column_stack([oof_stack, meta_oof])
    all_test = np.column_stack([test_stack, meta_test])
    
    # Find optimal weights using simple grid search on OOF
    best_weight = None
    best_auc = 0
    n_models = len(models)
    
    # Simple equal weight
    equal_weight_oof = np.mean(all_oof, axis=1)
    equal_weight_test = np.mean(all_test, axis=1)
    eq_auc = roc_auc_score(y, equal_weight_oof)
    print(f"  Equal weight CV AUC: {eq_auc:.6f}")
    
    # Weight by CV score
    weights = np.array([cv_scores[name] for name in models] + [meta_auc])
    weights = weights / weights.sum()
    weighted_oof = np.average(all_oof, axis=1, weights=weights)
    weighted_test = np.average(all_test, axis=1, weights=weights)
    w_auc = roc_auc_score(y, weighted_oof)
    print(f"  CV-score weighted AUC: {w_auc:.6f}")
    print(f"  Weights: {dict(zip(list(models.keys()) + ['meta'], weights))}")
    
    # Use best
    if w_auc > eq_auc:
        final_test = weighted_test
        print(f"  Using CV-weighted ensemble")
    else:
        final_test = equal_weight_test
        print(f"  Using equal-weight ensemble")
    
    return final_test, oof_preds, test_preds, meta_test

def pseudo_labeling(X_train, y, X_test, test_preds, threshold_high=0.95, threshold_low=0.05):
    """Add confident pseudo-labeled test samples to training"""
    # Get confident predictions
    confident_high = test_preds >= threshold_high
    confident_low = test_preds <= threshold_low
    confident = confident_high | confident_low
    
    n_confident = confident.sum()
    if n_confident > 0:
        print(f"\n  Pseudo-labeling: {n_confident} confident samples ({confident_high.sum()} high, {confident_low.sum()} low)")
        
        # Create augmented training set
        X_pseudo = X_test[confident].copy()
        y_pseudo = (test_preds[confident] >= 0.5).astype(np.uint8)
        
        X_aug = pd.concat([X_train, X_pseudo], ignore_index=True)
        y_aug = np.concatenate([y, y_pseudo])
        
        return X_aug, y_aug
    else:
        print("\n  No confident pseudo-labels found")
        return X_train, y

print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

y = (
    train["Will_Buy_EV"].map({"No": 0, "Yes": 1})
    if not pd.api.types.is_numeric_dtype(train["Will_Buy_EV"])
    else train["Will_Buy_EV"]
).to_numpy(np.uint8)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

X_train, train_keys = build_row_local_features(train.drop(columns=["id", "Will_Buy_EV"]))
X_test, test_keys = build_row_local_features(test.drop(columns="id"))

print("\nBuilding full-data features...")
X_train, X_test = prepare_full_features(X_train, train_keys, X_test, test_keys, y)

# Income asymmetric features
income = train["Annual_Income_USD"].to_numpy()
test_income = test["Annual_Income_USD"].to_numpy()

for resolution in (8192, 16384, 32768):
    train_block, test_block = income_asymmetric_full(income, y, test_income, resolution)
    for j in range(1 if resolution == 16384 else 0, train_block.shape[1]):
        name = f"own_income_{resolution}_{j}"
        X_train[name] = train_block[:, j]
        X_test[name] = test_block[:, j]

print(f"\nTotal feature count: {X_train.shape[1]}")
print(f"Training rows: {len(X_train):,}")

# ============================================================
# First round: Train ensemble and get predictions
# ============================================================
print("\n" + "=" * 60)
print("ROUND 1: Training ensemble")
print("=" * 60)

test_preds_1, oof_1, test_1, meta_1 = train_ensemble(X_train, y, X_test, n_folds=5)

# ============================================================
# Second round: Pseudo-labeling
# ============================================================
print("\n" + "=" * 60)
print("ROUND 2: Pseudo-labeling + retrain")
print("=" * 60)

X_aug, y_aug = pseudo_labeling(X_train, y, X_test, test_preds_1)

if len(X_aug) > len(X_train):
    test_preds_2, _, _, _ = train_ensemble(X_aug, y_aug, X_test, n_folds=3)
    # Blend round 1 and round 2
    final_pred = 0.7 * test_preds_1 + 0.3 * test_preds_2
    print(f"\n  Blended final (0.7 * round1 + 0.3 * round2)")
else:
    final_pred = test_preds_1

# ============================================================
# Final blend with reference if available
# ============================================================
try:
    reference = pd.read_csv(REFERENCE_PATH)
    reference = reference.set_index("id").loc[test["id"]].reset_index()
    reference_pred = reference["Will_Buy_EV"].to_numpy(dtype=np.float64)
    
    # Optimize blend weight
    best_weight = 0.9
    best_auc = 0
    # We can't compute AUC without true labels, but we can use OOF from round 1
    # as proxy. For now use fixed weights.
    blend = 0.85 * reference_pred + 0.25 * final_pred  # overclocked
    print(f"\nBlending with reference: 0.85 * ref + 0.25 * model")
    print(f"Reference range: {reference_pred.min():.6f} to {reference_pred.max():.6f}")
    print(f"Model range: {final_pred.min():.6f} to {final_pred.max():.6f}")
    print(f"Blend range: {blend.min():.6f} to {blend.max():.6f}")
    final_pred = blend
except:
    print("\nNo reference file found, using model predictions only")

# Clip to valid range
final_pred = np.clip(final_pred, 1e-7, 1 - 1e-7)

# Save
submission = pd.DataFrame({"id": test["id"], "Will_Buy_EV": final_pred})
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print(f"\nSubmission saved to {DATA_DIR}/submission.csv")
print(f"Final prediction stats: min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}")