import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
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
    
    # Polynomial features
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

    for key in train_keys.columns:
        freq = train_keys[key].value_counts(normalize=True, dropna=False)
        name = f"{key}_fe"
        X_train[name] = map_zero(train_keys[key], freq)
        X_test[name] = map_zero(test_keys[key], freq)

    for col in CATEGORICAL_FEATURES:
        train_strings = X_train[col].astype(str).reset_index(drop=True)
        test_strings = X_test[col].astype(str).reset_index(drop=True)
        categories = sorted(train_strings.unique().tolist())
        unseen_test = set(test_strings.unique()) - set(categories)
        if unseen_test:
            test_strings = test_strings.map(lambda x: x if x in categories else categories[0])
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        X_train[col] = train_strings.astype(dtype)
        X_test[col] = test_strings.astype(dtype)

    # Target encoding with different smoothing
    for smooth, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100"), (1000.0, "1000")):
        encoder = TargetEncoder(cv=5, smooth=smooth, random_state=42)
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

def train_lgbm_fold(X_train, y, X_test, params, n_folds=5, seed=42):
    """Train LGBM with cross-validation, return OOF and test predictions"""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    scores = []
    
    cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == 'category']
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = LGBMClassifier(**params, random_state=seed + fold, n_jobs=6, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                  callbacks=[])  # No early stopping for speed
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_pred += model.predict_proba(X_test)[:, 1] / n_folds
        scores.append(roc_auc_score(y_val, val_pred))
    
    return oof, test_pred, np.mean(scores)

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

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target dist: {pd.Series(y).value_counts().to_dict()}")

X_train, train_keys = build_row_local_features(train.drop(columns=["id", "Will_Buy_EV"]))
X_test, test_keys = build_row_local_features(test.drop(columns="id"))

print("\nBuilding features...")
X_train, X_test = prepare_full_features(X_train, train_keys, X_test, test_keys, y)

income = train["Annual_Income_USD"].to_numpy()
test_income = test["Annual_Income_USD"].to_numpy()

for resolution in (8192, 16384, 32768):
    train_block, test_block = income_asymmetric_full(income, y, test_income, resolution)
    for j in range(1 if resolution == 16384 else 0, train_block.shape[1]):
        name = f"own_income_{resolution}_{j}"
        X_train[name] = train_block[:, j]
        X_test[name] = test_block[:, j]

print(f"\nTotal features: {X_train.shape[1]}")
print(f"Training rows: {len(X_train):,}")

# ============================================================
# Train 3 diverse LGBM models with CV
# ============================================================
print("\n" + "=" * 60)
print("Training 3 diverse LGBM models (5-fold CV)")
print("=" * 60)

models_config = [
    {
        'name': 'lgbm_shallow',
        'params': dict(n_estimators=800, learning_rate=0.03, max_depth=4, num_leaves=31,
                       min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.35,
                       reg_alpha=0.05, reg_lambda=1.5, max_bin=255, feature_pre_filter=False, metric="auc")
    },
    {
        'name': 'lgbm_deep',
        'params': dict(n_estimators=1000, learning_rate=0.02, max_depth=5, num_leaves=63,
                       min_child_samples=20, subsample=0.75, subsample_freq=1, colsample_bytree=0.4,
                       reg_alpha=0.1, reg_lambda=2.0, max_bin=255, feature_pre_filter=False, metric="auc")
    },
    {
        'name': 'lgbm_wide',
        'params': dict(n_estimators=600, learning_rate=0.04, max_depth=6, num_leaves=127,
                       min_child_samples=10, subsample=0.85, subsample_freq=1, colsample_bytree=0.5,
                       reg_alpha=0.01, reg_lambda=1.0, max_bin=255, feature_pre_filter=False, metric="auc")
    }
]

all_oof = []
all_test = []
all_scores = []

for cfg in models_config:
    print(f"\n  Training {cfg['name']}...")
    oof, test_pred, cv_score = train_lgbm_fold(X_train, y, X_test, cfg['params'], n_folds=5, seed=42)
    all_oof.append(oof)
    all_test.append(test_pred)
    all_scores.append(cv_score)
    print(f"    CV AUC: {cv_score:.6f}")

all_oof = np.column_stack(all_oof)
all_test = np.column_stack(all_test)

# Simple weighted ensemble (weight by CV score)
weights = np.array(all_scores)
weights = weights / weights.sum()
ensemble_oof = np.average(all_oof, axis=1, weights=weights)
ensemble_test = np.average(all_test, axis=1, weights=weights)
ensemble_auc = roc_auc_score(y, ensemble_oof)

print(f"\n  Individual CV scores: {[f'{s:.6f}' for s in all_scores]}")
print(f"  Weights: {weights}")
print(f"  Weighted ensemble CV AUC: {ensemble_auc:.6f}")

# Also try equal weight
equal_oof = np.mean(all_oof, axis=1)
equal_test = np.mean(all_test, axis=1)
equal_auc = roc_auc_score(y, equal_oof)
print(f"  Equal weight ensemble CV AUC: {equal_auc:.6f}")

# Use best
if ensemble_auc > equal_auc:
    final_oof = ensemble_oof
    final_test = ensemble_test
    print(f"  Using weighted ensemble")
else:
    final_oof = equal_oof
    final_test = equal_test
    print(f"  Using equal-weight ensemble")

# ============================================================
# Pseudo-labeling with confident predictions
# ============================================================
print("\n" + "=" * 60)
print("Pseudo-labeling confident test samples")
print("=" * 60)

# Use high-confidence predictions as pseudo-labels
threshold_high = 0.97
threshold_low = 0.03

confident_high = final_test >= threshold_high
confident_low = final_test <= threshold_low
confident = confident_high | confident_low
n_confident = confident.sum()

if n_confident > 100:  # Only if we have enough confident samples
    print(f"  Found {n_confident} confident samples ({confident_high.sum()} high, {confident_low.sum()} low)")
    
    X_pseudo = X_test[confident].copy()
    y_pseudo = (final_test[confident] >= 0.5).astype(np.uint8)
    
    X_aug = pd.concat([X_train, X_pseudo], ignore_index=True)
    y_aug = np.concatenate([y, y_pseudo])
    
    print(f"  Augmented train: {len(X_aug)} rows (original: {len(X_train)})")
    
    # Retrain a single model on augmented data
    aug_model = LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=4, num_leaves=31,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.35,
        reg_alpha=0.05, reg_lambda=1.5, max_bin=255, feature_pre_filter=False,
        random_state=42, n_jobs=6, verbose=-1, metric="auc"
    )
    aug_model.fit(X_aug, y_aug)
    aug_pred = aug_model.predict_proba(X_test)[:, 1]
    
    # Blend with previous
    final_test = 0.7 * final_test + 0.3 * aug_pred
    print(f"  Blended with pseudo-labeled model (0.7 * ensemble + 0.3 * pseudo)")
else:
    print(f"  Not enough confident samples ({n_confident}), skipping pseudo-labeling")

# ============================================================
# Final blend with reference
# ============================================================
try:
    reference = pd.read_csv(REFERENCE_PATH)
    reference = reference.set_index("id").loc[test["id"]].reset_index()
    reference_pred = reference["Will_Buy_EV"].to_numpy(dtype=np.float64)
    
    # Optimize blend: 0.85 * reference + 0.25 * model (overclocked)
    blend = 0.85 * reference_pred + 0.25 * final_test
    print(f"\nBlending with reference: 0.85 * ref + 0.25 * model")
    print(f"  Reference range: {reference_pred.min():.6f} to {reference_pred.max():.6f}")
    print(f"  Model range: {final_test.min():.6f} to {final_test.max():.6f}")
    print(f"  Blend range: {blend.min():.6f} to {blend.max():.6f}")
    final_test = blend
except:
    print("\nNo reference file found, using model predictions only")

final_test = np.clip(final_test, 1e-7, 1 - 1e-7)

submission = pd.DataFrame({"id": test["id"], "Will_Buy_EV": final_test})
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print(f"\nSubmission saved!")
print(f"Final stats: min={final_test.min():.6f}, max={final_test.max():.6f}, mean={final_test.mean():.6f}")