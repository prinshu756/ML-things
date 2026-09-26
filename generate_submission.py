import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

CATEGORICAL_FEATURES = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level"
]

# Local paths
DATA_DIR = "D:/MachineLearning/Random_shit/playground-series-s6e9"
REFERENCE_PATH = f"{DATA_DIR}/sample_submission.csv"  # fallback to sample submission

def build_row_local_features(frame):
    out = frame.copy()
    income = out["Annual_Income_USD"].to_numpy(dtype=np.int64)
    commute_x10 = np.round(out["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64)

    out["inc_d1"] = (income % 10).astype("int8")
    out["inc_d2"] = (income // 10 % 10).astype("int8")
    out["inc_d3"] = (income // 100 % 10).astype("int8")
    out["inc_mod100"] = (income % 100).astype("int16")
    out["inc_mod1000"] = (income % 1000).astype("int16")
    out["km_d1"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int8")

    for d in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{d}"] = (income // d).astype("int32")

    for d in (5, 10, 25, 50):
        out[f"km_q{d}"] = (commute_x10 // d).astype("int32")

    out["is_30k_spike"] = (income == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income >= 170537).astype("int8")
    out["is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype("int8")
    out["is_env_hater"] = (out["Environmental_Concern_Level"] == 1).astype("int8")

    keys = pd.DataFrame(index=out.index)
    keys["k_inc_exact"] = income.astype(str)
    keys["k_inc100"] = (income // 100).astype(str)
    keys["k_inc1000"] = (income // 1000).astype(str)
    keys["k_km_int"] = (commute_x10 // 10).astype(str)

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
            raise RuntimeError(f"Unseen category in {col}: test={sorted(unseen_test)}")

        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        X_train[col] = train_strings.astype(dtype)
        X_test[col] = test_strings.astype(dtype)

    for smooth, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        encoder = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
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

print("Loading data...")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

y = (
    train["Will_Buy_EV"].map({"No": 0, "Yes": 1})
    if not pd.api.types.is_numeric_dtype(train["Will_Buy_EV"])
    else train["Will_Buy_EV"]
).to_numpy(np.uint8)

X_train, train_keys = build_row_local_features(train.drop(columns=["id", "Will_Buy_EV"]))
X_test, test_keys = build_row_local_features(test.drop(columns="id"))

print("Building full-data features...")

X_train, X_test = prepare_full_features(X_train, train_keys, X_test, test_keys, y)

income = train["Annual_Income_USD"].to_numpy()
test_income = test["Annual_Income_USD"].to_numpy()

for resolution in (8192, 16384):
    train_block, test_block = income_asymmetric_full(income, y, test_income, resolution)

    for j in range(1 if resolution == 16384 else 0, train_block.shape[1]):
        name = f"own_income_{resolution}_{j}"
        X_train[name] = train_block[:, j]
        X_test[name] = test_block[:, j]

print(f"Feature count: {X_train.shape[1]}")
assert X_train.shape[1] == 112, f"Expected 112 features, got {X_train.shape[1]}"

print(f"Training rows: {len(X_train):,}")
print("Training 112-feature model on full dataset...")

model = LGBMClassifier(n_estimators=650, learning_rate=0.04, max_depth=4, num_leaves=32, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=255, feature_pre_filter=False, random_state=42, n_jobs=6, verbose=-1, metric="auc")

model.fit(X_train, y)
pred = model.predict_proba(X_test)[:, 1]
print(f"Model prediction range: {pred.min():.10f} to {pred.max():.10f}")

# Try to load reference, fall back to sample submission
try:
    reference = pd.read_csv(REFERENCE_PATH)
    print(f"Loaded reference from {REFERENCE_PATH}")
except:
    print("Reference file not found, using model predictions only")
    reference = None

if reference is not None:
    assert test["id"].is_unique, "Test IDs contain duplicates"
    assert reference["id"].is_unique, "Reference IDs contain duplicates"
    assert len(test) == len(reference), "Test and reference row counts differ"
    assert set(test["id"]) == set(reference["id"]), "Test and reference IDs differ"
    reference = reference.set_index("id").loc[test["id"]].reset_index()
    assert (reference["id"].to_numpy() == test["id"].to_numpy()).all()
    reference_pred = reference["Will_Buy_EV"].to_numpy(dtype=np.float64)
    assert len(pred) == len(reference_pred)
    blend = 0.9 * reference_pred + 0.2 * pred  # overclocked ;)
    print(f"Reference range: {reference_pred.min():.10f} to {reference_pred.max():.10f}")
    print(f"Blend range: {blend.min():.10f} to {blend.max():.10f}")
    final_pred = blend
else:
    final_pred = pred

pd.DataFrame({"id": test["id"], "Will_Buy_EV": final_pred}).to_csv(f"{DATA_DIR}/submission.csv", index=False)
print("submission.csv saved!")
if reference is not None:
    print("Blend: 0.9 × reference + 0.2 × 112-feature model (overclocked)")
else:
    print("Using model predictions only")