# PREPROCESSING 2 - Scenario 2 (booking-time): drops ADR, ReservationStatus, ReservationStatusDate and IsCanceled.
# Steps: split train/test -> prepare Agent and new features -> impute missing values -> one-hot encode and scale -> verify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Step 0: load the cleaned data and verify the starting point ----
df = pd.read_csv("dataset_cleaned1.csv")
print("Loaded shape:", df.shape)

# Counts rows that repeat an earlier row
print("Remaining duplicates:", df.duplicated().sum())

# Counts missing cells per column, then shows only the columns that have any
missing = df.isna().sum()
print(missing[missing > 0])

# Counts rows where the price is 0 and shows the highest price
print("Zero ADR:", (df["ADR"] == 0).sum())
print("Max ADR:", df["ADR"].max())

# ---- Step 2: features and target (Scenario 2: also drop the columns that only exist after the booking) ----
post_booking_cols = ["ReservationStatus", "ReservationStatusDate", "IsCanceled"]
y = df["ADR"]
X = df.drop(columns=["ADR"] + post_booking_cols)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---- Step 3: train/test split (same settings as Scenario 1) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# ---- Step 4a: prepare special columns and new features (works row by row, learns nothing) ----
def prepare(data):
    data = data.copy()
    # Agent is an ID, not a quantity: turn it into text, keep missing as missing
    data["Agent"] = data["Agent"].apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    # Feature engineering: new columns built from existing ones
    data["TotalNights"] = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    data["TotalGuests"] = data["Adults"] + data["Children"].fillna(0) + data["Babies"]
    return data

X_train = prepare(X_train)
X_test = prepare(X_test)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print("\nColumns after preparation:", X_train.shape[1])
print("Categorical:", len(cat_cols), cat_cols)
print("Numerical:", len(num_cols), num_cols)

# ---- Step 4b and 5: missing values, encoding, scaling ----
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fills missing numbers with the train median
    ("scale", StandardScaler()),                    # rescales each number column (mean 0, std 1)
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),                        # missing text -> "Missing"
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),           # each category -> its own 0/1 column
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

X_train_prep = preprocess.fit_transform(X_train)   # learns from train only, then transforms it
X_test_prep = preprocess.transform(X_test)         # applies the train values to test

print("\nTrain after preprocessing:", X_train_prep.shape)
print("Test after preprocessing:", X_test_prep.shape)

# ---- Save the prepared data for the baseline files ----
#from scipy import sparse
#def dense(m):
#    return m.toarray() if sparse.issparse(m) else m

#np.savez_compressed(
#    "prep_scenario2.npz",          
#    X_train=dense(X_train_prep),
#    X_test=dense(X_test_prep),
#    y_train=y_train.to_numpy(),
#    y_test=y_test.to_numpy(),
#)
#print("Saved prepared data")



# ---- Step 6: verification checks ----
#from scipy import sparse
#def dense(m):
#    return m.toarray() if sparse.issparse(m) else m

#tr = dense(X_train_prep)
#te = dense(X_test_prep)
#n_num = len(num_cols)

#print("\n--- Verification ---")
#print("Rows train/test:", tr.shape[0], te.shape[0])
#print("Columns train/test:", tr.shape[1], te.shape[1])
#print("NaN train/test:", np.isnan(tr).sum(), np.isnan(te).sum())
#print("Inf train/test:", np.isinf(tr).sum(), np.isinf(te).sum())
#print("Train numeric mean (max abs):", np.abs(tr[:, :n_num].mean(axis=0)).max().round(4))
#print("Train numeric std (min/max):", tr[:, :n_num].std(axis=0).min().round(3),
#      tr[:, :n_num].std(axis=0).max().round(3))
#print("Test numeric mean (max abs):", np.abs(te[:, :n_num].mean(axis=0)).max().round(3))
#print("One-hot values:", np.unique(tr[:, n_num:]))