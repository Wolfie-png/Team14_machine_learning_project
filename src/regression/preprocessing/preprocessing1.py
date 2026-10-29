# PREPROCESSING 1 - Scenario 1 (all columns): keeps every column except ADR, including the post-booking columns.
# Steps: split train/test -> prepare Agent and date -> impute missing values -> one-hot encode and scale
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Step 0: load the cleaned data and verify the starting point ----
df = pd.read_csv("dataset_cleaned1.csv")
print("Loaded shape:", df.shape)

#Counts rows that repeat an earlier row
print("Remaining duplicates:", df.duplicated().sum())

#Counts missing cells per column, then shows only the columns that have any
missing = df.isna().sum()
print(missing[missing > 0])

#Counts rows where the price is 0 (expected 1077) and shows the highest price
print("Zero ADR:", (df["ADR"] == 0).sum())
print("Max ADR:", df["ADR"].max())

#features and target
y = df["ADR"]
X = df.drop(columns=["ADR"])

print("X shape:", X.shape)
print("y shape:", y.shape)

#Randomly splits the rows: 80% for training and 20% for testing. It returns four pieces: training features, test features, training prices, test prices.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# ---- Step 4a: prepare special columns and new features (works row by row, learns nothing) ----
def prepare(data):
    data = data.copy()
    # Agent is an ID, not a quantity: turn it into text, keep missing as missing
    data["Agent"] = data["Agent"].apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    # Split the date into numbers the model can use
    d = pd.to_datetime(data["ReservationStatusDate"])
    data["StatusYear"] = d.dt.year
    data["StatusMonth"] = d.dt.month
    data["StatusDay"] = d.dt.day
    # Feature engineering: new columns built from existing ones
    data["TotalNights"] = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    data["TotalGuests"] = data["Adults"] + data["Children"].fillna(0) + data["Babies"]
    return data.drop(columns=["ReservationStatusDate"])

X_train = prepare(X_train)
X_test = prepare(X_test)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print("\nColumns after preparation:", X_train.shape[1])
print("\nCategorical:", cat_cols)
print("Numerical:", num_cols)

# ---- Step 4b and 5: missing values, encoding, scaling ----
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()), #rescales each number column to a similar range,
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")), # missing text -> "Missing"
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)), #turns each category into its own 0/1 column
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

X_train_prep = preprocess.fit_transform(X_train)   # learns each column's mean and std from train, then scales
X_test_prep = preprocess.transform(X_test)         # applies what was learned, scales test using the train values

print("\nTrain after preprocessing:", X_train_prep.shape)
print("Test after preprocessing:", X_test_prep.shape)


# ---- Save the prepared data for the baseline files ----
#from scipy import sparse
#def dense(m):
#    return m.toarray() if sparse.issparse(m) else m

#np.savez_compressed(
#    "prep_scenario1.npz",          # use "prep_scenario2.npz" in preprocessing2.py
#    X_train=dense(X_train_prep),
#    X_test=dense(X_test_prep),
#    y_train=y_train.to_numpy(),
#    y_test=y_test.to_numpy(),
#)
#print("Saved prepared data")



#The checks to confirm the preprocessing 
#from scipy import sparse

#def dense(m):
#    return m.toarray() if sparse.issparse(m) else m

#tr = dense(X_train_prep)
#te = dense(X_test_prep)
#n_num = len(num_cols)

# 1. Row counts unchanged
#print("Rows train/test:", tr.shape[0], te.shape[0])

# 2. Same number of columns in train and test
#print("Columns train/test:", tr.shape[1], te.shape[1])

# 3. No missing or infinite values left
#print("NaN train/test:", np.isnan(tr).sum(), np.isnan(te).sum())
#print("Inf train/test:", np.isinf(tr).sum(), np.isinf(te).sum())

# 4. Scaled numeric columns: train mean ~0 and std ~1
#print("Train numeric mean (max abs):", np.abs(tr[:, :n_num].mean(axis=0)).max().round(4))
#print("Train numeric std (min/max):", tr[:, :n_num].std(axis=0).min().round(3),
#      tr[:, :n_num].std(axis=0).max().round(3))
#print("Test numeric mean (max abs):", np.abs(te[:, :n_num].mean(axis=0)).max().round(3))

# 5. One-hot columns contain only 0 and 1
#print("One-hot values:", np.unique(tr[:, n_num:]))