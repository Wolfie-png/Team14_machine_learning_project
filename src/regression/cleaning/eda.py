
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_excel("dataset.csv.xlsx")

# print("\nADR values equal to zero:", (df["ADR"] == 0).sum())
# print("Negative ADR values:", (df["ADR"] < 0).sum())

# q1 = df["ADR"].quantile(0.25)
# q3 = df["ADR"].quantile(0.75)
# iqr = q3 - q1
# upper_bound = q3 + 1.5 * iqr
# lower_bound = q1 - 1.5 * iqr

# print("\nIQR lower bound:", lower_bound)
# print("IQR upper bound:", upper_bound)
# print("Values above upper bound:", (df["ADR"] > upper_bound).sum())
# print("Values below lower bound:", (df["ADR"] < lower_bound).sum())

# print("\nRows with the 10 highest ADR values:")
# print(
#     df.nlargest(10, "ADR")[
#         ["ADR", "Adults", "Children", "Babies",
#          "StaysInWeekendNights", "StaysInWeekNights",
#          "ReservedRoomType", "AssignedRoomType"]
#     ]
# )

# # Plot ADR distribution
# plt.figure(figsize=(10, 5))
# sns.histplot(df["ADR"], bins=60)
# plt.title("Distribution of ADR")
# plt.xlabel("ADR")
# plt.ylabel("Number of bookings")
# plt.tight_layout()
# plt.show()

# # Plot ADR boxplot
# plt.figure(figsize=(10, 3))
# sns.boxplot(x=df["ADR"])
# plt.title("ADR Boxplot")
# plt.xlabel("ADR")
# plt.tight_layout()
# plt.show()

# # correlations between numerical variables and ADR
# numeric_df = df.select_dtypes(include="number")
# correlations = numeric_df.corr()["ADR"].drop("ADR").sort_values(
#     ascending=False
# )

# print("\nNumerical correlations with ADR:")
# print(correlations)

# #  exact duplicate examples
# duplicates = df[df.duplicated(keep=False)]

# print("\nRows involved in exact duplication:", len(duplicates))
# print("\nFirst 10 rows involved in duplication:")
# print(duplicates.head(10).to_string(index=False))

# # unusual values in selected columns
# columns_to_check = [
#     "Adults",
#     "Children",
#     "Babies",
#     "StaysInWeekendNights",
#     "StaysInWeekNights",
#     "LeadTime",
#     "RequiredCarParkingSpaces",
# ]

# print("\nMinimum values of selected columns:")
# print(df[columns_to_check].min())

# print("\nZero counts in selected columns:")
# print((df[columns_to_check] == 0).sum())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA AND NORMALIZE MISSING VALUES
# ============================================================

df = pd.read_excel("dataset.csv.xlsx")

# Remove leading/trailing spaces from text columns
text_cols = df.select_dtypes(
    include=["object", "str", "string"]
).columns

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Convert null-like strings to actual missing values
df = df.replace(
    ["NULL", "null", "None", "NAN", "nan", ""],
    pd.NA,
)

print("Dataset shape:", df.shape)
print("Missing ADR values:", df["ADR"].isna().sum())

# ============================================================
# 2. ADR ZERO, NEGATIVE VALUES, AND OUTLIERS
# ============================================================

print("\n--- ADR Checks ---")

print("ADR values equal to zero:", (df["ADR"] == 0).sum())
print("Negative ADR values:", (df["ADR"] < 0).sum())

q1 = df["ADR"].quantile(0.25)
q3 = df["ADR"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("\nIQR lower bound:", lower_bound)
print("IQR upper bound:", upper_bound)
print("Values below lower bound:", (df["ADR"] < lower_bound).sum())
print("Values above upper bound:", (df["ADR"] > upper_bound).sum())

print("\nADR percentiles:")
print(df["ADR"].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

print("\nRows with the 10 highest ADR values:")
print(
    df.nlargest(10, "ADR")[
        ["ADR", "Adults", "Children", "Babies",
         "StaysInWeekendNights", "StaysInWeekNights",
         "ReservedRoomType", "AssignedRoomType",
         "IsCanceled", "MarketSegment", "ReservationStatus"]
    ].to_string(index=False)
)

# ============================================================
# 3. ADR DISTRIBUTION
# ============================================================

# Full distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="ADR", bins=60)
plt.title("Distribution of ADR (Full Range)")
plt.xlabel("ADR")
plt.ylabel("Number of bookings")
plt.tight_layout()
plt.show()

# Distribution zoomed to the 99th percentile
# This makes the main distribution easier to examine
adr_99 = df["ADR"].quantile(0.99)

plt.figure(figsize=(10, 5))
sns.histplot(
    data=df[df["ADR"] <= adr_99],
    x="ADR",
    bins=60
)
plt.title("Distribution of ADR (Up to 99th Percentile)")
plt.xlabel("ADR")
plt.ylabel("Number of bookings")
plt.tight_layout()
plt.show()

# ADR boxplot
plt.figure(figsize=(10, 3))
sns.boxplot(x=df["ADR"])
plt.title("ADR Boxplot (Full Range)")
plt.xlabel("ADR")
plt.tight_layout()
plt.show()

# ============================================================
# 4. CORRELATIONS WITH ADR
# ============================================================

numeric_df = df.select_dtypes(include="number")

if "ADR" in numeric_df.columns:
    correlations = (
        numeric_df.corr()["ADR"]
        .drop("ADR")
        .sort_values(ascending=False)
    )

    print("\n--- Numerical Correlations with ADR ---")
    print(correlations)

    plt.figure(figsize=(9, 6))
    correlations.sort_values().plot(kind="barh")
    plt.title("Correlation of Numerical Variables with ADR")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Numerical variable")
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. EXACT DUPLICATES
# ============================================================

print("\n--- Exact Duplicates ---")

print("Duplicate occurrences:", int(df.duplicated().sum()))

duplicates = df[df.duplicated(keep=False)]

print("Rows involved in duplicate groups:", len(duplicates))
print("\nFirst 10 rows involved in duplication:")

print(duplicates.head(10).to_string(index=False))

# ============================================================
# 6. UNUSUAL VALUES IN SELECTED COLUMNS
# ============================================================

columns_to_check = [
    "Adults",
    "Children",
    "Babies",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "LeadTime",
    "RequiredCarParkingSpaces",
]

print("\n--- Minimum Values ---")
print(df[columns_to_check].min())

print("\n--- Zero Counts ---")
print((df[columns_to_check] == 0).sum())

print("\n--- Negative Counts ---")
print((df[columns_to_check] < 0).sum())

# Investigate bookings with zero guests
zero_guests = (
    (df["Adults"] == 0)
    & (df["Children"] == 0)
    & (df["Babies"] == 0)
)

print(
    "\nBookings with zero adults, children and babies:",
    int(zero_guests.sum())
)

# ============================================================
# 7. FINAL EDA SUMMARY
# ============================================================

print("\n--- Final EDA Summary ---")
print("Dataset shape:", df.shape)
print("Total missing cells:", int(df.isna().sum().sum()))
print("\nColumns with missing values:")
print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))
