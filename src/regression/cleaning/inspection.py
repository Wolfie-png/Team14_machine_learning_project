
import pandas as pd

# ============================================================
# 1. LOAD DATASET AND NORMALIZE MISSING VALUES
# ============================================================

df = pd.read_excel("dataset.csv.xlsx")

original_shape = df.shape

print("Original shape:", original_shape)

# Strip whitespace from text columns
text_cols = df.select_dtypes(
    include=["object", "str", "string"]
).columns

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Convert null-like strings into actual missing values
df = df.replace(
    ["NULL", "null", "None", "NAN", "nan", ""],
    pd.NA,
)

print("\n--- Dataset Overview ---")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n--- Missing Values ---")
missing = df.isna().sum().sort_values(ascending=False)
print(missing)

print("\nTotal missing cells:", int(df.isna().sum().sum()))

print("\nMissing ADR values:", df["ADR"].isna().sum())

# Verify Agent and Company
for col in ["Agent", "Company"]:
    print(f"\n{col}:")
    print("Missing values:", df[col].isna().sum())
    print("Non-missing values:", df[col].notna().sum())
    print("Missing percentage:", round(df[col].isna().mean() * 100, 2), "%")
    print("Most common values:")
    print(df[col].value_counts(dropna=False).head(15))


# ============================================================
# 3. EXACT DUPLICATES
# ============================================================

print("\n--- Exact Duplicates ---")
print("Original rows:", len(df))
print("Duplicate occurrences:", int(df.duplicated().sum()))
print("Rows after dropping exact duplicates:",
      len(df.drop_duplicates()))


# ============================================================
# 4. NUMERICAL SUMMARY
# ============================================================

print("\n--- Numerical Summary ---")
print(df.describe())

print("\n--- ADR Summary ---")
print(df["ADR"].describe())

print("\nADR percentiles:")
print(
    df["ADR"].quantile(
        [0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    )
)


# ============================================================
# 5. NEGATIVE VALUES IN NONNEGATIVE FIELDS
# ============================================================

print("\n--- Negative Values ---")

nonnegative_cols = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ADR",
]

for col in nonnegative_cols:
    count = (df[col] < 0).sum()
    if count:
        print(f"{col}: {count}")

print("Zero ADR values:", (df["ADR"] == 0).sum())
print("Negative ADR values:", (df["ADR"] < 0).sum())


# ============================================================
# 6. BINARY VARIABLES
# ============================================================

print("\n--- Binary Variables ---")

for col in ["IsCanceled", "IsRepeatedGuest"]:
    print(f"\n{col} values:")
    print(df[col].value_counts(dropna=False))


# ============================================================
# 7. ARRIVAL DATE COMPONENTS
# ============================================================

print("\n--- Arrival Dates ---")

print("\nArrival years:")
print(df["ArrivalDateYear"].value_counts().sort_index())

print("\nArrival months:")
print(df["ArrivalDateMonth"].value_counts())

print("\nArrival day range:")
print(
    df["ArrivalDateDayOfMonth"].min(),
    df["ArrivalDateDayOfMonth"].max()
)


# ============================================================
# 8. GUEST COUNTS
# ============================================================

print("\n--- Guest Counts ---")

zero_guests = (
    (df["Adults"] == 0)
    & (df["Children"] == 0)
    & (df["Babies"] == 0)
)

print(
    "Bookings with zero adults, children and babies:",
    int(zero_guests.sum())
)

zero_adults = df[df["Adults"] == 0]

print("\nBookings with zero adults:", len(zero_adults))
print("\nZero-adult bookings by ADR:")
print(zero_adults["ADR"].describe())

print("\nExamples of zero-adult bookings:")
print(
    zero_adults[[
        "Adults", "Children", "Babies", "ADR",
        "StaysInWeekendNights", "StaysInWeekNights",
        "IsCanceled", "CustomerType", "ReservationStatus"
    ]].head(15).to_string(index=False)
)


# ============================================================
# 9. ADR OUTLIERS AND UNUSUAL VALUES
# ============================================================

print("\n--- ADR Investigation ---")

q1 = df["ADR"].quantile(0.25)
q3 = df["ADR"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("IQR lower bound:", lower_bound)
print("IQR upper bound:", upper_bound)
print("ADR below lower bound:", int((df["ADR"] < lower_bound).sum()))
print("ADR above upper bound:", int((df["ADR"] > upper_bound).sum()))

if df["ADR"].notna().any():
    print("\nRecord with the highest ADR:")
    print(df.loc[df["ADR"].idxmax()].to_string())

print("\nExamples of zero-ADR bookings:")
print(
    df.loc[df["ADR"] == 0, [
        "ADR", "Adults", "Children", "Babies",
        "LeadTime", "StaysInWeekendNights",
        "StaysInWeekNights", "CustomerType",
        "MarketSegment", "ReservedRoomType",
        "ReservationStatus"
    ]].head(10).to_string(index=False)
)

print("\nBookings with ADR below 9:")
print("Count:", int((df["ADR"] < 9).sum()))

print(
    df.loc[df["ADR"] < 9, [
        "ADR", "Adults", "Children", "Babies",
        "CustomerType", "MarketSegment", "ReservationStatus"
    ]].head(15).to_string(index=False)
)


# ============================================================
# 10. CATEGORICAL VARIABLES
# ============================================================

print("\n--- Categorical Variables ---")

categorical_cols = [
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "CustomerType",
    "ReservationStatus",
]

for col in categorical_cols:
    print(f"\n{col} categories:")
    print(df[col].value_counts(dropna=False).head(20))


# ============================================================
# 11. RESERVATION STATUS DATE
# ============================================================

print("\n--- Reservation Status Date ---")

parsed_dates = pd.to_datetime(
    df["ReservationStatusDate"],
    errors="coerce"
)

print(
    "Unparseable or missing ReservationStatusDate values:",
    int(parsed_dates.isna().sum())
)


# ============================================================
# 12. CLEANING DECISION SUMMARY
# ============================================================

print("\n--- Cleaning Decision Summary ---")

print("Starting rows:", original_shape[0])
print("Current rows:", len(df))
print("Rows removed so far:", original_shape[0] - len(df))
print("Current columns:", len(df.columns))
print("Remaining exact duplicates:", int(df.duplicated().sum()))
print("Remaining missing cells:", int(df.isna().sum().sum()))

print("\nColumns containing missing values:")
print(
    df.isna().sum()[df.isna().sum() > 0].sort_values(
        ascending=False
    )
)
