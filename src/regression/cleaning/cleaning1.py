# Cleaning version: normalize missing values, remove exact duplicates, drop Company column, and inspect ADR for zero/negative values and outliers.
import pandas as pd

# 1. Load the original dataset
df = pd.read_excel("dataset.csv.xlsx")

original_shape = df.shape

print("Original shape:", original_shape)

# 2. Normalize text and missing-value placeholders
text_cols = df.select_dtypes(
    include=["object", "str", "string"]
).columns

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

df = df.replace(
    ["NULL", "null", "None", "NAN", "nan", ""],
    pd.NA,
)

print("\n--- Missing Values After Normalization ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 3. Investigate duplicates before removing Company
duplicates_before = int(df.duplicated().sum())

print("\n--- Duplicate Check Before Cleaning ---")
print("Exact duplicate occurrences:", duplicates_before)

# Check whether rows become duplicates when Company is excluded.
# This does NOT modify df or delete any rows.
if "Company" in df.columns:
    df_without_company = df.drop(columns=["Company"])

    duplicate_groups_without_company = df_without_company[
        df_without_company.duplicated(keep=False)
    ]

    print("\nDuplicate rows when Company is excluded:",
          len(duplicate_groups_without_company))

    if not duplicate_groups_without_company.empty:
        print("\nRows involved in duplicate groups without Company:")
        print(duplicate_groups_without_company)

# 4. Remove exact duplicate rows using all original columns,
# including Company.
df = df.drop_duplicates().copy()

print("\n--- Duplicate Removal ---")
print("Exact duplicate rows removed:", duplicates_before)
print("Shape after removing duplicates:", df.shape)
print("Remaining exact duplicates:", int(df.duplicated().sum()))

# 5. Drop Company due to excessive missingness
if "Company" in df.columns:
    company_missing_pct = df["Company"].isna().mean() * 100

    print("\nCompany missing percentage:",
          round(company_missing_pct, 2), "%")

    df = df.drop(columns=["Company"])

print("\nShape after dropping Company:", df.shape)

# 6. Inspect remaining missing values
print("\n--- Remaining Missing Values ---")

missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])

print("Total remaining missing cells:",
      int(df.isna().sum().sum()))

# 7. Check target variable ADR
print("\n--- ADR Summary ---")
print(df["ADR"].describe())

print("Missing ADR values:", int(df["ADR"].isna().sum()))
print("Negative ADR values:", int((df["ADR"] < 0).sum()))
print("Zero ADR values:", int((df["ADR"] == 0).sum()))

# 8. Final cleaning summary
print("\n--- Cleaning Summary ---")
print("Original shape:", original_shape)
print("Final shape:", df.shape)
print("Rows removed:", original_shape[0] - df.shape[0])
print("Columns removed:", original_shape[1] - df.shape[1])
print("Remaining exact duplicates:", int(df.duplicated().sum()))
print("Remaining missing cells:", int(df.isna().sum().sum()))

# 9. Save the cleaned dataset
output_path = "dataset_cleaned1.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset to {output_path}")
