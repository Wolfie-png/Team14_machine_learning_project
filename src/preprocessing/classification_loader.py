
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"


def load_data():
    """Load the H2 City Hotel dataset."""

    df = pd.read_csv(DATA_PATH)

    return df


def inspect_data(df):
    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())
    
    
    

def inspect_data_quality(df):

    print("\n===== DUPLICATE ROWS =====")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n===== INVALID GUEST COUNTS =====")

    zero_guests = (
        df["Adults"].fillna(0)
        + df["Children"].fillna(0)
        + df["Babies"].fillna(0)
    ) == 0

    print("Bookings with zero guests:", zero_guests.sum())

    guest_columns = ["Adults", "Children", "Babies"]

    negative_guests = (df[guest_columns] < 0).any(axis=1)

    print("Bookings with negative guests:", negative_guests.sum())

    print("\n===== INVALID STAYS =====")

    zero_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    ) == 0

    print("Bookings with zero nights:", zero_nights.sum())

    print("\n===== ADR CHECK =====")

    print("Negative ADR:", (df["ADR"] < 0).sum())
    print("Zero ADR:", (df["ADR"] == 0).sum())
    print("Maximum ADR:", df["ADR"].max())
    print("Minimum ADR:", df["ADR"].min())

    print("\n===== CATEGORICAL PLACEHOLDERS =====")

    for column in ["Agent", "Company"]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(10))



def inspect_duplicates(df):

    print("\n===== DUPLICATE INVESTIGATION =====")

    
    print("Unique rows:", len(df.drop_duplicates()))

    
    print("Duplicate occurrences:", df.duplicated().sum())

    # Compare cancellation rates
    duplicate_mask = df.duplicated(keep=False)

    duplicate_rows = df[duplicate_mask]
    non_duplicate_rows = df[~duplicate_mask]

    print("\nRows belonging to duplicate groups:",
          len(duplicate_rows))

    print("\nCancellation rate in duplicate groups:",
          duplicate_rows["IsCanceled"].mean())

    print("Cancellation rate in non-duplicate groups:",
          non_duplicate_rows["IsCanceled"].mean())

    # Check whether identical predictor values
    # have different target labels
    predictor_columns = [
        col for col in df.columns
        if col not in [
            "IsCanceled",
            "ReservationStatus",
            "ReservationStatusDate"
        ]
    ]

    conflicting_groups = (
        df.groupby(
            predictor_columns,
            dropna=False
        )["IsCanceled"]
        .nunique()
    )

    print("\nPredictor groups with conflicting labels:",
          (conflicting_groups > 1).sum())




def inspect_class_balance(df):

    print("\n===== CLASS BALANCE =====")

    # Number of bookings in each class
    counts = df["IsCanceled"].value_counts().sort_index()

    print("\nBooking counts:")
    print(counts)

    # Percentage of bookings in each class
    percentages = (
        df["IsCanceled"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print("\nBooking percentages:")
    print(percentages.round(2))

    # Check for unexpected target values
    print("\nUnique target values:")
    print(df["IsCanceled"].unique())



def analyze_lead_time(df):

    print("\n===== LEAD TIME ANALYSIS =====")

    # Define lead-time categories
    bins = [-1, 7, 30, 90, 180, float("inf")]

    labels = [
        "0-7 days",
        "8-30 days",
        "31-90 days",
        "91-180 days",
        "181+ days"
    ]

    # Create a temporary copy
    analysis_df = df.copy()

    # Assign each booking to a lead-time group
    analysis_df["LeadTimeGroup"] = pd.cut(
        analysis_df["LeadTime"],
        bins=bins,
        labels=labels
    )

    # Calculate bookings and cancellation rate
    results = analysis_df.groupby(
        "LeadTimeGroup",
        observed=True
    )["IsCanceled"].agg(
        BookingCount="count",
        CancellationRate="mean"
    )

    # Convert rate to percentage
    results["CancellationRate"] *= 100

    print(results.round(2))

    return results



def analyze_deposit_type(df):

    print("\n===== DEPOSIT TYPE ANALYSIS =====")

    results = df.groupby("DepositType")["IsCanceled"].agg(
        BookingCount="count",
        CancellationRate="mean"
    )

    results["CancellationRate"] *= 100

    # Sort by cancellation rate, highest first
    results = results.sort_values(
        by="CancellationRate",
        ascending=False
    )

    print(results.round(2))

    return results


def analyze_market_segment(df):

    print("\n===== MARKET SEGMENT ANALYSIS =====")

    results = df.groupby("MarketSegment")["IsCanceled"].agg(
        BookingCount="count",
        CancellationRate="mean"
    )

    results["CancellationRate"] *= 100

    results = results.sort_values(
        by="CancellationRate",
        ascending=False
    )

    print(results.round(2))

    return results


def inspect_feature_list(df):

    print("\n===== FEATURE LIST =====")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")


if __name__ == "__main__":

    df = load_data()

   # print("Dataset loaded successfully!")

   # print("Dataset shape:", df.shape)

   # print("\nFirst 5 rows:")
    #print(df.head())
    #inspect_data(df)
    #inspect_data_quality(df)
    #inspect_duplicates(df)
    #inspect_class_balance(df)
    #analyze_lead_time(df)
    #analyze_deposit_type(df)
    #analyze_market_segment(df)
    inspect_feature_list(df)
    



