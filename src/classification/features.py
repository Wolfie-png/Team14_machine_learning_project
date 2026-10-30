
# Target variable
TARGET = "IsCanceled"

# Features provisionally available at booking creation
FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "CustomerType",
]

# Features excluded because of potential data leakage
LEAKAGE_FEATURES = [
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
    "ReservationStatus",
    "ReservationStatusDate",
]

# Decided after checking cancellation rate per value on H2.csv
KEEP_AFTER_REVIEW = [
    "Agent",                     # real signal, known at booking time
]

DROPPED_AFTER_REVIEW = [
    "RequiredCarParkingSpaces",  # 0% cancellations when >=1 space -> leaks outcome
    "ADR",                       # almost no difference between classes
    "DepositType",               # Non Refund cancels 99.8% -> likely set from outcome
]

STILL_UNDECIDED = [
    "TotalOfSpecialRequests",    # strong, but accumulates over time; discuss with teammate
    "Company",                   # not checked yet
]
