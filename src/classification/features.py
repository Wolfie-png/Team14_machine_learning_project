
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

# Features requiring further verification
REVIEW_FEATURES = [
    "DepositType",
    "Agent",
    "Company",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]
