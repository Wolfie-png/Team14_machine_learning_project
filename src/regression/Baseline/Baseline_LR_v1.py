import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def run_linear_regression(file_name, scenario_name):

    print(f"\n========== {scenario_name} ==========")

    # Load preprocessed data
    data = np.load(file_name)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    # Create Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    return {
        "Scenario": scenario_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# Scenario 1
result1 = run_linear_regression(
    "prep_scenario1.npz",
    "Scenario 1 - All Columns"
)

# Scenario 2
result2 = run_linear_regression(
    "prep_scenario2.npz",
    "Scenario 2 - Booking Time"
)

# Compare results
results_df = pd.DataFrame([result1, result2])

print("\n========== FINAL COMPARISON ==========")
print(results_df.to_string(index=False))
