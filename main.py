import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data from Excel
file_path = "c-data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure your data has columns: Month, Year, Sales, Manufacture
# Example structure:
#    Month  Year  Sales  Manufacture
# 0      1  2020    500          300
# 1      2  2020    520          320
# ...

# Create a datetime column and sort data
# data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))  # Create Date column
# data = data.sort_values('Date')  # Sort by Date
# data['TimeIndex'] = np.arange(len(data))  # Create a sequential time index

# Parse the Date column into a datetime format
data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime
data = data.sort_values('Date')  # Sort by date

# Create a sequential time index
data['TimeIndex'] = np.arange(len(data))  # Sequential index for time

# Prepare features (X) and targets (y)
X = data[['TimeIndex']]  # Predictor: sequential time index
y = data[['Close', 'Volume']]  # Targets: Sales and Manufacture

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict future values
future_periods = 12  # Predict for the next 12 months
last_time_index = data['TimeIndex'].iloc[-1]
future_time_indexes = np.arange(last_time_index + 1, last_time_index + 1 + future_periods).reshape(-1, 1)
future_predictions = model.predict(future_time_indexes)

# # Create a DataFrame for future predictions
# future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=future_periods, freq='MS')

# Create future dates
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.offsets.MonthBegin(1), periods=future_periods, freq='MS')
future_df = pd.DataFrame({
    'Date': future_dates,
    'PredictedClose': future_predictions[:, 0],
    'PredictedVolume': future_predictions[:, 1]
})

# Visualize the results
plt.figure(figsize=(14, 8))
plt.plot(data['Date'], data['Close'], label='Actual Close', marker='o')
plt.plot(data['Date'], data['Volume'], label='Actual Volume', marker='o')
plt.plot(future_df['Date'], future_df['PredictedClose'], label='Predicted Close', linestyle='--', marker='o', color='red')
plt.plot(future_df['Date'], future_df['PredictedVolume'], label='Predicted Volume', linestyle='--', marker='o', color='green')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Close and Volume Predictions')
plt.legend()
plt.grid()
plt.show()

# Save predictions to an Excel file
future_df.to_excel("future_sales_manufacture_predictions.xlsx", index=False)
print("Predictions saved to 'future_sales_manufacture_predictions.xlsx'")
