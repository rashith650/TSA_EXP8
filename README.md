# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date:05/09/2026 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# --- Load Data ---
data = pd.read_csv("index_1.csv")

# Check column names
print("Columns:", data.columns)

# Select 'date' and 'money' columns and rename 'money' to 'Price'
data = data[['date', 'money']].rename(columns={'money': 'Price'})

# Convert Date column to datetime
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

print("Shape of the dataset:", data.shape)
print("First 10 rows:")
print(data.head(10))

# --- Original Data Plot ---
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Original Coffee Price')
plt.title('Original Coffee Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# --- Moving Averages ---
rolling_mean_5 = data['Price'].rolling(window=5).mean()
rolling_mean_10 = data['Price'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='5-Day Moving Average')
plt.plot(rolling_mean_10, label='10-Day Moving Average')
plt.title('Moving Averages of Coffee Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# --- Resample Monthly ---
data_monthly = data.resample('MS').mean()

# --- Scaling ---
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# --- Train-Test Split (80/20) ---
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# --- Holt-Winters Model ---
model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=None
).fit()

test_predictions = model.forecast(steps=len(test_data))

# --- Plot Train vs Test ---
plt.figure(figsize=(12, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Actual Test Data')
test_predictions.plot(label='Predicted')
plt.title('Holt-Winters Forecast (Train vs Test)')
plt.legend()
plt.grid()
plt.show()

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("Root Mean Squared Error (RMSE):", rmse)

# --- Forecast for Future (Next 12 months) ---
future_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal=None
).fit()

future_forecast = future_model.forecast(steps=12)

plt.figure(figsize=(12, 6))
scaled_data.plot(label='Historical Data')
future_forecast.plot(label='Next 12 Months Forecast')
plt.title('Future Forecast of Coffee Price')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="1083" height="335" alt="image" src="https://github.com/user-attachments/assets/08cc3235-5535-4efd-aa8c-cd571eadc4af" />

<img width="1071" height="540" alt="image" src="https://github.com/user-attachments/assets/a85e945c-390a-4385-b50b-7b08a1afa691" />

<img width="1048" height="543" alt="image" src="https://github.com/user-attachments/assets/ee03cdd3-6bb8-4ac7-b0e6-0a6d77162d66" />

<img width="1135" height="587" alt="image" src="https://github.com/user-attachments/assets/94ec28d2-ac47-4efc-b862-d82a7940681b" />

<img width="1102" height="556" alt="image" src="https://github.com/user-attachments/assets/c95a3027-f0c3-45f3-8b6b-b92b156d8131" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
