import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


file_name = "Walmart.csv"  
data = pd.read_csv(file_name)


print("\nDataset - First Few Rows:")
print(data.head())


data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)


data_grouped = data.groupby('Date')['Weekly_Sales'].sum().reset_index()

data_grouped = data_grouped.sort_values('Date')



print("\nDataset Information:")
print(data_grouped.info())


print("\nMissing Values:")
print(data_grouped.isnull().sum())


plt.figure(figsize=(12, 6))
plt.plot(data_grouped['Date'], data_grouped['Weekly_Sales'], label='Weekly Sales')
plt.title("Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()


train = data_grouped.iloc[:-10] 
test = data_grouped.iloc[-10:]  


model = ExponentialSmoothing(
    train['Weekly_Sales'], 
    seasonal='add', 
    seasonal_periods=52, 
    trend='add'
).fit()

test['Forecast'] = model.forecast(len(test))


plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Weekly_Sales'], label='Train')
plt.plot(test['Date'], test['Weekly_Sales'], label='Test', color='orange')
plt.plot(test['Date'], test['Forecast'], label='Forecast', color='green')
plt.title("Sales Forecasting with Holt-Winters")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

mse = mean_squared_error(test['Weekly_Sales'], test['Forecast'])
print(f"\nMean Squared Error (MSE): {mse:.2f}")


test.to_csv("sales_forecast.csv", index=False)
print("\nForecast saved as 'sales_forecast.csv'.")
