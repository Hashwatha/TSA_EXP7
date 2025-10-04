# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



## AIM:
To Implementat an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Load dataset
file_path = 'cardekho.csv'
df = pd.read_csv(file_path)

print("Dataset Columns:", df.columns)
print(df.head())

df['sale_date'] = pd.to_datetime(df['year'], format='%Y')

df.set_index('sale_date', inplace=True)

default_column_name = 'selling_price' if 'selling_price' in df.columns else df.columns[0]
series = df[default_column_name]
print(f"Using column '{default_column_name}' for time series.")

adf_result = adfuller(series.dropna())
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(series.dropna(), lags=30, ax=plt.gca())
plt.subplot(122)
plot_pacf(series.dropna(), lags=30, ax=plt.gca())
plt.show()

model = AutoReg(train, lags=5)  
model_fitted = model.fit()

predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.title('AR Model - Actual vs Predicted')
plt.show()

mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse}")

```
## OUTPUT:

<img width="1032" height="586" alt="image" src="https://github.com/user-attachments/assets/6b0b4393-20a9-4a65-be60-fd908b8cde57" />

<img width="1263" height="651" alt="image" src="https://github.com/user-attachments/assets/58fe3f60-f771-4d2a-9ac3-790db1e1a203" />

<img width="1170" height="668" alt="image" src="https://github.com/user-attachments/assets/fc3c8a01-58ef-4d1d-b6d9-fc0addd3e090" />

<img width="542" height="52" alt="image" src="https://github.com/user-attachments/assets/93807916-451f-4614-88ae-f98eb65f1de3" />

## RESULT:
Thus we have successfully implemented the auto regression function using python.
