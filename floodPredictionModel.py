import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load historical water level monitoring data
data = pd.read_csv('water_level_data.csv')

# Preprocess the data
# Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])
# Extract features (e.g., month, day, hour) from the date
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

# Split data into training and testing sets
X = data[['month', 'day', 'hour']]
y = data['water_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Predict water level for future dates
future_dates = pd.date_range(start='2024-05-11', end='2024-05-20', freq='H')
future_data = pd.DataFrame({'date': future_dates})
future_data['month'] = future_data['date'].dt.month
future_data['day'] = future_data['date'].dt.day
future_data['hour'] = future_data['date'].dt.hour
future_predictions = model.predict(future_data[['month', 'day', 'hour']])
print("Predicted water levels for future dates:")
print(future_predictions)
