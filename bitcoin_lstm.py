# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('btc_dataset.csv')
print("Initial dataset:")
print(df.head())

# Sort by date and set 'Date' as index
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Use only 'Close' price for prediction
data = df[['Close']].values

# Normalize data using MinMaxScaler for better convergence
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create dataset for LSTM input
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Split data into train and test sets
train_size = int(len(scaled_data) * 0.70)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create input-output pairs
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape to [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predicted vs actual
plt.figure(figsize=(14,5))
plt.plot(y_test_unscaled, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title("Bitcoin Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Print RMSE for evaluation
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
print("Root Mean Squared Error:", rmse)
