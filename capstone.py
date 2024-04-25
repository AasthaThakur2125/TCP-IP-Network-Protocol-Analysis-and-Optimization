import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical

# Load the dataset from CSV
df = pd.read_csv(r'C:\Desktop\capstonenetworkdata.csv')  # Adjust the file path if needed

# Define the target variable based on a condition (e.g., delay exceeding 50)
df['congestion'] = (df['delay'] > 50).astype(int)

# Preprocess the data
features = ['packet_size', 'arrival_time', 'delay', 'jitter']
X = df[features]
y = df['congestion']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM (samples, timesteps, features)
timesteps = 1  # We are using one timestep per sample
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))

# Convert y to categorical using to_categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Build and compile the LSTM model
model = Sequential()
model.add(Input(shape=(timesteps, X_train_scaled.shape[1])))
model.add(LSTM(50))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train_categorical, epochs=100, batch_size=64, validation_data=(X_test_reshaped, y_test_categorical), verbose=2)