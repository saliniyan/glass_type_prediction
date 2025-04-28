import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import joblib

# Load and preprocess the dataset
df1 = pd.read_csv("glass.csv")
df1 = df1.drop(columns=['Index'])
x_data = df1.drop(['Class'], axis=1)
y_data = df1['Class']

label_encoder = LabelEncoder()
y_data_mapped = label_encoder.fit_transform(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data_mapped, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    SimpleRNN(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_reshaped, y_train, epochs=70, batch_size=16, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)

y_train_pred = model.predict(X_train_reshaped)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

train_accuracy = accuracy_score(y_train, y_train_pred_classes)
print(f"Training Accuracy: {train_accuracy}")

model.save('rnn_model.keras')
scaler = StandardScaler()
scaler.fit(x_data)
joblib.dump(scaler, 'scaler.joblib')