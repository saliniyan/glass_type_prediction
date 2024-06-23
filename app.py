from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('rnn_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        input_data = pd.DataFrame([data])
        input_data = input_data.astype(float)

        input_data_reshaped = input_data.values.reshape((input_data.shape[0], input_data.shape[1], 1))
        
        # Make prediction using the loaded model
        raw_predictions = loaded_model.predict(input_data_reshaped)
        
        predicted_class_index = raw_predictions.argmax(axis=1)[0]
        
        predicted_class = predicted_class_index + 1  # Adding 1 to match original class labels
        
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)



"""df1= pd.read_csv("/home/saliniyan/Downloads/Study material/glass.csv")
x_data = df1.drop(['Class'], axis=1)
y_data = df1['Class']

y_data_mapped = y_data - 1

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data_mapped, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Reshape input data for SimpleRNN layer
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = Sequential([
    SimpleRNN(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # Updated to 6 units for 0-5 classes
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
np.random.seed(42)
model.fit(X_train_reshaped, y_train_encoded, epochs=50, batch_size=64, validation_split=0.1)
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_encoded)
print(f"Test Accuracy: {test_accuracy}")
model.save('rnn_model.keras')
"""
