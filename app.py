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
        print(input_data_reshaped)
        raw_predictions = loaded_model.predict(input_data_reshaped)
        print(raw_predictions)
        predicted_class = raw_predictions.argmax(axis=1)[0]
    
        glass_classes = [
        "Building Windows (Float Processed)", "Used for windows in buildings",
        "Building Windows (Non-Float Processed)", "Used for windows in buildings",
        "Vehicle Windows (Float Processed)", "Used for windows in vehicles",
        "Vehicle Windows (Non-Float Processed)", "Used for windows in vehicles",
        "Containers", "Used for making containers",
        "Tableware", "Used for making tableware",
        "Headlamps", "Used for making headlamps",
        ]
        use_case= glass_classes[predicted_class]
        
        return render_template('result.html', predicted_class=predicted_class, use_case=use_case)

if __name__ == '__main__':
    app.run(debug=True)
