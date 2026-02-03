from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__, static_folder='statics')

# URL of your Flask API for making predictions
api_url = 'http://localhost:90/predict'  

# Load the TensorFlow model
model = tf.keras.models.load_model('my_model.keras')  # Replace 'my_model.keras' with the actual model file
class_labels = ['Class 0', 'Class 1', 'Class 2']

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except:
    scaler = None 

@app.route('/')
def home():
    return "Welcome to the Wine Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            # sepal_length = float(data['sepal_length'])
            # sepal_width = float(data['sepal_width'])
            # petal_length = float(data['petal_length'])
            # petal_width = float(data['petal_width'])

            # # Perform the prediction
            # input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]
            # prediction = model.predict(input_data)
            # predicted_class = class_labels[np.argmax(prediction)]

            # # Return the predicted class in the response
            # # Use jsonify() instead of json.dumps() in Flask
            # return jsonify({"predicted_class": predicted_class})

            alcohol = float(data['alcohol'])
            malic_acid = float(data['malic_acid'])
            ash = float(data['ash'])
            alcalinity_of_ash = float(data['alcalinity_of_ash'])
            magnesium = float(data['magnesium'])
            total_phenols = float(data['total_phenols'])
            flavanoids = float(data['flavanoids'])
            nonflavanoid_phenols = float(data['nonflavanoid_phenols'])
            proanthocyanins = float(data['proanthocyanins'])
            color_intensity = float(data['color_intensity'])
            hue = float(data['hue'])
            od280_od315 = float(data['od280_od315'])
            proline = float(data['proline'])

            # Perform the prediction
            input_data = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                                   total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                                   color_intensity, hue, od280_od315, proline]])
            
            if scaler:
                input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            # Return the predicted class in the response
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": f"{confidence * 100:.2f}%"
            })
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4060)
