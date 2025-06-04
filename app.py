from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
std = pickle.load(open('standardscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))     

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract form inputs
        Bedrooms = float(request.form['Bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        sqft_lot = float(request.form['sqft_lot'])
        floors = float(request.form['floors'])
        waterfront = float(request.form['waterfront'])
        view = float(request.form['view'])
        condition = float(request.form['condition'])
        grade = float(request.form['grade'])
        sqft_living15 = float(request.form['sqft_living15'])
        sqft_lot15 = float(request.form['sqft_lot15'])

        # Combine features into array
        features = [Bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                    waterfront, view, condition, grade, sqft_living15, sqft_lot15]
        single_pred = np.array(features).reshape(1, -1)

        # Apply preprocessing (if any)
        if mx:
            single_pred = mx.transform(single_pred)
        if std:
            single_pred = std.transform(single_pred)

        # Predict
        prediction = model.predict(single_pred)[0]
        result = f"🏠 Predicted House Price: ${prediction:,.2f}"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
