from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open('classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    feature_values = [float(request.form[feature]) for feature in ['number_project', 'average_monthly_hours', 'promotion_last_5years', 'dept', 'salary', 'company']]
    input_data = np.array([feature_values])

   
    prediction = clf.predict(input_data)
    
    
    satisfaction_status = "low" if prediction <= 0.5 else "okay"
    
    return render_template('index.html', prediction_text=f'Satisfaction of employee is {satisfaction_status}: {prediction[0]:.2f}')

@app.route('/company_info')
def company_info():
    return render_template('company_info.html')

if __name__ == "__main__":
    app.run(debug=True)
