from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys

sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

# GIAI ĐOẠN 1: TIỀN XỬ LÝ DATASET
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop('loan_id', axis=1)
    data.columns = data.columns.str.strip()
    data['self_employed'] = data['self_employed'].str.strip()
    data['education'] = data['education'].str.strip()
    data['loan_status'] = data['loan_status'].str.strip()
    data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
    data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
    data.insert(0, 'bias', 1)
    
    if data.isna().sum().sum() > 0:
        raise ValueError("Dữ liệu chứa giá trị NaN")

    scaler = MinMaxScaler()
    feature_columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 
                        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
                        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    return data, scaler

def train_model(data):
    features_data = data[['bias', 'no_of_dependents', 'education', 'self_employed', 
                          'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                          'residential_assets_value', 'commercial_assets_value', 
                          'luxury_assets_value', 'bank_asset_value']]
    lables_data = data['loan_status']
    
    features_training, features_testing, lables_training, lables_testing = train_test_split(
        features_data, lables_data, test_size=0.1, random_state=42
    )
    
    def sigmoidf(z):
        return 1 / (1 + np.exp(-z))

    def predictf(features, weights):
        return sigmoidf(np.dot(features, weights))

    def log_lossf(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = len(y_true)
        return - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient_descent(features, labels, weights, learning_rate, iterations):
        m = len(labels)
        for i in range(iterations):
            y_pred = predictf(features, weights)
            gradient = np.dot(features.T, (y_pred - labels)) / m
            weights -= learning_rate * gradient
            if i % 100 == 0:
                loss = log_lossf(labels, y_pred)
                print(f"Iteration {i}: Loss = {loss}")
        return weights

    weights = np.zeros(features_training.shape[1])
    learning_rate = 0.001
    iterations = 1000
    weights = gradient_descent(features_training, lables_training, weights, learning_rate, iterations)
    
    y_pred_testing = predictf(features_testing, weights)
    y_pred_testing = [1 if i > 0.5 else 0 for i in y_pred_testing]

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    accuracy_score = accuracy(lables_testing, y_pred_testing)
    print(f"Accuracy: {accuracy_score * 100}%")

    return weights, predictf

file_path = "C:/Users/lebin/Downloads/Loan_Approval/Training/loan_approval_dataset.csv"
data, scaler = preprocess_data(file_path)
model_weights, predictf = train_model(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'no_of_dependents': float(request.form['no_of_dependents']),
            'education': int(request.form['education']),
            'self_employed': int(request.form['self_employed']),
            'income_annum': float(request.form['income_annum'].replace(',', '')),
            'loan_amount': float(request.form['loan_amount'].replace(',', '')),
            'loan_term': int(request.form['loan_term']),
            'cibil_score': float(request.form['cibil_score']),
            'residential_assets_value': float(request.form['residential_assets_value'].replace(',', '')),
            'commercial_assets_value': float(request.form['commercial_assets_value'].replace(',', '')),
            'luxury_assets_value': float(request.form['luxury_assets_value'].replace(',', '')),
            'bank_asset_value': float(request.form['bank_asset_value'].replace(',', ''))
        }
        input_data = pd.DataFrame([data])
        input_data = scaler.transform(input_data)
        input_data = np.insert(input_data, 0, 1, axis=1)  # Thêm cột bias
        prediction = predictf(input_data, model_weights)
        result = "Được chấp thuận" if prediction[0] > 0.5 else "Bị từ chối"
        return f'Kết quả dự đoán: {result}'
    except ValueError as e:
        return f'Lỗi dữ liệu đầu vào: {str(e)}'
    except Exception as e:
        return f'Lỗi không xác định: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
