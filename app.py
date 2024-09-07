import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Đọc dữ liệu từ file CSV
file_path = "C:/Users/lebin/Downloads/Loan_Approval/Training/loan_approval_dataset.csv"
data = pd.read_csv(file_path)
print(data.head(5))

# Ánh xạ các giá trị chuỗi sang số
data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
print(data.head(5))

print(data.isna().sum())
# Chọn các cột cần chuẩn hóa, bao gồm cả các cột mới
columns_to_normalize = [
    'no_of_dependents', 'income_annum', 'loan_amount', 
    'loan_term', 'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

# Sử dụng Min-Max Scaling, đưa dữ liệu trong các cột nằm trong khoảng 0 - 1 
scaler = MinMaxScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Tách các đặc trưng và nhãn
X = np.array(data.drop(['loan_id', 'loan_status'], axis=1).values, dtype=float)
y = np.array(data['loan_status'].values, dtype=float)

# Thêm cột bias (cột toàn giá trị 1) vào x
X = np.insert(X, 0, 1, axis=1)

# Khởi tạo tham số theta
theta = np.zeros(X.shape[1])

# Cài đặt hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cài đặt hàm chi phí (cost function)
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost

# Cài đặt Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1/m) * X.T @ (sigmoid(X @ theta) - y)
        theta -= learning_rate * gradient
    return theta

# Huấn luyện mô hình
theta = gradient_descent(X, y, theta, learning_rate=0.01, iterations=2000)

# Dự đoán và thiết lập ngưỡng
def predict(X, theta, threshold=0.5):
    return sigmoid(X @ theta) >= threshold

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_loan():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    input_df['education'] = input_df['education'].map({'Graduate': 1, 'Not Graduate': 0})
    input_df['self_employed'] = input_df['self_employed'].map({'Yes': 1, 'No': 0})
    
    # Chuyển đổi các giá trị đầu vào thành số thực
    for column in columns_to_normalize:
        input_df[column] = pd.to_numeric(input_df[column])
    
    input_df[columns_to_normalize] = scaler.transform(input_df[columns_to_normalize])
    input_df.insert(0, 'bias', 1)
    
    # Đảm bảo số lượng cột khớp với mô hình
    if input_df.shape[1] != len(theta):
        return render_template('index.html', prediction_text='Error: Input data does not match model dimensions.')
    
    prediction = predict(input_df.values, theta)
    return render_template('index.html', prediction_text='Loan Status: {}'.format('Approved' if prediction[0] else 'Rejected'))

# Thêm dòng này vào cuối file
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
