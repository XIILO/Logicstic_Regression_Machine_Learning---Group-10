from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

# GIAI ĐOẠN 1: TIỀN XỬ LÝ DATASET
# 1.1 Import data 
file_path = "C:/Users/lebin/Downloads/Loan_Approval/Training/loan_approval_dataset.csv"
data = pd.read_csv(file_path)
# Xóa cột 'loan_id' khỏi data
data = data.drop('loan_id', axis=1)   

# 1.2 Xử lý khoảng trắng của các cột
data.columns = data.columns.str.strip()

# 1.3 Xử lý khoảng trắng trong dữ liệu thuộc các cột (trường)
data['self_employed'] = data['self_employed'].str.strip()
data['education'] = data['education'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()

# 1.4 Ánh xạ dữ từ chữ sang số 
data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# 1.5 Thêm cột bias vào dataset
data.insert(0, 'bias', 1)
# 1.6 Kiểm tra dữ liệu có NaN hay không ?
print(data.isna().sum())    

# 1.7 Chuyển đổi dữ liệu trong các cột, để nó nằm trong khoảng MinMax(0 - 1), trừ cột loan_status
scaler = MinMaxScaler()
data[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']] = scaler.fit_transform(data[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']])


# GIAI ĐOẠN 2: CHUẨN BỊ DỮ LIỆU
# 2.1 Tách đặc trưng và nhãn
features_data = data[['bias','no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]
lables_data = data['loan_status']

# 2.2 Tách dữ liệu thành tập huấn luyện và kiểm tra
features_training, features_testing, lables_training, lables_testing = train_test_split(features_data, lables_data, test_size=0.1, random_state=42)

# 2.3 Chuyển labels về dạng numpy arrays
lables_training = lables_training.values
lables_testing = lables_testing.values

# GIAI ĐOẠN 3: CÀI ĐẶT THUẬT TOÁN HỒI QUY LOGISTIC
# 3.1 Hàm Sigmoid
def sigmoidf(z):
    return 1 / (1 + np.exp(-z))

# 3.2 Hàm dự đoán
def predictf(features, weights):
    z = np.dot(features, weights)
    return sigmoidf(z)
    
# 3.3 Hàm mất mát
def log_lossf(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = len(y_true)
    loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 3.4 Hàm Gradient Descent
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

# 3.5 Khởi tạo trọng số với các giá trị ngẫu nhiên
weights = np.zeros(features_training.shape[1])
# 3.6 Chọn learning rate và số vòng lặp
learning_rate = 0.001
iterations = 1000
# 3.7 Huấn luyện mô hình
weights = gradient_descent(features_training, lables_training, weights, learning_rate, iterations)

# 3.8 Kiểm tra kết quả dự đoán trên tập kiểm tra
y_pred_testing = predictf(features_testing, weights)
y_pred_testing = [1 if i > 0.5 else 0 for i in y_pred_testing]

# 3.8 In kết quả
print(f"Predicted labels: {y_pred_testing}")

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 3.9 Đánh giá độ chính xác
accuracy_score = accuracy(lables_testing, y_pred_testing)
print(f"Accuracy: {accuracy_score * 100}%")

# Lưu trọng số vào biến toàn cục để sử dụng trong Flask
global model_weights
model_weights = weights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    prediction = predictf(input_data, model_weights)  # Truyền weights vào đây
    return f'Kết quả dự đoán: {"Được chấp thuận" if prediction[0] > 0.5 else "Bị từ chối"}'

if __name__ == '__main__':
    app.run(debug=True)
