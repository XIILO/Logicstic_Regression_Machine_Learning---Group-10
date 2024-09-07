import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
sys.stdout.reconfigure(encoding = "utf-8")

# GIAI ĐOẠN 1: TIỀN XỬ LÝ DATASET
# 1.1 Import data 
file_path = "C:/Users/lebin/Downloads/Loan_Approval/Training/loan_approval_dataset.csv"
data = pd.read_csv(file_path)   
print(data.head(2))

# 1.2 Xử lý khoảng trắng của các cột
data.columns = data.columns.str.strip()

# 1.3 Xử lý khoảng trắng trong dữ liệu thuộc các cột (trường)
data['self_employed'] = data['self_employed'].str.strip()
data['education'] = data['education'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()
print(data.head(2))

# 1.4 Ánh xạ dữ từ chữ sang số 
data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
print(data.head(2))

# 1.5 Kiểm tra dữ liệu có NaN hay không ?
print(data.isna().sum())

# 1.6 Chuyển đổi dữ liệu trong các cột, để nó nằm trong khoảng MinMax(0 - 1)
scaler = MinMaxScaler()
data[['loan_id', 'no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'loan_status']] = scaler.fit_transform(data[['loan_id', 'no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'loan_status']])
print(data.head(2))


#Giai Đoạn 3: CHUẨN BỊ DỮ LIỆU

# 3.1 Thêm cột bias vào dataset
data.insert(0, 'bias', 1)
# 3.2 Tách đặc trưng và nhãn
features_data = data[['bias','education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]
lables_data = data['loan_status']
print("Tách đặc trưng (features)")
print(features_data.head(2))
print("Tách nhãn (lables)")
print(lables_data.head(2))

# 3.3 Tách dữ liêu thành tập huấn luyện và kiểm tra
features_training, features_tesing, lables_training, lables_testing = train_test_split(features_data,lables_data, test_size=0.2, random_state=42)


# GIAI ĐOẠN 4: CÀI ĐẶT THUẬT TOÁN HỒI QUY LOGISTIC
# 2.1 Hàm Sigmoid
def sigmoidf(z):
    return (1/(1 +np.exp(-z)))

# 2.2 hàm dự đoán
def predictf(features_training, weight_by_gradientdescent):
    z = np.dot(features_training, weight_by_gradientdescent)
    return sigmoidf(z)

# 2.3 Hàm mất mát
def log_lossf(y_true, y_pred):
    m = len(y_true)
    loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def gradient_descent(features_training, lables_training, weights, learning_rate, iterations):
    m = len(lables_training)
    for i in range(iterations):
        y_pred = predictf(features_training,weights)
        gradient = np.dot(features_training.T, (y_pred - lables_training)) / m
        weights -= learning_rate * gradient
        
        # Tính và in hàm mất mát sau mỗi vài vòng lặp
        if i % 100 == 0:
            loss = log_lossf(y, y_pred)
            print(f"Iteration {i}: Loss = {loss}")
    
    return weights


# Khởi tạo trọng số với các giá trị ngẫu nhiên
weights = np.zeros(features_training.shape[1])
# Chọn learning rate và số vòng lặp
learning_rate = 0.01
iterations = 1000

# Huấn luyện mô hình
weights = gradient_descent(features_training, lables_training, weights, learning_rate, iterations)

# Kiểm tra kết quả dự đoán trên tập kiểm tra
y_pred_testing = predictf(features_tesing, weights)
y_pred_testing = [1 if i > 0.5 else 0 for i in y_pred_testing]

# In kết quả
print(f"Predicted labels: {y_pred_testing}")

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Đánh giá độ chính xác
accuracy_score = accuracy(lables_testing, y_pred_testing)
print(f"Accuracy: {accuracy_score * 100}%")

    