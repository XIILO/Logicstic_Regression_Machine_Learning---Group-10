import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf8')
file_path = "C:\\Users\\lebin\\Downloads\\Loan_Approval\\Training\\loan_approval_dataset2.xlsx"

def pre_process_data(file_path):
    data = pd.read_excel(file_path)

    # 1.1 Loại bỏ cột thừa
    data.drop('Số thứ tự', axis=1, inplace=True)
    
    # 1.2 Ánh xạ dữ liệu sang số
    data['Trình độ học vấn'] = data['Trình độ học vấn'].map({'Trung cấp': 1, 'Cao đẳng': 2, 'Đại học': 3, 'Thạc sĩ': 4, 'Tiến sĩ': 5})
    data['Tự kinh doanh'] = data['Tự kinh doanh'].map({'Có': 1, 'Không': 0})
    data['Kết quả'] = data['Kết quả'].map({'Chấp nhận': 1, 'Từ chối': 0})
    
    # 1.3 Xử lý các ký tự không mong muốn và chuyển đổi dữ liệu
    for col in ['Thu nhập hằng tháng', 'Số tiền vay', 'Thời hạn vay', 'Điểm tín dụng', 
                'Tài sản nhà ở', 'Tài sản kinh doanh', 'Tài sản các vật phẩm thế chấp', 'Tài sản của ngân hàng']:
        data[col] = data[col].replace({',': ''}, regex=True).astype(float)
        
    # 1.4 Scale data trong khoảng 0 - 1
    scaler = MinMaxScaler()
    scaler_array = scaler.fit_transform(data)  # Sửa tên biến từ scaler_aray thành scaler_array
    data = pd.DataFrame(scaler_array, columns=data.columns)
    
    # 1.5 Kiểm tra các giá trị bị nan nếu có thì xóa
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    
    # 1.6 Thêm cột bias tính hệ số b0
    data.insert(0, 'Bias', 1)

    # 1.7 Tách đặc trưng và nhãn
    features_data = data.drop('Kết quả', axis=1)
    labels_data = data['Kết quả']  # Sửa lỗi chính tả từ lables_data thành labels_data
    
    # 1.8 Tách dữ liệu thành training và testing
    features_train, features_test, labels_train, labels_test = train_test_split(features_data, labels_data, test_size=0.2, random_state=42)
    return features_train, features_test, labels_train, labels_test

    

# 2.1 Viết hàm sigmoid, tính xác suất
def sigmoid_predict(features_train, thetas):
    z = np.dot(features_train, thetas)
    predictions = 1 / (1 + np.exp(-z))
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)  # Giới hạn giá trị dự đoán
    return predictions

# 2.2 Viết hàm mất mát
def loss_function(labels_predict, labels_train):
    epsilon = 1e-10
    labels_predict = np.clip(labels_predict, epsilon, 1 - epsilon)
    loss = - (1 / len(labels_train)) * np.sum(labels_train * np.log(labels_predict) + (1 - labels_train) * np.log(1 - labels_predict))
    return loss

# 2.3 Viết hàm hạ gradient
def gradient_descent(features_train, labels_train, thetas, learning_rate, iterations):
    n = len(labels_train)
    for i in range(iterations):
        labels_predict = sigmoid_predict(features_train, thetas)
        lost_derivative = np.dot(features_train.T, (labels_predict - labels_train)) / n
        thetas -= learning_rate * lost_derivative
        
        if i % 10000 == 0:
            loss = loss_function(labels_predict, labels_train) 
            print(f"Iteration {i}: Loss = {loss}") 
    return thetas
features_train, features_test, labels_train, labels_test = pre_process_data(file_path)

# Khởi tạo các tham số
learning_rate = 0.01
iterations = 100000 
thetas = np.zeros(features_train.shape[1])
thetas = gradient_descent(features_train, labels_train, thetas, learning_rate, iterations)

# 3 Tính toán độ chính xác
# 3.1 Đem thetas tối ưu nhân với các giá trị kiểm thử trong cột đặc trưng kiểm tra
labels_predict_test = sigmoid_predict(features_test, thetas)

# 3.2 Chuyển đổi dự đoán xác suất thành nhãn nhị phân
def predict(labels_predict):
    return (labels_predict >= 0.5).astype(int)

# 3.3 Tính toán độ chính xác
def accuracy(labels_true, labels_pred):
    return np.mean(labels_true == labels_pred)

# 3.4 So sánh dự đoán với nhãn thực tế
predicted_labels_test = predict(labels_predict_test)
accuracy_score_test = accuracy(labels_test, predicted_labels_test) 
print(f"Test Accuracy: {accuracy_score_test * 100}%")
