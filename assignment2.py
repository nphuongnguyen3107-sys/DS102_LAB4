import numpy as np
from data_processing import load_and_preprocess_data
from models.random_forest import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Tải và xử lý dữ liệu
X_train, X_test, y_train, y_test = load_and_preprocess_data()
print(f"Data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

# Khởi tạo mô hình Random Forest với các siêu tham số đã chọn
clf = RandomForestClassifier(n_estimators=20, max_depth=12, min_samples_split=5, min_samples_leaf=2)

print("\nĐang huấn luyện Random Forest scratch (vui lòng chờ)...")
clf.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = clf.predict(X_test)

# Đánh giá kết quả
print("\n" + "="*30)
print("=== Random Forest Scratch ===")
print("="*30)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

