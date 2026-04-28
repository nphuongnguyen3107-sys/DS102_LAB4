import numpy as np
from data_processing import load_and_preprocess_data

# Import trực tiếp mô hình từ thư viện scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Tải và xử lý dữ liệu
X_train, X_test, y_train, y_test = load_and_preprocess_data()
print(f"Dữ liệu: {X_train.shape[0]} mẫu train, {X_test.shape[0]} mẫu test\n")

print("="*40)
print("1. KẾT QUẢ DECISION TREE (SCIKIT-LEARN)")
print("="*40)

# Desicion Tree với các siêu tham số tương tự bài tự code để dễ so sánh
# Khởi tạo mô hình với các siêu tham số tương tự bài tự code để dễ so sánh
dt_clf = DecisionTreeClassifier(max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42)
dt_clf.fit(X_train, y_train)

dt_y_pred = dt_clf.predict(X_test)

print(f"Accuracy    : {accuracy_score(y_test, dt_y_pred):.4f}")
print(f"F1 macro    : {f1_score(y_test, dt_y_pred, average='macro'):.4f}")
print(f"F1 weighted : {f1_score(y_test, dt_y_pred, average='weighted'):.4f}")

# Random Forest với các siêu tham số tương tự bài tự code để dễ so sánh
print("\n" + "="*40)
print("2. KẾT QUẢ RANDOM FOREST (SCIKIT-LEARN)")
print("="*40)

# Khởi tạo Random Forest (n_jobs=-1 giúp mô hình sử dụng toàn bộ nhân CPU để chạy siêu tốc)
rf_clf = RandomForestClassifier(n_estimators=20, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

rf_y_pred = rf_clf.predict(X_test)

print(f"Accuracy    : {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"F1 macro    : {f1_score(y_test, rf_y_pred, average='macro'):.4f}")
print(f"F1 weighted : {f1_score(y_test, rf_y_pred, average='weighted'):.4f}")

