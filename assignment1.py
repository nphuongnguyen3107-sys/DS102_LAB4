from data_processing import load_and_preprocess_data
from models.decision_tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score as accuracy

X_train, X_test, y_train, y_test = load_and_preprocess_data()
print("Train:", X_train.shape, "| Test:", X_test.shape)

clf = DecisionTreeClassifier(max_depth=12, min_samples_split=5, min_samples_leaf=2)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

print("\n=== Training ===")
print(f"Accuracy    : {accuracy(y_train, y_pred_train):.4f}")
print(f"F1 macro    : {f1_score(y_train, y_pred_train, average='macro'):.4f}")
print(f"F1 weighted : {f1_score(y_train, y_pred_train, average='weighted'):.4f}")

print("\n=== Testing ===")
print(f"Accuracy    : {accuracy(y_test, y_pred_test):.4f}")
print(f"F1 macro    : {f1_score(y_test, y_pred_test, average='macro'):.4f}")
print(f"F1 weighted : {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
