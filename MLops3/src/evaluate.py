import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_data
X_train, X_test, y_train, y_test = load_data()
model = joblib.load("models/model.pkl")
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))