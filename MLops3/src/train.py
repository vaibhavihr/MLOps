import joblib
from sklearn.tree import DecisionTreeClassifier
from preprocessing import load_data
X_train, X_test, y_train, y_test = load_data()
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/model.pkl")