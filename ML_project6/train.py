import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

mlflow.set_tracking_uri(
	os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
mlflow.set_experiment("Iris Model Versioning")

# STEP 1: Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# STEP 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42
)

# STEP 4: Start V2 MLflow Run
with mlflow.start_run(run_name="RandomForest_V2"):
	# STEP 5: Define Random Forest Parameters
	model_type = "RandomForest"
	n_estimators = 100
	max_depth = 5

	# STEP 6: Create Random Forest Model
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		random_state=42
	)

	# STEP 7: Train Model
	model.fit(X_train, y_train)

	# STEP 8: Make Predictions
	predictions = model.predict(X_test)

	# STEP 9: Calculate Accuracy
	accuracy = accuracy_score(y_test, predictions)
	print("Model:", model_type)
	print("Number of Estimators:", n_estimators)
	print("Maximum Depth:", max_depth)
	print("Accuracy:", accuracy)

	# STEP 10: Log Parameters
	mlflow.log_param("model_type", model_type)
	mlflow.log_param("n_estimators", n_estimators)
	mlflow.log_param("max_depth", max_depth)

	# STEP 11: Log Accuracy
	mlflow.log_metric("accuracy", accuracy)

	# STEP 12: Generate Confusion Matrix
	cm = confusion_matrix(y_test, predictions)
	plt.figure(figsize=(6, 5))
	plt.imshow(cm)
	plt.title("Confusion Matrix - Random Forest V2")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.colorbar()

	for row_index in range(len(cm)):
		for column_index in range(len(cm)):
			plt.text(
				column_index,
				row_index,
				cm[row_index, column_index],
				ha="center",
				va="center"
			)

	plt.tight_layout()
	plt.savefig("confusion_matrix_v2.png")
	plt.close()

	# STEP 13: Log Confusion Matrix
	mlflow.log_artifact("confusion_matrix_v2.png")

	# STEP 14: Register V2 Model
	mlflow.sklearn.log_model(
		model,
		"model",
		registered_model_name="Iris_Classifier"
	)

# STEP 15: Display Result
print("Registered RandomForest as V2")
print("Accuracy:", accuracy)