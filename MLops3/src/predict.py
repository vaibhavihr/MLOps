import joblib
model = joblib.load("models/model.pkl")
flower = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}
sl = float(input("Enter Sepal Length : "))
sw = float(input("Enter Sepal Width : "))
pl = float(input("Enter Petal Length : "))
pw = float(input("Enter Petal Width : "))
sample = [[sl, sw, pl, pw]]
prediction = model.predict(sample)
print("Predicted Flower Species :", flower[prediction[0]])