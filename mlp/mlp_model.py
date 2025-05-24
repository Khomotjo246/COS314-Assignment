import numpy as np
import pandas as pd

def sigmoid(n):
    return 1/ (1 + np.exp(-n))

def sigmoid_derivative(n): 
    return n * (1 - n)

trainData = pd.read_csv("../Data/BTC_train.csv")
xTrain = trainData.iloc[:, :-1].values
yTrain = trainData.iloc[:, -1].values.reshape(-1, 1)
yTrain = trainData["Output"].values

testData = pd.read_csv("../Data/BTC_test.csv")
xTest = testData.iloc[:, :-1].values
yTest = testData.iloc[:, -1].values.reshape(-1, 1)

inputSize = xTrain.shape[1]
hiddenSize = [12] * 3   # 4 hidden layers with 12 neurons
outputSize = 1
layerSize = [inputSize] + hiddenSize + [outputSize]

#Weights
np.random.seed(41)
weights = [np.random.uniform(-0.5, 0.5, (layerSize[i], layerSize[i+1])) for i in range(len(layerSize)-1)]
biases = [np.random.uniform(-0.5, 0.5, (1, size)) for size in layerSize[1:]]

learningRate = 0.05
epochs = 1300

for epoch in range(epochs):
    for x,y in zip(xTrain, yTrain):
        x = x.reshape(1, -1)
        y = np.array([[y]])
        activations = [x]

        z_arr = []
        for w, b in zip(weights, biases):
            z = np.dot(activations[-1], w) + b
            z_arr.append(z)
            activate = sigmoid(z)
            activations.append(activate)

        # Backpropagation
        delta = (y - activations[-1]) * sigmoid_derivative(activations[-1])
        delta_arr = [delta]
        for i in reversed(range(len(weights)-1)):
            delta = np.dot(delta_arr[-1], weights[i+1].T) * sigmoid_derivative(activations[i+1])
            delta_arr.append(delta)
        delta_arr.reverse()

        # Update
        for i in range(len(weights)):
            weights[i] += learningRate * np.dot(activations[i].T, delta_arr[i])
            biases[i] += learningRate * delta_arr[i]

    if epoch % 100 == 0:
        preds = []
        for x in xTrain:
            a = x.reshape(1, -1)
            for w,b in zip(weights, biases):
                a = sigmoid(np.dot(a, w) + b)
            preds.append(a[0][0])
        preds = np.array(preds).reshape(-1, 1)
        loss = np.mean((yTrain.reshape(-1, 1) - preds) ** 2)
        loss = round(loss, 4)
        print(f"Epoch {epoch}, Loss: {loss}")

results = []
preds = []
correct = 0
for x,y in zip(xTest, yTest):
    a = x.reshape(1, -1)
    for w, b in zip(weights, biases):
        a = sigmoid(np.dot(a, w) + b)
    pred = int(a[0][0] > 0.5)
    correct += pred == y[0]
    preds.append(pred)
    results.append({
        "Input_Open": x[0],
        "Input_High": x[1],
        "Input_Low": x[2],
        "Input_Close": x[3],
        "Input_AdjClose": x[4] if len(x) > 4 else None,
        "Actual": int(y[0]),
        "Predicted": pred
    })

accuracy = round((correct / len(yTest))*100, 4)
print(f"Test accuracy: {accuracy}%")

results = pd.DataFrame(results)

with pd.ExcelWriter("BTC prediction results.xlsx", engine="openpyxl") as writer:
    results.to_excel(writer, sheet_name="BTC preditctions", index=False)

    summary = pd.DataFrame([{
        "Metric": "Test Accuracy (%)",
        "Value": accuracy
    }])
    summary.to_excel(writer, sheet_name="Summary", index= False)

print("Written to excel file")