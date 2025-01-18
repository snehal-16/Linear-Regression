import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X_data = pd.read_csv('linearX.csv').values.flatten()
Y_data = pd.read_csv('linearY.csv').values.flatten()

X_norm = (X_data - np.mean(X_data)) / np.std(X_data)
Y_norm = (Y_data - np.mean(Y_data)) / np.std(Y_data)

theta_0 = 0
theta_1 = 0
learning_rate = 0.5
iterations = 50
m = len(Y_norm)

#cost fxn
def compute_cost(X, Y, theta_0, theta_1):
    predictions = theta_0 + theta_1 * X
    cost = (1 / (2 * len(Y))) * np.sum((predictions - Y) ** 2)
    return cost

#gradient descent
cost_history = []
for _ in range(iterations):
    predictions = theta_0 + theta_1 * X_norm
    error = predictions - Y_norm

    theta_0 -= learning_rate * (1 / m) * np.sum(error)
    theta_1 -= learning_rate * (1 / m) * np.sum(error * X_norm)

    cost_history.append(compute_cost(X_norm, Y_norm, theta_0, theta_1))

# plotting cost vs iteration
plt.figure(figsize=(8, 6))
plt.plot(range(iterations), cost_history, label='Cost Function')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations')
plt.legend()
plt.grid()
plt.show()

# denormalising for visualisation
theta_1_denorm = theta_1 * (np.std(Y_data) / np.std(X_data))
theta_0_denorm = np.mean(Y_data) - theta_1_denorm * np.mean(X_data)

# predictions using regression line
X_line = np.linspace(min(X_data), max(X_data), 100)
Y_line = theta_0_denorm + theta_1_denorm * X_line

# plotting dataset and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_data, Y_data, label='Data Points', color='blue')
plt.plot(X_line, Y_line, label='Regression Line', color='red')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid()
plt.show()

# testing diff learning rates
learning_rates = [0.005, 0.5, 5]
cost_histories = {}

for lr in learning_rates:
    theta_0, theta_1 = 0, 0
    cost_history = []

    for _ in range(iterations):
        predictions = theta_0 + theta_1 * X_norm
        error = predictions - Y_norm

        theta_0 -= lr * (1 / m) * np.sum(error)
        theta_1 -= lr * (1 / m) * np.sum(error * X_norm)

        cost_history.append(compute_cost(X_norm, Y_norm, theta_0, theta_1))

    cost_histories[lr] = cost_history

# plotting cost function for different learning rates
plt.figure(figsize=(10, 8))
for lr, cost_history in cost_histories.items():
    plt.plot(range(iterations), cost_history, label=f'lr = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Changes for Different Learning Rates')
plt.legend()
plt.grid()
plt.show()
