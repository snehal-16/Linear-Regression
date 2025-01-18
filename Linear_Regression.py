import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
X_data = pd.read_csv('linearX.csv').values.flatten()
Y_data = pd.read_csv('linearY.csv').values.flatten()

# Normalize the predictor (X) and target variable (Y)
X_norm = (X_data - np.mean(X_data)) / np.std(X_data)
Y_norm = (Y_data - np.mean(Y_data)) / np.std(Y_data)

# Initialize parameters
theta_0 = 0  
theta_1 = 0  
learning_rate = 0.5 
iterations = 1000  
m = len(Y_norm) 
convergence_threshold = 1e-6  

# Function to compute cost
def compute_cost(X, Y, theta_0, theta_1):
    predictions = theta_0 + theta_1 * X
    cost = (1 / (2 * len(Y))) * np.sum((predictions - Y) ** 2)
    return cost

# Batch Gradient Descent
cost_history = []
for i in range(iterations):
    predictions = theta_0 + theta_1 * X_norm
    error = predictions - Y_norm

    # Update parameters
    theta_0 -= learning_rate * (1 / m) * np.sum(error)
    theta_1 -= learning_rate * (1 / m) * np.sum(error * X_norm)

    # Compute cost
    current_cost = compute_cost(X_norm, Y_norm, theta_0, theta_1)
    cost_history.append(current_cost)

    # Check convergence
    if i > 0 and abs(cost_history[-1] - cost_history[-2]) < convergence_threshold:
        break

# Final parameters and cost
final_cost = cost_history[-1]
print(f"Final Cost: {final_cost}")
print(f"Final Parameters: theta_0 = {theta_0}, theta_1 = {theta_1}")

# Plot cost vs iteration for the first 50 iterations
plt.figure(figsize=(8, 6))
plt.plot(range(min(50, len(cost_history))), cost_history[:50], label='Cost Function')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations (First 50 Iterations)')
plt.legend()
plt.grid()
plt.show()

# Denormalize parameters for visualization
theta_1_denorm = theta_1 * (np.std(Y_data) / np.std(X_data))
theta_0_denorm = np.mean(Y_data) - theta_1_denorm * np.mean(X_data)

# Generate regression line
X_line = np.linspace(min(X_data), max(X_data), 100)
Y_line = theta_0_denorm + theta_1_denorm * X_line

# Plot dataset and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_data, Y_data, label='Data Points', color='blue')
plt.plot(X_line, Y_line, label='Regression Line', color='red')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid()
plt.show()

# Testing Different Learning Rates
learning_rates = [0.005, 0.5, 5]
cost_histories = {}

for lr in learning_rates:
    theta_0, theta_1 = 0, 0
    cost_history = []

    for _ in range(50):  # First 50 iterations
        predictions = theta_0 + theta_1 * X_norm
        error = predictions - Y_norm

        theta_0 -= lr * (1 / m) * np.sum(error)
        theta_1 -= lr * (1 / m) * np.sum(error * X_norm)

        cost_history.append(compute_cost(X_norm, Y_norm, theta_0, theta_1))

    cost_histories[lr] = cost_history

# Plot cost function for different learning rates
plt.figure(figsize=(10, 8))
for lr, cost_history in cost_histories.items():
    plt.plot(range(50), cost_history, label=f'lr = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Changes for Different Learning Rates')
plt.legend()
plt.grid()
plt.show()

# Stochastic Gradient Descent
theta_0, theta_1 = 0, 0
cost_history_sgd = []

for _ in range(50):
    for i in range(m):
        prediction = theta_0 + theta_1 * X_norm[i]
        error = prediction - Y_norm[i]

        theta_0 -= learning_rate * error
        theta_1 -= learning_rate * error * X_norm[i]

    cost_history_sgd.append(compute_cost(X_norm, Y_norm, theta_0, theta_1))

# Mini-batch Gradient Descent
batch_size = 10
theta_0, theta_1 = 0, 0
cost_history_mbgd = []

for _ in range(50):
    for i in range(0, m, batch_size):
        X_batch = X_norm[i:i + batch_size]
        Y_batch = Y_norm[i:i + batch_size]

        predictions = theta_0 + theta_1 * X_batch
        error = predictions - Y_batch

        theta_0 -= learning_rate * (1 / len(Y_batch)) * np.sum(error)
        theta_1 -= learning_rate * (1 / len(Y_batch)) * np.sum(error * X_batch)

    cost_history_mbgd.append(compute_cost(X_norm, Y_norm, theta_0, theta_1))

# Plot comparisons of gradient descent methods
plt.figure(figsize=(10, 8))
plt.plot(range(50), cost_history[:50], label='Batch Gradient Descent')
plt.plot(range(50), cost_history_sgd, label='Stochastic Gradient Descent')
plt.plot(range(50), cost_history_mbgd, label='Mini-batch Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Comparison of Gradient Descent Methods')
plt.legend()
plt.grid()
plt.show()
