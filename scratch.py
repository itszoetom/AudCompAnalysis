import numpy as np
# Q1: Least Square (35 Points)
# Q1a: Gradient Descent (20 Points)
x = np.array([[1, 1], [-1, -2], [2, -1]])
y = np.array([-0.8, 0.1, -5.0])
weights = np.array([0.0, 0.0, 0.0])
alpha = 0.05
T =4
n = len(y)
for t in range(T):
    y_pred = weights[0] + weights[1] * x[:, 0] + weights[2] * x[:, 1]
    errors = y - y_pred
    grad_w0 = -2 / n * np.sum(errors)
    grad_w1 = -2 / n * np.sum(errors * x[:, 0])
    grad_w2 = -2 / n * np.sum(errors * x[:, 1])
    weights[0] -= alpha * grad_w0
    weights[1] -= alpha * grad_w1
    weights[2] -= alpha * grad_w2
    total_loss = np.sum(errors**2)
    avg_loss = total_loss / n
    print(f"Iteration {t+1}: Weights = {weights}, Total Loss = {total_loss:.3f}, Average Loss = {avg_loss:.3f}")
# Q1b: Closed-Form Solution (15 Points)
w = np.linalg.inv(x.T @ x) @ x.T @ y
y_pred = x @ w
total_error = np.sum((y - y_pred)**2)
print(f"Weights (w0, w1, w2): {w}")
print(f"Total Error: {total_error:.4f}")
# Q2: Perceptrons (30 Points)
data = [
    [1, 1, 1],
    [2, -1, -1],
    [-3, -1, 1],
    [-3, 1, 1]
]
def perceptron(data, order):
    w = np.array([0, 0])
    print(f"Initial weights: {w}")
    for i in order:
        x = np.array(data[i][:2])
        y = data[i][2]

        prediction = np.sign(np.dot(w, x))
        if prediction != y:
            w=w+y* x
        print(f"Iteration {i+1}: x = {x}, y = {y}, Updated weights: {w}")
    return w
# Q2a: i = 1, 2, 3, 4 (15 Points)
print("\nPart (a): Order 1, 2, 3, 4")
order_a = [0, 1, 2, 3]
final_weights_a = perceptron(data, order_a)
# Q2b: i = 2, 1, 3, 4 (15 Points)
print("\nPart (b): Order 2, 1, 3, 4")
order_b = [1, 0, 2, 3]
final_weights_b = perceptron(data, order_b)
# Q3: Logistic Regression (35 Points)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Q3a: (15 Points)
def logistic_regression(data, alpha, T):
    x = np.array([d[:2] for d in data])
    y = np.array([d[2] for d in data])
    n, d = x.shape
    weights = np.array([0.0] * d)
    for t in range(T):
        z = np.dot(x, weights)
        predictions = sigmoid(z)
        errors = y - predictions
        grad_w = np.dot(errors, x) / n
        weights += alpha * grad_w
        log_likelihood = np.mean(y * np.log(predictions) + (1 - y) * np.log(1 -
predictions))
        print(f"Iteration {t+1}: Weights = {weights}, Log-Likelihood = {log_likelihood:.3f}")
    return weights
alpha_1 = 0.01
alpha_2 = 0.2
T =4
# Q3b: (15 Points)
print("\nLogistic Regression with alpha = 0.01")
weights_alpha_1 = logistic_regression(data, alpha_1, T)
print("\nLogistic Regression with alpha = 0.2")
weights_alpha_2 = logistic_regression(data, alpha_2, T)

# Q3c: (5 Points)
print("\nFinal Weights and Comparison:")
print(f"Final Weights (alpha = 0.01): {weights_alpha_1}")
print(f"Final Weights (alpha = 0.2): {weights_alpha_2}")
