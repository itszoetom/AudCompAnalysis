import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# Initialize parameters and variables
alphas = np.logspace(-6, 3, 200)
test_acc = np.empty_like(alphas)
train_acc = np.empty_like(alphas)

# Generate a 2D array of X values
X = np.linspace(1, 100, 50)
epsilon = 2 * np.random.randn(50)
Y = 3 * X.flatten() + epsilon  # Generate Y values with some noise

# Split data into 80-20 train-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

best_alpha = None
bes_mse = float('inf')  # Initialize with a very high MSE for comparison

# Iterate over alpha (lambda) values
for i, alpha in enumerate(alphas):
    # Initialize Ridge model with the current alpha value
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = ridge_reg.predict(X_test)

    # Calculate Mean Squared Error (MSE) for the test set
    mse = mean_squared_error(Y_test, Y_pred)

    # Check if the current model has the lowest MSE
    if mse < bes_mse:
        bes_mse = mse
        best_alpha = alpha

    # Calculate R^2 values
    train_rsq = ridge_reg.score(X_train, Y_train)
    test_rsq = ridge_reg.score(X_test, Y_test)
    train_acc[i] = train_rsq
    test_acc[i] = test_rsq

# Plot the training and test R^2 scores
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_acc, label="Train $R^2$", color="blue")
plt.plot(alphas, test_acc, label="Test $R^2$", color="red")
plt.axvline(best_alpha, color='green', linestyle='--', label=f"Best alpha = {best_alpha:.4f}")
plt.xscale("log")
plt.xlabel("Alpha (Regularization Parameter)")
plt.ylabel("$R^2$ Score")
plt.title("Ridge Regression: Train and Test $R^2$ vs. Alpha")
plt.legend()
plt.show()

# Display the best alpha and error list
print("Best Alpha:", best_alpha)
print("Lowest Test MSE:", bes_mse)
