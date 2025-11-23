import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Setup ---
# Transcribing the data from the image table
# w1 (Class 1)
w1 = np.array([
    [0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8],
    [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]
])

# w2 (Class 2)
w2 = np.array([
    [7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9],
    [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]
])

# w4 (Class 4)
w4 = np.array([
    [-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
    [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]
])

def prepare_data(pos_class_data, neg_class_data):
    """
    1. Augments data with x0 = 1 (Bias term).
    2. Normalizes the negative class by multiplying by -1.
    """
    # Add bias term (column of 1s)
    pos_aug = np.c_[np.ones(len(pos_class_data)), pos_class_data]
    neg_aug = np.c_[np.ones(len(neg_class_data)), neg_class_data]
    
    # Normalize: Negate the samples from the second category
    neg_aug = -1 * neg_aug
    
    # Combine
    X = np.vstack((pos_aug, neg_aug))
    return X

# --- 2. Algorithm Implementation ---

def batch_perceptron(X, learning_rate, max_epochs=1000):
    """
    Implements Batch Gradient Descent for the Perceptron Criterion.
    J_p(a) = sum(-a^t * y) for misclassified samples y.
    Gradient = sum(-y)
    Update: a(k+1) = a(k) + learning_rate * sum(y_misclassified)
    """
    num_samples, num_features = X.shape
    
    # Initialize weights (random small numbers or zeros)
    # Using zeros ensures deterministic behavior for the homework check
    a = np.zeros(num_features) 
    
    cost_history = []
    ops_count = 0 # Crude estimation for Part (b)
    
    for epoch in range(max_epochs):
        # 1. Calculate Discriminant function: a^t * y
        # Matrix multiplication: (N x 3) dot (3 x 1)
        g_x = np.dot(X, a)
        ops_count += num_samples * num_features # mults + adds
        
        # 2. Find misclassified samples
        # A sample is misclassified if a^t * y <= 0
        misclassified_indices = np.where(g_x <= 0)[0]
        
        # 3. Calculate Criterion Function Value (Cost)
        # J = sum(-a^t * y) for misclassified
        if len(misclassified_indices) > 0:
            # Sum of discriminant values of misclassified samples (negated)
            cost = -np.sum(g_x[misclassified_indices])
            cost_history.append(cost)
            
            # 4. Calculate Gradient and Update Weights
            # Gradient of J wrt a is sum(-y). 
            # Update rule: a = a - eta * gradient => a = a + eta * sum(y)
            
            # Sum the misclassified vectors
            sum_y_mis = np.sum(X[misclassified_indices], axis=0)
            ops_count += len(misclassified_indices) * num_features # adds
            
            # Update weights
            update_vector = learning_rate * sum_y_mis
            a = a + update_vector
            ops_count += num_features * 2 # scalar mult + add
            
        else:
            # No misclassified samples -> Convergence
            cost_history.append(0)
            break
            
    return a, cost_history, epoch + 1, ops_count

# --- 3. Execution: Part (a) ---

# Case 1: w1 vs w2
X_12 = prepare_data(w1, w2)
weights_12, history_12, epochs_12, ops_12 = batch_perceptron(X_12, learning_rate=0.1)

# Case 2: w1 vs w4
X_14 = prepare_data(w1, w4)
weights_14, history_14, epochs_14, ops_14 = batch_perceptron(X_14, learning_rate=0.1)

# Plotting Part (a)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(history_12) + 1), history_12, marker='o')
plt.title(r'Convergence: $\omega_1$ vs $\omega_2$ ($\eta=0.1$)')
plt.xlabel('Iteration number')
plt.ylabel('Criterion Function $J_p(a)$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(history_14) + 1), history_14, marker='o', color='orange')
plt.title(r'Convergence: $\omega_1$ vs $\omega_4$ ($\eta=0.1$)')
plt.xlabel('Iteration number')
plt.ylabel('Criterion Function $J_p(a)$')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"w1 vs w2 converged in {epochs_12} iterations.")
print(f"w1 vs w4 converged in {epochs_14} iterations.")

# --- 4. Execution: Part (c) ---
# Plot convergence time vs learning rate

learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0]
convergence_times_12 = []
convergence_times_14 = []

print("\n--- Part (c) Analysis ---")
for lr in learning_rates:
    _, _, e_12, _ = batch_perceptron(X_12, learning_rate=lr, max_epochs=2000)
    _, _, e_14, _ = batch_perceptron(X_14, learning_rate=lr, max_epochs=2000)
    
    convergence_times_12.append(e_12)
    convergence_times_14.append(e_14)
    
    # Check for failure to converge (hit max_epochs)
    if e_12 >= 2000:
        print(f"w1 vs w2 failed to converge at learning rate: {lr}")
    if e_14 >= 2000:
        print(f"w1 vs w4 failed to converge at learning rate: {lr}")

plt.figure(figsize=(10, 6))
plt.plot(learning_rates, convergence_times_12, marker='s', label=r'$\omega_1$ vs $\omega_2$')
plt.plot(learning_rates, convergence_times_14, marker='^', label=r'$\omega_1$ vs $\omega_4$')
plt.title('Convergence Time vs Learning Rate')
plt.xlabel('Learning Rate $\eta$')
plt.ylabel('Iterations to Converge')
plt.xscale('log') # Log scale helps visualize the spread
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()

# --- Part (b) Estimate ---
print("\n--- Part (b) Estimate ---")
print(f"Approximate elementary operations for w1 vs w2: {ops_12}")
print(f"Approximate elementary operations for w1 vs w4: {ops_14}")