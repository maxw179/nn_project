import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

# Normalize the data
X /= 255.0

# Step 2: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define a simple neural network model using TensorFlow
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Step 4: Extract the coefficients (weights) of the model
weights = model.get_weights()

# For simplicity, let's focus on the first dense layer's weights
# Extract the weight matrix and bias vector of the first Dense layer
W1, b1 = weights[0], weights[1]

# Step 5: Create a graph Laplacian from the weight matrix
# Construct a similarity matrix based on the weights
similarity_matrix = np.dot(W1.T, W1)

# Build the graph using the similarity matrix
graph = nx.from_numpy_array(similarity_matrix)

# Step 6: Compute the graph Laplacian
L = nx.laplacian_matrix(graph).todense()

# Step 7: Compute eigenvalues and eigenvectors of the Laplacian
eigenvalues, eigenvectors = linalg.eigh(L)

# Step 8: Zero out some eigenvalues
zeroed_eigenvalues = eigenvalues.copy()
zeroed_eigenvalues[::2] = 0  # Zero out every second eigenvalue (for example)

# Step 9: Reconstruct the Laplacian after modifying eigenvalues
L_reconstructed = eigenvectors @ np.diag(zeroed_eigenvalues) @ eigenvectors.T

# Step 10: Reconstruct the model (simplified example) and test on the test set
# For simplicity, we just pass the reconstructed weights into the model as a rough way to "test"
model.set_weights(weights)  # Apply the original weights

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy before modification: {test_acc}')

# Apply the eigenvalue modification to the weight matrix
W1_reconstructed = W1 @ (eigenvectors @ np.diag(zeroed_eigenvalues) @ eigenvectors.T)

# Create a new weights list with the modified first layer weights
new_weights = weights.copy()
new_weights[0] = W1_reconstructed

# Set the new weights back into the model
model.set_weights(new_weights)  # Apply the modified weights

# Step 11: Evaluate performance after zeroing eigenvalues
# Evaluate the model on the test set using the modified weights
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy after modification: {test_acc}')

# -------------------------------
# Step 12 (revised): Test importance of first 50 eigenvectors and plot results
# -------------------------------
import matplotlib.pyplot as plt

# 1) Save original weights & baseline accuracy
original_weights = model.get_weights()
baseline_acc = model.evaluate(X_test, y_test, verbose=0)[1]

# 2) Decide how many eigenvectors to test
num_to_test = 50
indices = np.arange(num_to_test)

# 3) Allocate list for accuracy drops
accuracy_drops = []

# 4) Loop over just those first 50 eigenvectors
for i in indices:
    # a) Zero out the i-th eigenvalue
    mod_eigs = eigenvalues.copy()
    mod_eigs[i] = 0

    # b) Reconstruct the filtered Laplacian
    filt = eigenvectors @ np.diag(mod_eigs) @ eigenvectors.T

    # c) Apply it to the first layer’s weights
    W1_filtered = W1 @ filt

    # d) Swap in modified weights
    new_weights = original_weights.copy()
    new_weights[0] = W1_filtered
    model.set_weights(new_weights)

    # e) Evaluate and record drop
    _, acc_mod = model.evaluate(X_test, y_test, verbose=0)
    accuracy_drops.append(baseline_acc - acc_mod)

# 5) Restore the original weights
model.set_weights(original_weights)

# 6) Plot results for only those 50 components
plt.figure(figsize=(10, 6))
plt.plot(indices, accuracy_drops, marker='o')
plt.xticks(indices)  # or plt.xticks(indices, rotation=90) if crowded
plt.xlabel("Eigenvector index (0–49)")
plt.ylabel("Accuracy drop when removed")
plt.title("Importance of First 50 Spectral Components")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 13: Per‐digit test accuracy bar chart
# -------------------------------

# 1) Get predicted labels on the test set
probs = model.predict(X_test)                # shape (N, 10)
y_pred = np.argmax(probs, axis=1)            # shape (N,)

# 2) Compute accuracy for each digit 0–9
classes = np.arange(10)
per_class_acc = [
    np.mean(y_pred[y_test == i] == i)
    for i in classes
]

# 3) Plot as a bar chart
plt.figure(figsize=(8, 6))
plt.bar(classes, per_class_acc, color='skyblue')
plt.xticks(classes)
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.title("Per‐Digit Test Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
