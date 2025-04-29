# -------------------------------
# Step 13: Per‐digit test accuracy bar chart
# -------------------------------

# 1) Get predicted labels on the test set
from matplotlib import pyplot as plt


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
