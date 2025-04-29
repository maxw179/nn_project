# -------------------------------
# Per‐digit test accuracy bar chart
# -------------------------------

from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
images = mnist.data.astype(np.float32)
labels = mnist.target.astype(np.int64)

# Normalize the data
images /= 255.0

#create a test set
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)


def plot_digits(weights):
    """this function takes in a list of weights (i.e. a list of one numpy array for each layer)
    and graphs the accuracy for classifying each digit"""

    #first initialize the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    #set weights
    model.set_weights(weights)

    #use the model to classify each image
    probs = model.predict(images_test)           # shape (N, 10)
    label_pred = np.argmax(probs, axis=1)            # shape (N,)

    # Compute accuracy for each digit 0–9
    digits = np.arange(10)
    #loop over each digit i
    #for each digit labelled i, count how many we correctly classify as i
    accuracy_per_digit = [np.mean(label_pred[label_test == i] == i) for i in digits]

    # Plot as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(digits, accuracy_per_digit, color='skyblue')
    plt.xticks(digits)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Per‐Digit Test Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
