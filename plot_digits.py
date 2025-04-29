# -------------------------------
# Per‐digit test accuracy bar chart
# -------------------------------

from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

'''
# load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
images = mnist.data.astype(np.float32)
labels = mnist.target.astype(np.int64)

# Normalize the data
images /= 255.0

#create a test set
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)
'''
#load the mnist numbers dataset
#KEEP THE SEED AT 1234
read_config = tfds.ReadConfig(shuffle_seed=1234)
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    read_config=read_config,
)

def plot_digits(weights):
    """this function takes in a list of weights and graphs per-digit accuracy on ds_test"""

    # 1) build & set up your model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(28, activation="relu"),
        tf.keras.layers.Dense(10),
    ])
    model.set_weights(weights)

    # 2) extract ALL images & labels from ds_test into numpy arrays
    #    (you could do this once outside the function if you call it repeatedly)
    images_list = []
    labels_list = []
    for img, lbl in ds_test:
        images_list.append(img.numpy())
        labels_list.append(lbl.numpy())
    images_test = np.stack(images_list)   # shape (10000, 28, 28, 1)
    labels_test = np.array(labels_list)   # shape (10000,)

    # 3) run your predictions
    probs      = model.predict(images_test)                # (10000, 10)
    label_pred = np.argmax(probs, axis=1)                  # (10000,)

    # 4) compute per-digit accuracy
    digits = np.arange(10)
    accuracy_per_digit = [
        np.mean(label_pred[labels_test == d] == d)
        for d in digits
    ]

    # 5) plot
    plt.figure(figsize=(8, 6))
    plt.bar(digits, accuracy_per_digit, color="skyblue")
    plt.xticks(digits)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Per‐Digit Test Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion(weights):
    """this function takes in a list of weights and graphs the confusion matrix on ds_test"""

    # 1) build & set up your model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(28, activation="relu"),
        tf.keras.layers.Dense(10),
    ])
    model.set_weights(weights)

    # 2) extract ALL images & labels from ds_test into numpy arrays
    images_list, labels_list = [], []
    for img, lbl in ds_test:
        images_list.append(img.numpy())
        labels_list.append(lbl.numpy())
    images_test = np.stack(images_list)   # (num_samples, 28, 28, 1)
    labels_test = np.array(labels_list)   # (num_samples,)

    # 3) run your predictions
    probs      = model.predict(images_test)  # (num_samples, 10)
    label_pred = np.argmax(probs, axis=1)    # (num_samples,)

    # 4) compute confusion matrix
    digits = np.arange(10)
    cm = confusion_matrix(labels_test, label_pred, labels=digits)

    # 5) plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    ax.invert_yaxis()
    ax.set_title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.show()