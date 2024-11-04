import os

# Get the directory where mnist.py is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

import pickle

from torchvision import datasets


def download_mnist():
    # Downloads to './data' by default
    train_dataset = datasets.MNIST("./data", train=True, download=True)
    test_dataset = datasets.MNIST("./data", train=False, download=True)

    # Convert to numpy arrays in the same format as the original code
    train_images = train_dataset.data.numpy().reshape(-1, 28 * 28)
    train_labels = train_dataset.targets.numpy()
    test_images = test_dataset.data.numpy().reshape(-1, 28 * 28)
    test_labels = test_dataset.targets.numpy()

    # Save in the same format as the original code
    mnist = {
        "training_images": train_images,
        "training_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    # Use absolute path for saving
    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def load():
    # Use absolute path for loading
    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    with open(pkl_path, "rb") as f:
        mnist = pickle.load(f)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


if __name__ == "__main__":
    download_mnist()
