import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def evaluate_model(encoder, decoder, test_data):
    """Evaluates the encoder-decoder model on test data."""
    encoded_features = encoder.predict(test_data)
    reconstructed = decoder.predict(encoded_features)

    mse = np.mean((test_data - reconstructed) ** 2)
    print(f"Mean Squared Error on Test Data: {mse}")


def compute_recognition_accuracy(encoder, test_data, test_labels):
    """Computes recognition accuracy using Euclidean distance."""
    encoded_features = encoder.predict(test_data)

    distances = euclidean_distances(encoded_features, encoded_features)
    correct = 0
    for i, dists in enumerate(distances):
        predicted_label = test_labels[np.argmin(dists[1:])]  # Exclude self-comparison
        correct += (predicted_label == test_labels[i])

    accuracy = correct / len(test_labels)
    print(f"Recognition Accuracy: {accuracy}")