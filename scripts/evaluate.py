# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances


# def evaluate_model(encoder, decoder, test_data):
#     """Evaluates the encoder-decoder model on test data."""
#     encoded_features = encoder.predict(test_data)
#     reconstructed = decoder.predict(encoded_features)

#     mse = np.mean((test_data - reconstructed) ** 2)
#     print(f"Mean Squared Error on Test Data: {mse}")


# def compute_recognition_accuracy(encoder, test_data, test_labels):
#     """Computes recognition accuracy using Euclidean distance."""
#     encoded_features = encoder.predict(test_data)

#     distances = euclidean_distances(encoded_features, encoded_features)
#     correct = 0
#     for i, dists in enumerate(distances):
#         predicted_label = test_labels[np.argmin(dists[1:])]  # Exclude self-comparison
#         correct += (predicted_label == test_labels[i])

#     accuracy = correct / len(test_labels)
#     print(f"Recognition Accuracy: {accuracy}")




# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
# import matplotlib.pyplot as plt

# def evaluate_model(encoder, decoder, test_data, test_labels=None):
#     """Evaluates the encoder-decoder model on test data and visualizes accuracy."""
#     # Extract feature vectors (encoded features)
#     encoded_features = encoder.predict(test_data)
    
#     # Reconstruct the data to measure MSE
#     reconstructed = decoder.predict(encoded_features)
#     mse = np.mean((test_data - reconstructed) ** 2)
#     print(f"Mean Squared Error on Test Data: {mse}")

#     accuracies = []
#     if test_labels is not None:
#         # Compute recognition accuracy using Euclidean distance
#         accuracy = compute_recognition_accuracy(encoded_features, test_labels)
#         accuracies.append(accuracy)
#         print(f"Recognition Accuracy: {accuracy}")

#     # Plot the accuracies over epochs (or as they come)
#     if accuracies:
#         plt.plot(accuracies, label="Test Set Accuracy")
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.title("Test Set Accuracy")
#         plt.show()

#     return accuracies

# def compute_recognition_accuracy(encoded_features, test_labels):
#     """Computes recognition accuracy using Euclidean distance."""
#     distances = euclidean_distances(encoded_features, encoded_features)
#     correct = 0
#     for i, dists in enumerate(distances):
#         # Exclude self-comparison by starting from index 1
#         predicted_label = test_labels[np.argmin(dists[1:]) + 1]  # Skip self, +1 to adjust index
#         correct += (predicted_label == test_labels[i])
    
#     accuracy = correct / len(test_labels)
#     return accuracy






import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def evaluate_model(encoder, decoder, test_data, test_labels=None):
    """Evaluates the encoder-decoder model on test data and visualizes accuracy."""
    # Extract feature vectors (encoded features)
    encoded_features = encoder.predict(test_data)
    
    # Reconstruct the data to measure MSE
    reconstructed = decoder.predict(encoded_features)
    mse = np.mean((test_data - reconstructed) ** 2)
    print(f"Mean Squared Error on Test Data: {mse}")

    accuracies = []
    if test_labels is not None:
        # Compute recognition accuracy using Euclidean distance
        accuracy = compute_recognition_accuracy(encoded_features, test_labels)
        accuracies.append(accuracy)
        print(f"Recognition Accuracy: {accuracy}")

    return accuracies  # Return accuracies for later plotting

def compute_recognition_accuracy(encoded_features, test_labels):
    """Computes recognition accuracy using Euclidean distance."""
    distances = euclidean_distances(encoded_features, encoded_features)
    correct = 0
    for i, dists in enumerate(distances):
        # Exclude self-comparison by starting from index 1
        predicted_label = test_labels[np.argmin(dists[1:]) + 1]  # Skip self, +1 to adjust index
        correct += (predicted_label == test_labels[i])
    
    accuracy = correct / len(test_labels)
    return accuracy