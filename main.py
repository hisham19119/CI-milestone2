# from scripts.preprocess import split_dataset
# from scripts.train import train_model, load_data, cross_validate
# from scripts.evaluate import evaluate_model
# from models.autoencoder import convolutional_autoencoder, vanilla_autoencoder, variational_autoencoder
# from pathlib import Path
# import matplotlib.pyplot as plt

# # Step 1: Preprocess and split dataset
# input_dir = './data/CASIA'
# output_dir = './data/split'
# split_dataset(input_dir, output_dir)

# # Step 2: Load training and validation data
# train_data = load_data('./data/split/train')
# val_data = load_data('./data/split/val')

# # Step 3: Cross-Validation
# configs = [{'type': 'vanilla'}, {'type': 'convolutional'}, {'type': 'variational'}]

# def model_selector(config):
#     if config['type'] == 'vanilla':
#         return vanilla_autoencoder((64, 64, 3))
#     elif config['type'] == 'convolutional':
#         return convolutional_autoencoder((64, 64, 3))
#     elif config['type'] == 'variational':
#         return variational_autoencoder((64, 64, 3))

# best_config = cross_validate(train_data, val_data, model_selector, configs)

# # Step 4: Train with Best Config
# encoder, decoder = model_selector(best_config)
# history = train_model(encoder, decoder, train_data, val_data, epochs=20)

# # Visualize training process
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Step 5: Evaluate model
# test_data = load_data('./data/split/test')
# evaluate_model(encoder, decoder, test_data)





# from scripts.preprocess import split_dataset
# from scripts.train import train_model, load_data, cross_validate
# from scripts.evaluate import evaluate_model
# from models.autoencoder import convolutional_autoencoder, vanilla_autoencoder, variational_autoencoder
# from pathlib import Path
# import matplotlib.pyplot as plt

# # Step 1: Preprocess and split dataset
# input_dir = './data/CASIA'
# output_dir = './data/split'
# split_dataset(input_dir, output_dir)

# # Step 2: Load training and validation data
# train_data = load_data('./data/split/train')
# val_data = load_data('./data/split/val')

# # Step 3: Cross-Validation
# configs = [{'type': 'vanilla'}, {'type': 'convolutional'}, {'type': 'variational'}]

# def model_selector(config):
#     if config['type'] == 'vanilla':
#         return vanilla_autoencoder((64, 64, 3))
#     elif config['type'] == 'convolutional':
#         return convolutional_autoencoder((64, 64, 3))
#     elif config['type'] == 'variational':
#         return variational_autoencoder((64, 64, 3))

# best_config = cross_validate(train_data, val_data, model_selector, configs)

# # Step 4: Train with Best Config
# encoder, decoder = model_selector(best_config)
# history = train_model(encoder, decoder, train_data, val_data, epochs=20)

# # Visualize training process: loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title("Training and Validation Loss")
# plt.show()

# # Step 5: Evaluate model
# test_data = load_data('./data/split/test')
# test_labels = test_data.classes  # Get the labels from the test data

# # Evaluate model and compute recognition accuracy
# accuracy = evaluate_model(encoder, decoder, test_data, test_labels)

# # Visualize test set accuracy
# plt.plot(accuracy, label='Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title("Test Accuracy Over Epochs")
# plt.show()






from scripts.preprocess import split_dataset
from scripts.train import train_model, load_data, cross_validate
from scripts.evaluate import evaluate_model
from models.autoencoder import convolutional_autoencoder, vanilla_autoencoder, variational_autoencoder
from pathlib import Path
import matplotlib.pyplot as plt

# Step 1: Preprocess and split dataset
input_dir = './data/CASIA'
output_dir = './data/split'
split_dataset(input_dir, output_dir)

# Step 2: Load training and validation data
train_data = load_data('./data/split/train')
val_data = load_data('./data/split/val')

# Step 3: Cross-Validation
configs = [{'type': 'vanilla'}, {'type': 'convolutional'}, {'type': 'variational'}]

def model_selector(config):
    if config['type'] == 'vanilla':
        return vanilla_autoencoder((64, 64, 3))
    elif config['type'] == 'convolutional':
        return convolutional_autoencoder((64, 64, 3))
    elif config['type'] == 'variational':
        return variational_autoencoder((64, 64, 3))

best_config = cross_validate(train_data, val_data, model_selector, configs)

# Step 4: Train with Best Config
encoder, decoder = model_selector(best_config)
history = train_model(encoder, decoder, train_data, val_data, epochs=20)

# Step 5: Visualize training process: loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  # Create a subplot for loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss")

# Step 6: Evaluate model
test_data = load_data('./data/split/test')
test_labels = test_data.classes  # Get the labels from the test data

# Evaluate model and compute recognition accuracy
accuracies = evaluate_model(encoder, decoder, test_data, test_labels)

# Step 7: Visualize test set accuracy
plt.subplot(1, 2, 2)  # Create a subplot for accuracy
plt.plot(accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Test Accuracy Over Epochs")

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()