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

# Visualize training process
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 5: Evaluate model
test_data = load_data('./data/split/test')
evaluate_model(encoder, decoder, test_data)