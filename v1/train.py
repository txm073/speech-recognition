import os
import librosa
from neuralnet import build_model
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from preprocess import one_hot_encode
import time

def load_batch(base_path, batch):
    x, y = [], []
    for file in batch:
        data = np.load(base_path + "\\" + file, allow_pickle=True)["a"]
        x.append(data[0].reshape(128, 251, 1))
        y.append(one_hot_encode(data[1]))
    return np.array(x), np.array(y)

def load_file(filepath):
    data = np.load(filepath, allow_pickle=True)["a"]
    return data[0].reshape(1, 128, 251, 1), data[1]

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

def train(dataset_path, epochs=40, batch_size=32):
    files = os.listdir(dataset_path + "\\arrays")
    remainder = len(files) % batch_size
    files = files[:-remainder]
    
    n_batches = len(files) // batch_size
    batches = np.array_split(files, n_batches)

    return load_batch(dataset_path + "\\arrays", batches[2])[1][0].shape

    split = int(n_batches * 0.9)
    training_batches = batches[:split]
    test_batches = batches[split:]

    for epoch in range(epochs): 
        start = time.time()
        print(f"\nStarting epoch {epoch + 1} / {epochs}...")
        for b in tqdm(training_batches):
            x_batch, y_batch = load_batch(dataset_path + "\\arrays", b)
            loss = train_step(x_batch, y_batch)

        accuracy = train_acc.result()
        print(f"Training accuracy over epoch: {accuracy}")
        train_acc.reset_states()

        for b in testing_batches:
            x_batch, y_batch = load_batch(dataset_path + "\\arrays", b)
            test_step(x_batch, y_batch)

        accuracy = test_acc.result()
        print(f"Accuracy on validation data: {accuracy}")
        test_acc.reset_states()
        print(f"Time taken to complete epoch: {time.time() - start} seconds")

    print("Training complete!")
    model.save("model.h5")
    print("Model saved!")
    

if __name__ == "__main__":

    model = build_model((128, 251, 1), 1e-3)

    loss_fn = CategoricalCrossentropy(from_logits=True)
    
    #x, y = load_file("C:\\Users\\User\\Desktop\\Common Voice\\arrays\\000a24b376ea168f37fffb466429b9aef246bd14b99c12b85c7f7f3d6b594d223fd876243ff441e880c6f38713a992dc6994c3844cf3fb17e8589485e10b8144.npz")

    output = train("C:\\Users\\User\\Desktop\\Common Voice")
    print(output)