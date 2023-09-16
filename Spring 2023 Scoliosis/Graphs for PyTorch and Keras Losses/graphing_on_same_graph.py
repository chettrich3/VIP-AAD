import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

pytorch_train_loss_df = pd.read_csv("train_loss_pytorch.txt", header=None, names=["training losses"])
pytorch_train_y = pytorch_train_loss_df["training losses"].to_numpy()

pytorch_val_loss_df = pd.read_csv("val_loss_pytorch.txt", header=None, names=["validation losses"])
pytorch_val_y = pytorch_val_loss_df["validation losses"].to_numpy()

keras_train_loss_df = pd.read_csv("train_loss_keras.txt", header=None, names=["training losses"])
keras_train_y = keras_train_loss_df["training losses"].to_numpy()

keras_val_loss_df = pd.read_csv("val_loss_keras.txt", header=None, names=["validation losses"])
keras_val_y = keras_val_loss_df["validation losses"].to_numpy()

fig, ax = plt.subplots(2)

ax[0].plot(x_values, pytorch_train_y, "r", label="PyTorch Training Loss")
ax[0].plot(x_values, keras_train_y, "b", label="Keras Training Loss")
ax[0].set_title('PyTorch and Keras Training Losses')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Training Loss')
ax[0].legend(loc="upper right")

ax[1].plot(x_values, pytorch_val_y, "r--", label="PyTorch Validation Loss")
ax[1].plot(x_values, keras_val_y, "b--", label="Keras Validation Loss")
ax[1].set_title('PyTorch and Keras Validation Losses')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Validation Loss')
ax[1].legend(loc="upper right")

fig.tight_layout()
plt.show()