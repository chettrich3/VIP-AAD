import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting_ten.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

keras_wh_train_loss_df = pd.read_csv("keras_wh_train.txt", header=None, names=["training losses"])
keras_wh_train_y = keras_wh_train_loss_df["training losses"].to_numpy()

keras_wh_val_loss_df = pd.read_csv("keras_wh_val.txt", header=None, names=["validation losses"])
keras_wh_val_y = keras_wh_val_loss_df["validation losses"].to_numpy()

keras_hm_train_loss_df = pd.read_csv("keras_hm_train.txt", header=None, names=["training losses"])
keras_hm_train_y = keras_hm_train_loss_df["training losses"].to_numpy()

keras_hm_val_loss_df = pd.read_csv("keras_hm_val.txt", header=None, names=["validation losses"])
keras_hm_val_y = keras_hm_val_loss_df["validation losses"].to_numpy()

keras_reg_train_loss_df = pd.read_csv("keras_reg_train.txt", header=None, names=["training losses"])
keras_reg_train_y = keras_wh_train_loss_df["training losses"].to_numpy()

keras_reg_val_loss_df = pd.read_csv("keras_reg_val.txt", header=None, names=["validation losses"])
keras_reg_val_y = keras_reg_val_loss_df["validation losses"].to_numpy()

pytorch_wh_train_loss_df = pd.read_csv("pytorch_wh_train.txt", header=None, names=["training losses"])
pytorch_wh_train_y = pytorch_wh_train_loss_df["training losses"].to_numpy()

pytorch_wh_val_loss_df = pd.read_csv("pytorch_wh_val.txt", header=None, names=["validation losses"])
pytorch_wh_val_y = pytorch_wh_val_loss_df["validation losses"].to_numpy()

pytorch_hm_train_loss_df = pd.read_csv("pytorch_hm_train.txt", header=None, names=["training losses"])
pytorch_hm_train_y = pytorch_hm_train_loss_df["training losses"].to_numpy()

pytorch_hm_val_loss_df = pd.read_csv("pytorch_hm_val.txt", header=None, names=["validation losses"])
pytorch_hm_val_y = pytorch_hm_val_loss_df["validation losses"].to_numpy()

pytorch_reg_train_loss_df = pd.read_csv("pytorch_reg_train.txt", header=None, names=["training losses"])
pytorch_reg_train_y = pytorch_wh_train_loss_df["training losses"].to_numpy()

pytorch_reg_val_loss_df = pd.read_csv("pytorch_reg_val.txt", header=None, names=["validation losses"])
pytorch_reg_val_y = pytorch_reg_val_loss_df["validation losses"].to_numpy()

fig, ax = plt.subplots(3)

ax[0].plot(x_values, pytorch_wh_train_y, "r", label="PyTorch wh Training Loss")
ax[0].plot(x_values, pytorch_wh_val_y , "r--", label="PyTorch wh Validation Loss")
ax[0].plot(x_values, keras_wh_train_y, "b", label="Keras wh Training Loss")
ax[0].plot(x_values, keras_wh_val_y , "b--", label="Keras wh Validation Loss")
ax[0].set_title('PyTorch and Keras wh Training and Validation Losses')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('wh Loss')
# ax[0].legend(loc="upper right")

ax[1].plot(x_values, pytorch_hm_train_y, "r", label="PyTorch hm Training Loss")
ax[1].plot(x_values, pytorch_hm_val_y, "r--", label="PyTorch hm Validation Loss")
ax[1].plot(x_values, keras_hm_train_y, "b", label="Keras hm Training Loss")
ax[1].plot(x_values, keras_hm_val_y, "b--", label="Keras Validation Loss")
ax[1].set_title('PyTorch and Keras hm Training and Validation Losses')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('hm Loss')
# ax[1].legend(loc="upper right")

ax[2].plot(x_values, pytorch_reg_train_y, "r", label="PyTorch reg Training Loss")
ax[2].plot(x_values, pytorch_reg_val_y, "r--", label="PyTorch reg Validation Loss")
ax[2].plot(x_values, keras_reg_train_y, "b", label="Keras reg Training Loss")
ax[2].plot(x_values, keras_reg_val_y, "b--", label="Keras reg Validation Loss")
ax[2].set_title('PyTorch and Keras reg Training and Validation Losses')
ax[2].set_xlabel('Number of Epochs')
ax[2].set_ylabel('reg Loss')
# ax[2].legend(loc="upper right")

fig.tight_layout()
plt.show()