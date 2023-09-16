import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

train_loss_df = pd.read_csv("train_loss_pytorch.txt", header=None, names=["training losses"])
train_y = train_loss_df = train_loss_df["training losses"].to_numpy()

val_loss_df = pd.read_csv("val_loss_pytorch.txt", header=None, names=["validation losses"])
val_y = val_loss_df = val_loss_df["validation losses"].to_numpy()

fix, ax = plt.subplots()

ax.plot(x_values, train_y, "r", label="Training Loss")
ax.plot(x_values, val_y, "r--", label="Validation Loss")
ax.set_title('PyTorch Losses')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Loss')
leg = ax.legend()

plt.show()