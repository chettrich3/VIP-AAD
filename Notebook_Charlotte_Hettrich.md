# Team Member

**Team Member:** Charlotte Hettrich

**Email:** chettrich3@gatech.edu

**Cell Phone:** 678-350-3033

**Interests:** Python, Football - Buffalo Bills, Reading, Traveling, and Hiking

**Sub-team:** Scoliosis

<img src="files/chettrich3/front_page_chettrich3.jpg" alt="drawing" width="300"/>

# 2023 Spring Notebook Entries

<details>
  <summary><b>2023 Spring Notebook Entries</b></summary>

## 24 April 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, April 24th

* We need to finalize our notebooks by Sunday.
* We need to visualize the data after evaluating by Sub-Team meeting on Thursday.
* Lastly, Minseung, Daniel, and need to make a slide based on evaluating graph and previous loss graph.
  * We can each take a graph in the slides.

### Meeting with Minseung Wednesday, April 26th

* We assigned slides for the Final Presentation Friday, April 28th.
* I did a lot of experimenting with the graphs during our meeting today. Originally I had 4 different plots as a 2 x 2 subplot, but I found by doing research online that the validation and training losses need to be modeled on the same graph for each model.
* I ended up making one graph with everything on it, both Keras training and validation losses as well as PyTorch training and validation losses. 
* Also, I made a 1 x 2 subplot with the PyTorch and Keras training losses on one plot, and the PyTorch an Keras validation losses on the second plot.
* In our meeting, we discussed what the graphs yielded and the different issues present within the plots. Specifically with the PyTorch losses being much lower than the Keras losses.
* Daniel things this is because something is converted wrong in a certain place, but it is hard to debug. PyTorch is better suited for debugging.

### Scoliosis Sub-Team Meeting Thursday, April 27th

* Finished making sure that the slides were assigned.
* Most people have included their name under the slides.
* Want to try to get a screenshot of what it looks like when we run NNLearner. There is still some stuff that folks need to iron out on the training end.
* Might need to change the axis titles depending on what exact loss metrics we have. Minseung, Daniel, and I need to figure out what loss metrics we are using.
  * The FocalLoss for "hm" and RegL1Loss for "wh" and "reg" are summed up to get the final loss for both PyTorch and Keras. Those are standard loss metrics. Should be able to explain at a high level how the losses are calculated.
  * Three total losses: 1 FocalLoss and 2 RegLoss.
* Daniel is running 10 epochs to see what a separation of losses will show with the PyTorch and Keras models.

[**Scoliosis Final Presentation Spring 2023**](https://docs.google.com/presentation/d/1PyhmQvF3Y53qIJXIGeG8v7GHpvdVqYhZPSQS6dpgAs4/edit?usp=sharing)

</details>

<details>
  <summary><b>Final Updates on Implementing the Keras Model</b></summary>

### Final Updates on Keras Model for the Final Presentation and Graphing

* Daniel finished converting the custom <code>loss.py</code> file into Keras. He passed this over to me, so that I was able to graph the Keras losses.
* He found that in each epoch loop, instead of calling <code>scheduler.step()</code>, <code>scheduler(epoch)</code> needs to be called. The Keras model successfully goes through epochs now. The losses are off compared to PyTorch. 
* Once he finished training the model, he sent me the losses which I then worked on plotting in the sections below.

</details>

<details>
  <summary><b>Graphing the Keras Model Losses</b></summary>

### Graphing the Keras Model Losses

* I used the same process for graphing the PyTorch losses for graphing the Keras losses.
* I created dataframes to read in the files and then translated them to numpy arrays to later plot in a subplot. Pandas can get problematic when making plots side by side and using the same dataframe twice. I ended up creating a few dataframes.
  * The <code>counting_df</code> is a dataframe of numbers 1-50 to use as x-values within the plots.
  * The <code>train_loss_df</code> is a dataframe of the training loss numbers.
  * The <code>val_loss_df</code> is a dataframe of the validation loss numbers.
* I transitioned to using numpy which is more basic than pandas, but the plots look better and there is more capability with what we want to do.
* I looked up how exactly loss is supposed to be graphed, and both the training and the validation loss should be on one graph. Therefore, I have the training loss as a solid blue line, and the validation loss as a dotted blue line. Keras is blue to differentiate from the red of PyTorch because eventually I will plot the PyTorch and Keras losses on the same plot.
* I had to use the subplot method in order to get an appropriate legend for the graphs, and easier organization.
* In the section below is the code which has been copied and pasted into Azure as well. In the section after the code is an image of the plots.

### Code for Creating the Keras Losses Graph

  ```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

train_loss_df = pd.read_csv("train_loss_keras.txt", header=None, names=["training losses"])
train_y = train_loss_df = train_loss_df["training losses"].to_numpy()

val_loss_df = pd.read_csv("val_loss_keras.txt", header=None, names=["validation losses"])
val_y = val_loss_df = val_loss_df["validation losses"].to_numpy()

fix, ax = plt.subplots()

ax.plot(x_values, train_y, "b", label="Training Loss")
ax.plot(x_values, val_y, "b--", label="Validation Loss")
ax.set_title('Keras Losses')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Loss')
leg = ax.legend()

plt.show()
  ```

### Keras Losses Graph
  
  ![Keras Losses](https://github.gatech.edu/storage/user/60851/files/4b633e1b-ef44-4ab9-b996-c0451a806543)

### What does the Keras Losses Graph Mean?

* Looking specifically at the Keras Losses graph in the section above, the training loss looks okay although it does not continue to decrease after around 2-3 epochs.
  * A refresher on epochs, an epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. Another way to define an epoch is the number of passes a training dataset takes around an algorithm.
* Also, the validation loss is lower than the training loss almost throughout the entirety of the graph with an exception at around 18 epochs.
* I believe the Keras graph is unrepresentative of the dataset.
  * One is that since the training curve looks okay, and the validation function moves noisily. This could mean that the validation data is scarce and not very representative of the training data. Therefore, the model struggles to model these examples which is unrepresentative of the dataset.
  * Two, since the validation loss is lower than the training loss, that reflects that the validation dataset is easier to predict than the training dataset. An explanation could be that the validation data is scarce but widely represented by the training dataset, so the model performs well on the few examples.

</details>

<details>
  <summary><b>Graphing the PyTorch and Keras Models Jointly</b></summary>

### Notes on Graphing the PyTorch and Keras Losses Jointly

* I created dataframes to read in the files and then translated them to numpy arrays to later plot in a subplot. Pandas can get problematic when making plots side by side and using the same dataframe twice. I ended up creating a few dataframes.
  * The <code>counting_df</code> is a dataframe of numbers 1-50 to use as x-values within the plots.
  * The <code>pytorch_train_loss_df</code> is a dataframe of the training loss numbers for the PyTorch model.
  * The <code>pytorch_val_loss_df</code> is a dataframe of the validation loss numbers for the PyTorch model.
  * The <code>keras_train_loss_df</code> is a dataframe of the training loss numbers for the Keras model.
  * The <code>keras_val_loss_df</code> is a dataframe of the validation loss numbers for the Keras model.
* I transitioned to using numpy which is more basic than pandas, but the plots look better and there is more capability with what we want to do.
* I looked up how exactly loss is supposed to be graphed, and both the training and the validation loss should be on one graph. Therefore, I have the training loss for Keras as a solid blue line, and the validation loss for Keras as a dotted blue line. Keras is blue to differentiate from the red of PyTorch. Similarly, the training loss for PyTorch is depicted as a solid red line and the validation loss for PyTorch is a dotted red line.
* I had to use the subplot method in order to get an appropriate legend for the graphs, and easier organization.
* In the section below is the code which has been pasted into Azure as well. In the section after the code is images of the plots.
* In the first section below is a graph of the PyTorch and Keras losses on a single plot. This is to see everything together. However, it almost looks like they are in different plots with how much better the PyTorch losses are compared to the Keras losses.
* In the second section, in order to compare the training losses of both the PyTorch and Keras models and the validation losses of both the PyTorch and Keras models separately I made it into subplots.
  * Suplots are easy to manage within the <code>matplotlib.pyplot</code> by splitting plots into <code>fig, ax = plt.subplots(2)</code>.

### Code for Graphing the PyTorch and Keras Losses on a Single Plot

  ```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

pytorch_train_loss_df = pd.read_csv("train_loss_pytorch.txt", header=None, names=["training losses"])
pytorch_train_y = pytorch_train_loss_df = pytorch_train_loss_df["training losses"].to_numpy()

pytorch_val_loss_df = pd.read_csv("val_loss_pytorch.txt", header=None, names=["validation losses"])
pytorch_val_y = pytorch_val_loss_df = pytorch_val_loss_df["validation losses"].to_numpy()

keras_train_loss_df = pd.read_csv("train_loss_keras.txt", header=None, names=["training losses"])
keras_train_y = keras_train_loss_df = keras_train_loss_df["training losses"].to_numpy()

keras_val_loss_df = pd.read_csv("val_loss_keras.txt", header=None, names=["validation losses"])
keras_val_y = keras_val_loss_df = keras_val_loss_df["validation losses"].to_numpy()

fig, ax = plt.subplots()

ax.plot(x_values, pytorch_train_y, "r", label="PyTorch Training Loss")
ax.plot(x_values, pytorch_val_y, "r--", label="PyTorch Validation Loss")
ax.plot(x_values, keras_train_y, "b", label="Keras Training Loss")
ax.plot(x_values, keras_val_y, "b--", label="Keras Validation Loss")
ax.set_title('PyTorch and Keras Losses')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Loss')
leg = ax.legend()

fig.tight_layout()
plt.show()
  ```

### Graph of the PyTorch and Keras Losses on a Single Plot
 
  ![PyTorch and Keras Losses](https://github.gatech.edu/storage/user/60851/files/ac55b6d3-8ba8-4baf-804e-25926e2e3967)

### Code for Graphing a Subplot of Keras and PyTorch Training Losses and Keras and PyTorch Validation Losses

  ```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

counting_df = pd.read_csv("counting.txt", header=None, names=["count"])

x_values = counting_df["count"].to_numpy()

pytorch_train_loss_df = pd.read_csv("train_loss_pytorch.txt", header=None, names=["training losses"])
pytorch_train_y = pytorch_train_loss_df = pytorch_train_loss_df["training losses"].to_numpy()

pytorch_val_loss_df = pd.read_csv("val_loss_pytorch.txt", header=None, names=["validation losses"])
pytorch_val_y = pytorch_val_loss_df = pytorch_val_loss_df["validation losses"].to_numpy()

keras_train_loss_df = pd.read_csv("train_loss_keras.txt", header=None, names=["training losses"])
keras_train_y = keras_train_loss_df = keras_train_loss_df["training losses"].to_numpy()

keras_val_loss_df = pd.read_csv("val_loss_keras.txt", header=None, names=["validation losses"])
keras_val_y = keras_val_loss_df = keras_val_loss_df["validation losses"].to_numpy()

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
  ```

### Graph of the Subplot of Keras and PyTorch Training Losses and Keras and PyTorch Validation Losses
  
  ![PyTorch and Keras Losses on Same Graph](https://github.gatech.edu/storage/user/60851/files/988f4b7d-e33c-47ed-ad24-00ac9291ef88)

### Comparison Between PyTorch and Keras Graphs Based on the Graphs in Above Sections

* An immediate tell between the two graphs is that the PyTorch model losses are less than the Keras model losses.
* Even with the training losses and validation losses for both models in separate graphs, the PyTorch model performs significantly better with regards to loss.
* When looking individually at the PyTorch and Keras models of training and validation losses in different ways they are an unrepresentative validation of the dataset. Maybe if we are able to make the PyTorch model more representative of the dataset, we can make the Keras model representative by changing the conversions accordingly.
* Another option is use what we know from the PyTorch model and just focus on the Keras model to make the model more representative of the data.
* A representative dataset reflects proportionally statistical characteristics in another dataset from the same domain. 
  * We could find the training dataset is not representative in relation to the validation dataset and vice-versa.

</details>

<details>
  <summary><b>Separating out the Loss Metrics for PyTorch and Keras</b></summary>

### What are our Loss Metrics for the PyTorch and Keras Models?

* FocalLoss used for “hm” (heatmap)
  * Focal Loss (FL) is an improved version of Cross-Entropy Loss (CE)  that tries to handle the class imbalance problem by assigning more weights to hard or easily misclassified examples.
* RegL1Loss used for “wh” (landmarks) and “reg” (corner offset)
  * L1 is Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.
  * Lasso shrinks the less important features coefficient to zero, removing some features altogether. 
  * Works well for feature selection because we have a huge number of features.

[Focal Loss](https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/)
[Regularization Model Loss](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

### What does Separated Losses Mean?

* During our weekly Scoliosis Sub-Team Meeting Thursday, April 27th, Eric asked what exactly our loss metrics were. Earlier in our dataset cleanup we used SMAPE and MSE as loss metrics. After discussing and doing some research I found that our loss metrics are FocalLoss and RegL1Loss. More information about our loss metrics are listed in the section above.
* We split up the three losses that we have into three separate graphs for comparison. FocalLoss is used for "hm" which is heatmap. RegL1Loss is used for "wh" which are the landmarks as well as "reg" which is the corner offset.
* This is for us to get a deeper look at what is not consistent between the PyTorch and Keras models. Daniel was able to separate the losses into those three categories. Since we were assigned this task April 27th and the presentation is April 28th, he could only train the model for 10 epochs. If we had more time we would be able to get the full 50 epochs. This could be a great future task as well as debugging and triple checking the conversions.
* In the sections below I will explain more in depth about the results in the graphs from separating the losses out into three sections.

### PyTorch Separated Losses

* I used a similar method to graphing the summed up losses for the PyTorch and Keras models. The code for graphing the PyTorch separated losses are in my Personal Scoliosis Repository as well as in my folder in Azure.
* The PyTorch model looks pretty good in the "wh" landmarks, and "hm" heatmaps for the most part. However, near the end in the heatmap graph, the validation loss increases for a few epochs.
* The "reg" corner offset graph is where we start to have issues with the large space between the training and validation loss. Also, the validation loss is small and not decreasing, which does not seem right.

  ![PyTorch Losses Separated](https://github.gatech.edu/storage/user/60851/files/5e723eca-2055-427c-82bc-92df3c4fd90f)

[Personal Scoliosis Repository](https://github.gatech.edu/chettrich3/chettrich3Scoliosis.git)

### Keras Separated Losses

* I used a similar method to graphing the summed up losses for the PyTorch and Keras models. The code for graphing the Keras separated losses are in my Personal Scoliosis Repository as well as in my folder in Azure.
* Within the Keras model, the "wh" landmarks, and "reg" corner offset seem to have lower values than the "hm" heatmap losses. Heatmap loss might be where the conversions are off from the PyTorch to Keras model.
* There are discrepencies across the board as the training loss is much higher than the validation loss which leads me to believe that the Keras model is an unrepresentative validation of the data.

  ![Keras Losses Separated](https://github.gatech.edu/storage/user/60851/files/75c901ac-5554-42a3-91b2-3cfbab6ccb3e)

[Personal Scoliosis Repository](https://github.gatech.edu/chettrich3/chettrich3Scoliosis.git)

### PyTorch and Keras Separated Losses Together

* I used a similar method to graphing the summed up losses for the PyTorch and Keras models. The code for graphing the PyTorch and Keras separated losses altogether is in my Personal Scoliosis Repository as well as in my folder in Azure.
* Below are the graphs combined with the PyTorch and Keras models with the separated losses. They are both the same graph, one without a legend and one with a legend. The graph with the legend seems distorted because in order for the legend to not cover most of the graph it has to be blown up, so I included one without the legend. PyTorch is in red with training loss as a solid line and validation loss as the dotted line. The same goes for the Keras model, but it is blue instead to differentiate from the PyTorch model.
* When comparing the graphs it is obvious which is the Keras model and which is PyTorch just based on the previous loss graphs without separating the losses. Again I believe both models to be a mostly unrepresentative validation of the data. However, now we have direction on where to start debugging conversions or even where to look within the PyTorch model.
 
  ![PyTorch and Keras Separated Losses without Legend](https://github.gatech.edu/storage/user/60851/files/947cdc47-f471-4ee5-84e8-ea3d301c5d66)
  ![PyTorch and Keras Separated Losses](https://github.gatech.edu/storage/user/60851/files/011552d3-c31e-4d1c-bfa7-589a20c3784d)

</details>

<details>
  <summary><b>Script for Final Presentation</b></summary>

### Script for the Slide on PyTorch to Keras Conversions

* I worked with Minseung and Daniel and converting the VFL Model from PyTorch to Keras.
* Keras is a neural network platform that runs on top of the open-source library TensorFlow, and PyTorch is a framework for building deep models.
* The VFL model has four PyTorch layers:
  * <code>nn.Conv2d</code>
  * <code>nn.BatchNorm2d</code>
  * <code>nn.ReLU</code>
  * <code>nn.MaxPool2d</code>
* Most of the parameters of the Keras equivalents are similar with a few key differences.
  * Passing as a parameter is not as flexible in Keras, so the <code>ZeroPadding2D()</code> layer must be used to have integer or tuple padding.
  * Keras is by default <code>channels last</code> while PyTorch is by default channels first, so that has to be explicitly specified.
* We also created a testing script to verify that comparison match.
  * The script takes in a random numpy array of any dimension, and feeds it to a Keras and a PyTorch model.
  * The output at the end is compared and checked to see that there is a small difference between them.
  * Both models must be in the same mode, PyTorch by default is in training mode, and Keras is by default in eval mode.

### Script for the Slide Explaining about the Loss Metrics

* FocalLoss used for “hm” (heatmap)
  * Focal Loss (FL) is an improved version of Cross-Entropy Loss (CE)  that tries to handle the class imbalance problem by assigning more weights to hard or easily misclassified examples.
* RegL1Loss used for “wh” (landmarks) and “reg” (corner offset)
  * L1 is Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.
  * Lasso shrinks the less important features coefficient to zero, removing some features altogether. 
  * Works well for feature selection because we have a huge number of features.

### Script for the Slide Comparing the PyTorch and Keras Models Losses without Separation

* Now looking at both the PyTorch and Keras training and validation losses altogether. It almost looks like the graph on the left is a subplot because the PyTorch model losses are almost half that of the Keras losses. This could happen because of a variety of things, normalization, incorrect conversions, the possibilities are endless with what could be wrong. This would definitely be worth looking into for next semester.
* Other than the fact that the PyTorch losses are significantly lower than the Keras losses. They are both are an unrepresentative validation of the datasets. PyTorch because after around 10 epochs the validation loss starts to increase rather than decrease, and Keras because the validation loss is lower than the training loss with an exception of around 18 epochs for a spike.
* Next steps would be to examine both models to see what is causing these losses. I would double check the conversions for the Keras model. Also, evaluating both models or getting them prepped for evaluation would be helpful in providing more metrics.

[Scoliosis Final Presentation Spring 2023](https://docs.google.com/presentation/d/1PyhmQvF3Y53qIJXIGeG8v7GHpvdVqYhZPSQS6dpgAs4/edit?usp=sharing)

</details>

<details>
  <summary><b>VIP Final Presentation Notes</b></summary>

### Scoliosis - My Sub-Team

* A doctor and Cole attended our Scoliosis presentation and had questions regarding our model and why we only have 17 vertebrae. We only have 17 because of the cropping and the fact that the AASCE model only has landmark truth data for 17 vertebrae.
* Our model currently predicts at about 5% error with a SMAPE value of around 11. The idea is to divide the SMAPE by 2 and that is around the percentage of error which is about 2 degrees off.
* Future direction would be to iron out the discrepancies between the PyTorch and Keras model that my sub-sub-team mainly worked on. This is due to the large differences in loss. Also, we would need to iron out the discrepancies between the Keras implementation in EMADE and the Keras model implemented from the PyTorch to Keras conversions.

### DevOps

* Sent out a survey and got user opinions of EMADE.
* A video walkthrough of the EMADE installation process would be very useful.
* Created a centralized repository for all DevOps related materials.
* For Windows, one of the members made a <code>conda</code> environment and installed EMADE on that in the video.
* Got EMADE working on M1 Mac, and created a guide for how to install it on Mac.
* File transfer guide from PACE ICE.
* Made a Git Guide on how to use repositories, installation, and authentication.
  * Short explanation of what each command does.
* Created an installation script for M1 tensorflow installation.
  * Set tensorflow dependencies before installing the rest of the dependencies.
* They want to improve existing unit tests.
* A major goal is to create a one click script for installation of EMADE. A downside of this is students would not struggle and learn terminal commands. I can second this because I learned a lot about SQL and the terminal during the EMADE installation.

### Image Processing

* Problem is to classify 14 different diseases with chest x-rays.
* Multi Label vs. Multi Classes
  * Multi class: when there is only one label for an image.
  * Multi label: an image that might have multiple diseases.
* Tried to incorporate transfer learning, which is an optimization that allows for rapid progress with learning from itself.
* They fixed overfitting with augmentation.
  * Make sure that the weights are constrained with L2 Regularization.
  * Another problem was unbalanced class weights, no disease was the largest.
  * Fixed the loss function from binary cross entropy to weighted binary cross entropy.
  * The changes made a difference.
  * In the future, adding more models would be very helpful.
* Wants to fix some of the weighted loss in the future, want to try ResNet as well.
* Created NNLearnerTL
  * Created a preprocessing layer list in addition.
* New primitives have been added for preprocessing specifically Keras preprocessing layers.
* Did not disrupt the main NNLearner method, making it safe to develop on for other teams.
* Worked on bug fixing to get NNLearnerTL trees to evaluate. One of the members thinks there is some kind of limit.
* Analyzed the best, worst, and average AUC (area under the curve) for every class for primitive analysis.
* Worked on fast failing, hard coded into <code>neural_network_methods.py</code>.
* At midterms working on FastFailing table and columns for loss values.
* Current errors in NNLearner for the VIP. originally tried to check if there is a seeded Pareto front. In actuality, the seeds would not become None, the list was empty.
* Lexicase selection has also been worked on. Lexicase is a selection method designed to help solve highly modal problems. EMADE has an implementation of Lexicase, but that implementation used objective scores.
* Next semester hope new members will do data mining and optimize runs.

### Stocks

* Price trend prediction overview goal is to use lookback windows to predict positive or negative trends in a given stocks price.
* Tried to scrape Twitter, but they closed their API to prevent it from working.
* Scraped Reddit API after midterms to add to the dataset. However, there was no way to order the data historically.
* TA-Lib stands for the Technical Analysis Library which contains many functions to help process stock data.
* They did NNLearner mutations tests on stocks data. Mutations do all operations in a try and except block so if they fail, they return the individual without mutation.
* They made steps towards replication of "Deep Learning for Portfolio Optimization". Did not know the exact dimensions, so there were a few issues.
* Fixed genNNLearner and individual generation. There were issues as a part of a process, but they have been fixed.
* Progress on EMADE runs! Launched a process with reuse flag 0, first generating a bunch of garbage individuals and then stop EMADE, run the seeding script, and then run the <code>launchEMADE.py</code> with reuse set to 1.
* Goal was to clean up the environments within the nn-vip branch for Stocks. First had authentication issues for running on new members laptops. Also, set up MySQL with PACE.
* Made a better way to onboard new members as it was taking too long.

### Island Migration

* Research island migration and understand the concepts, understand legacy code, and implement some island migration.
* Island migration is a technique used to try and create better evolutionary algorithms. A given population is divided into island and the solutions are evolved using reproduction, mutation, and selection.
* Island migration is divide and conquer. Key points from the research is important parameters are the interval and size, and frequent migration interferes with global diversity because of the domination of migrated individuals.
* There is difficulty in figuring out how to create tables within the database in order to track the evolution and migration of best individuals.
* Made different islands in parallel and migrated periodically.
* Added a SQL Query column: Island Number which will allow multiple individuals available in islands at once.
* Evolution and migration occur asynchronously to not sacrifice efficiency.
* EMADE takes a while for good individuals to be evaluated, so migration would want to be added in later.
* Made database alterations, number of unevaluated individuals are counted in each island.
* Now try to migrate in a ring. Instead of migrating copies, replace the island id in the database.
* Future goals include making it so that islands have different objectives, creating different migration pathways, and develop a history of which islands an individual has been in.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 04/24/2023 | 04/30/2023 | 04/28/2023 |
| Final Updates on Implementing the Keras Model | Completed | 04/24/2023 | 04/30/2023 | 04/26/2023 |
| Create a Graph for the Keras Training and Validation Losses | Completed | 04/24/2023 | 04/30/2023 | 04/26/2023 |
| Find out what the Keras Graphs Show with Training and Validation Loss | Completed | 04/24/2023 | 04/30/2023 | 04/26/2023 |
| Plot Graph from both PyTorch and Keras Training and Validation Losses | Completed | 04/24/2023 | 04/30/2023 | 04/26/2023 |
| Separate Metrics for PyTorch and Keras Models and Graph | Completed | 04/27/2023 | 04/38/2023 | 04/28/2023 |
| Make a Script for the Final Presentation | Completed | 04/24/2023 | 04/30/2023 | 04/28/2023 |
| VIP Final Presentation Notes | Completed | 04/24/2023| 04/30/2023 | 04/28/2023 |

## 17 April 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, April 17th

* NNLearner eager execution issue "resolved".
* We are getting a new error in NNLearner dealing with graph tensors.
* Currently, building out the PSet is resulting in new errors with running EMADE.
* Apical translation realized how to actually load the landmark data from images!

### Scoliosis Sub-Team Meeting Thursday, April 20th

* Most of the hackathon days are presentation prep and filling out slides.
* We should do some of that for the generic stuff that has to be there for presentations. We can take less time on that for the final presentation.
* We have to save room for everything else we have done this semester.
* The last day to make changes to our presentation would be next Thursday, April 27th at 6pm.
* Final peer evals are out, make sure to complete them!
* NNLearner execution has been fixed.
  * Eric is curious what the difference in the evaluated shapes is.
* If we call summary on the Keras models it should get the output of every layer, but we cannot reach that in the execution.
  * It cannot compile the layers together, so it will not be able to get the working summary.
* According to AASCE DecNet, it is supposed to take in a list of tensors of different sizes, but we are not sure of the dimensions.
* Daniel was running into shape issues with ResNet on Monday but it is fixed now. A part of DecNet is not connected yet where it updates all of the heads, the conversions are incorrect. 
* There is another part of DecNet has not been built, but Nathan is saying that it has been built within the neural networks file.
* Minseung and I are getting more metrics, running eval on the AASCE model.
  * We probably will have to modify things in the eval to record the specific metrics.
  * The losses will just be Mean Squared Error (MSE).
  * There is not really an accuracy idea in our model.
  * We need to focus on getting the MSE graph over training.
  * If we need help looking through eval let Eric know. He should be able to point out where we get the loop, and where they calculate the MSE.

</details>

<details>
  <summary><b>Evaluating the PyTorch Model</b></summary>

### Path Given from Minseung

* Minseung gave me a path that I need to input into the terminal within VSCode connected to Azure to run in order to evaluate the PyTorch model. Below is what to run to evaluate the PyTorch model. However, when we run this we get many errors.

  <img width="1034" alt="VIP What to Run to Evaluate the PyTorch Model" src="https://github.gatech.edu/storage/user/60851/files/c0d50e5c-6f54-458e-ae1f-9ea76ec0fdc9">

* Since, we cannot evaluate the Keras model completely, we decided to focus on modeling and finding out differences between the Keras and PyTorch validation and training losses. This would be a great task for members to pick up next semester.

### Great Future Work Task

* A great task to work on in the future is to evaluate the PyTorch model and get metrics using the <code>eval.py</code> to get Mean Standard Error (MSE) values for the model.
* You would also need to get the same metrics with the Keras model in order to compare and graph both to have of a basis if the Keras model is working.

</details>

<details>
  <summary><b>More Updates on Implementing the Keras Model</b></summary>

### Updates on Keras Model for This Week

* This week Daniel made progress on getting past the invalid axis error. That was a problem with how the PyTorch and Keras norm layers work. Keras takes in an <code>axis</code> argument, which should be <code>1</code> when the <code>data_format="channels_first"</code> is specified. Also, <code>channels_first</code> should be specified at every layer.
* Secondly, inside DecNet, the <code>DecNet</code> object has no attribute <code>heads</code>. This is a problem under <code>keras_model/dec_net.py</code>.
  * <code>DecNet</code> has an attribute called <code>heads</code> where <code>nn.Sequential</code> model gets set up with <code>Conv2d</code>, <code>ReLu</code>, and <code>Conv2d</code>.
* To model gradients, Daniel has to convert custom PyTorch loss class in <code>loss.py</code> into Keras. Eric suggested using Adam.
  * Adam enables L2 weight decay and <code>clip_by_global_norm</code> on gradients.
  * [Adam](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
  * This is what the Image Processing team has experimented with, specifically huggingface.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 04/17/2023 | 04/24/2023 | 04/24/2023 |
| Evaluating the PyTorch Model | Completed | 04/17/2023| 04/24/2023 | 04/24/2023 |
| More Updates on Implementing the Keras Model | Completed | 04/17/2023| 04/24/2023 | 04/24/2023 |

## 10 April 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, April 10th

* We had a NNLearner work session this weekend; hashed out some details with primitive/terminal organization and bottlenecked at a TensorFlow eager execution issues.
* The data loader is fixed for testing purposes, and we are awaiting results.
* We still have the SQL server, but Lucas is the only one with access. We thought we had lost it last Monday.
* Currently, we are running into issues with loading <code>.npz</code> data for EMADE.
* The bottleneck from last week for testing Keras version of PyTorch models, we made it easier and am now waiting on results for the Keras model.
* We should have the derived truth data from the landmarks. The main thing was Sunday we hammered out details with NNLearner. 
* All of the layers in the LayerList can compile and we can build out the model. Ran into issues with calling fit. One of the layers requires TensorFlow to have it on. They tried hard coding it to set it to true, but it would not work. Trying to turn a tensor to a numpy array. Numpy was not an attribute, sounds like a version issue.

### Scoliosis Sub-Team Meeting Thursday, April 13th

* I could not attend due to a Women in Engineering Banquet, so below are the notes I got from Eric.
* People are still working on NNLearner and getting it debugged.
* I think the big things were that NNLearner people managed to move past an old eager execution error but we’re getting new problems.
* In terms of my sub-sub-team Daniel and Minseung got the data loader prepped last Monday and were working on feeding images through last time Eric checked.

</details>

<details>
  <summary><b>Keras Model vs. PyTorch Model Research</b></summary>

### Updated Research on Keras vs. PyTorch

* Keras is an effective high-level neural network Application Programming Interface (API) written in Python. It is designed to provide fast experimentation with deep neural networks.
* PyTorch is a relatively new deep learning framework based on PyTorch. It was released in 2017 and it is used for natural language processing applications.
  * PyTorch is simple, easy to use, flexible, has efficient memory usage, and dynamic computational graphs.
  * PyTorch mades coding more manageable and increases processing speed.
* Keras is better suited for developers who want to do a plug-and-play framework that builds, train, and evaluates their models quickly. It offers more deployment options and easier model export.
* PyTorch is faster than Keras and has better debugging capabilities which we had definitely seen when trying to debug the Keras model.
* PyTorch has better community support.
* Keras is best when working with small datasets, rapid prototyping, and multiple back-end support. It is more popular because it its simplicity. However, as we have found, a lot of specifics provided in the PyTorch model are not provided within Keras, or a simpler version is provided that does not suit our needs as well as PyTorch.

[Keras vs. Tensorflow vs. PyTorch](https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article#:~:text=PyTorch%20vs%20Keras,-Both%20of%20these&text=Keras%20is%20better%20suited%20for,and%20has%20better%20debugging%20capabilities.)

</details>

<details>
  <summary><b>More Updates on Implementing the Keras Model</b></summary>

### Updates on Keras Model for This Week

* Daniel worked on the training script for the Keras model. The training script for the PyTorch model is more complicated. Eric says that we just need a more basic script to get the validation and training losses.
  * Daniel found that <code>tensorflow-gpu</code> must be installed when creating the training script.
* At this point, the training script should mostly work. The only problem is an invalid axis error when trying to run data through a portion of the ResNet model. The Keras data loader has inputs of the same shape, so the issue he says should lie in the Keras ResNet model conversions we all completed a few weeks ago.

[Vertebrae Landmark Detection GitHub ResNet.py](https://github.com/yijingru/Vertebra-Landmark-Detection/blob/master/models/resnet.py)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 04/10/2023 | 04/17/2023 | 04/17/2023 |
| More Keras vs. PyTorch Research | Completed | 04/10/2023| 04/17/2023 | 04/17/2023 |
| More Updates on Implementing the Keras Model | Completed | 04/10/2023| 04/17/2023 | 04/17/2023 |

## 3 April 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, April 3rd

* NNLearner is built up to the final decoder; we anticipate adding that this week and being able to run standalone/test for our keras implementation in EMADE.
* Our template <code>.xml</code> is updated for our corrected dataset.
* Primitive set looks straight forward enough to implement. We anticipate some extra work necessary for our custom ResBlocks but the other layers should carry over more or less one for one from the Athi branch, including skip layer functionality.
* We are also working on creating truth data for Apical Translation from the AASCE landmarks. We will research potential EMADE eval functions and finding ways to evaluate on Shriners.

### Scoliosis Sub-Team Meeting Thursday, April 6th

* Nathan did not make as much progress because Patrick mentioned some bugs. Spent most of the time fixing the bugs and getting the dimensions right. The images should all work in EMADE now.
  * He knows which pictures are inverted because he looked through them for the poster presentation and looking at why the SMAPE was so low. He has a list of inverted images. All of the images picked out from the SMAPE process they are not in the 15 inverted list.
  * Those included in the images for 10-15 have been removed for the SMAPE
* Determining that it is not broken without doing a run. Nathan did it and ran it with an input. The output was kind of fishy, but it should be able to run at least. Patrick thinks that he tested it on his own.
* The NLP team has pre-trained embeddings like BERT, our layers are not pre-trained they are just layers.
* Need to confirm that the other primitives are good to include them all in the primitive set.
* Nathan has investigated brightness primitives for contrast enhancement.
* DecNet is supposedly fixed in NNLearner but not fully tested. Getting an error when putting it into the LayerList.
* Looking to get at least one run on standalone before finals--Eric is emphasizing getting an individual properly working (i.e. generated) without too much focus on the generated outputs. Then, we should (theoretically) be able to tune that to more closely match (and hopefully exceed) our expected outputs.
* Adil and Pujith are still working on looking at different evaluation metrics for Azure datasets.
* Lucas's Azure database exists but he can only connect to it through his PC because only his IP address is whitelisted.

</details>

<details>
  <summary><b>What does our Training and Validation Loss Mean?</b></summary>

### Training and Validation

* One of the most widely used metrics combinations is training loss and validation loss over time. The training loss indicates how well the model is fitting the training data, and the validation loss indicates how well the model fits new data.
* A typical plot looks like the below image, similar to an exp(-x) plot starting at the origin.

  <img width="286" alt="Screenshot 2023-04-26 at 12 17 34 PM" src="https://github.gatech.edu/storage/user/60851/files/77ba0dfe-3f56-419e-8001-4f43117b4203">

* An epoch is the number of complete passes through the training dataset, in our case, we use 50 of them.
* A common practice is putting two different models in the same chart.
* High bias occurs when the learning algorithm is not taking into account all the relevant information, becoming unable to capture the model's richness and complexity.
* Underfitting is when the algorithm is not able to model either the training data or new data, consistently obtaining high error values that do not decrease over time.
* A biased model does not take into account relevant information, which leads to underfitting.
* We could just use the training curve and check that the loss is high and it does not decrease to see that it is underfitting.
* High variance happens when the model is too complex and does not represent the simpler real patterns existing in the data.
* Overfitting is when the algorithm trains the data well, but it performs poorly on new data.
* The higher the variance of the model, the more it overfits the training data.
* The training process should be stopped when the validation error trend changes from descending to ascending.
  * If the process is stopped before that point, the model will underfit.
  * If the process is stopped after that point, the model will overfit.
* A representative dataset reflects proportionally statistical characteristics in another dataset from the same domain.
* An unrepresentative training dataset happens when the data available during training is not enough to capture the model, relative to the validation dataset.
* Some scenarios:
  * If the training curve looks okay, but the validation is noisy around the training curve, then it could be the case that the validation data is scarce and not representative of the training data.
  * If the validation loss is much better than the training loss, the validation dataset is easier to predict than the training dataset. An explanation could be that the validation data is scarce but widely represented by the training dataset, so it performs extremely well on the few examples.

[Learning Curves in Machine Learning](https://www.baeldung.com/cs/learning-curve-ml#:~:text=One%20of%20the%20most%20widely,the%20model%20fits%20new%20data.)

### PyTorch Validation and Train Losses

  ![PyTorch Losses](https://github.gatech.edu/storage/user/60851/files/6119631f-0f98-465c-9363-4c75542d6cd5)

* Looking specifically at the PyTorch Losses Graph, the training process should be stopped when the validation error trend changes from descending to ascending which happens around epoch 10.
* I do believe the PyTorch model has an underepresenative training dataset. This is when the data available during training is not enough to capture the model relative to the validation dataset. The train and validation curves are improving, but there is a big gap between them. They operate like datasets from different distributions.
  * Between 0 epochs and 10 the curves are improving, but then a gap starts to grow between them.
* Within this model, it could also being overfitted. The training loss goes down over time, achieving low error values, and the validation loss goes down until a turning point is found around 10 epochs, and then it starts going up again. That point represents the beginning of overfitting.

</details>

<details>
  <summary><b>Implementing the Keras Model - Initial Research</b></summary>

### Keras Model Implementation - Data Loading

* When building out the Keras model, Daniel has run into problems. We are trying to implement the Vertebra Landmark Detection that did well in the AASCE competition in Azure to eventually run EMADE on the model. Our ultimate goal is to improve the model that already works very well with a low SMAPE value. Their model is implemented in PyTorch. Daniel, Minseung, and I have worked most of the semester double checking our conversions and converting the PyTorch model into Keras because EMADE is more compatible with Keras.
* Daniel's main task was converting the PyTorch model completely to Keras with the conversions we have already tested. While Minseung and I worked on getting the validation and training losses from the PyTorch model. Eventually we will work on evaluating the PyTorch model, as the Keras model will be evaluated within our EMADE runs along with NNLearner that other teammates have been working on.
* Daniel ended up converting the DataLoader to Keras.
  * For batched data loading, PyTorch has two classes <code>torch.utils.data.Dataset</code> and <code>torch.utils.data.Dataloader</code>. <code>Dataset</code> is used to create a custom function. This is what the VFL model uses. <code>DataLoader</code> takes in <code>Dataset</code> classes and creates an iterator after performing actions on the <code>Dataset</code>.
  * In Keras, <code>tf.keras.utils.Sequence</code> is the equivalent of <code>torch.utils.data.Dataset</code>, but there is no <code>DataLoader</code> to help with batching, shuffling, and multiprocessing. Therefore, Daniel had to implement the batching himself.

[Vertebrae Landmark Detection GitHub ResNet.py](https://github.com/yijingru/Vertebra-Landmark-Detection/blob/master/models/resnet.py)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 04/03/2023 | 04/10/2023 | 04/10/2023 |
| Find out what the PyTorch Graphs Show with Training and Validation Loss | Completed | 04/03/2023 | 04/10/2023 | 04/10/2023
| Implementing the Keras Model - Initial Research | Completed | 04/03/2023| 04/10/2023 | 04/10/2023 |

## 27 March 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, March 27th

* We have four new members that joined our team, so now our team is pretty huge.
* A new task is Apical Translation which is another way to measure scoliosis similar to the Cobb Angle.
* Another task is to update the template for EMADE runs and the updated dataset.
* We also have to work on getting an EMADE run which means writing primitives, terminals, and implementing skip connections.
* My team is the PyTorch and Keras conversions, we want to fully convert the model to run accuracy and loss comparisons.
* Lastly, we discussed image processing and the contrast issue.

### Scoliosis Sub-Team Meeting Thursday, March 30th

* Nathan is looking into contrast options.
  * Keras might get mad because there cannot be a multi-output layer. Not sure if this is a proven error in NNLearner, but we can fix it. It is not an EMADE problem, it is a Keras problem.
* Patrick has had a few strange bugs in some of the layers of NNLearner, and will take a long time to replicate the model in Keras. The PyTorch and Keras people aka my team, is working on the specifics to make sure the conversions work.
  * He has to put Nathan’s decoder in it. He has the path, so it should be okay.
* NNLearner is still not working, but we are still getting there.
* Brian thinks based on the branch he got DMed on Monday. The terminals are not evolving with the rest of the model. Either there are constant terminals or trained terminals.
  * If we want to evolve the stride of the Conv2D() we would need to create another individual. Dr. Zutty might be referring to the pre-trained terminals NLP used.
  * We should limit our terminals to be pre-trained, and not do something that has not been done before. We do not need to evolve the terminals, but be able to swap into the primitive. The structure of the tree will be evolved using EMADE.
  * Add layers and terminals in such a way that a layer can be built easily and fast and be able to test it based on what is fed into the terminals.
* Lucas will get the xml Monday.
* Atypical Translations is another task where doctors use this method to measure scoliosis.

</details>

<details>
  <summary><b>PyTorch VFL Model Training</b></summary>

* The code below is from the VFL Model GitHub Repository. This is what needs to be used in order to train the model in Azure.
  ```
  python main.py --data_dir dataPath --epochs 50 --batch_size 2 --dataset spinal --phase train
  ```
* First, I need to make sure EMADE is installed properly into Azure.
  * I have been having issues which I will get resolved tomorrow during the VIP meeting.
    * Turns out I did not need EMADE, but I actually just need to have Anaconda properly installed.
    * I used the data that was already trained within the folder in Azure.
  * [EMADE Instructions](https://github.gatech.edu/emade/emade)
* Minseung told me to change the 50 after <code>--epochs</code> to <code>num_epoch</code>.
* For now, I will write code locally to get two <code>.txt</code> files into a graph. I do not want to waste money running my compute node in Azure while I figure it out when my compute node is running on Azure.
* Right now, in Azure within the <code>weights_50e_original_spinal</code> folder it has two <code>.txt</code> files which houses the train and validation losses as  <code>train_loss.txt</code> and <code>val_loss.txt</code>. Since Minseung trained the PyTorch Model, I used the <code>.txt</code> files within his notebook.
* In the section below I will discuss what I did to get these two files into a graph to model the train and validation losses.

</details>

<details>
  <summary><b>Creating a Graph of PyTorch Losses</b></summary>

### What Did I Do?

* I first inputed the <code>.txt</code> files by using the <code>.read()<code> method, but I found that using pandas is much easier.
* I created dataframes to read in the files and then translated them to numpy arrays to later plot in a subplot. Pandas can get problematic when making plots side by side and using the same dataframe twice. I ended up creating a few dataframes.
  * The <code>counting_df</code> is a dataframe of numbers 1-50 to use as x-values within the plots.
  * The <code>train_loss_df</code> is a dataframe of the training loss numbers.
  * The <code>val_loss_df</code> is a dataframe of the validation loss numbers.
* I transitioned to using numpy which is more basic than pandas, but the plots look better and there is more capability with what we want to do.
* I looked up how exactly loss is supposed to be graphed, and both the training and the validation loss should be on one graph. Therefore, I have the training loss as a solid red line, and the validation loss as a dotted red line. PyTorch is red because fire, so I figure that is easier to remember for when the PyTorch and Keras losses are all plotted within the same plot.
* I had to use the subplot method in order to get an appropriate legend for the graphs, and easier organization.
* In the section below is the code which has been copied and pasted into Azure as well. In the section after the code is an image of the plots.

### Code for Creating the PyTorch Losses Graph

  ```
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
  ```

### Graph of PyTorch Training and Validation Losses

  ![PyTorch Losses](https://github.gatech.edu/storage/user/60851/files/5c1fac5a-ed5c-4caf-886a-716d611759c6)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 03/27/2023 | 04/03/2023 | 04/02/2023 |
| Train the PyTorch VFL Model | Completed | 03/27/2023 | 04/03/2023 | 04/02/2023 |
| Create a Graph for the PyTorch Training and Validation Losses | Completed | 03/27/2023 | 04/03/2023 | 04/02/2023 |
| Add New Members to Notebook | Completed | 03/27/2023 | 04/03/2023 | 04/03/2023 |

## 13 March 2023

<details>
  <summary><b>Team Meeting Notes - VIP Midterm Presentations</b></summary>

### Bootcamp Team 2

* Data processing for columns:
  * Names were replaced by their prefixes.
  * For the age column they used multivariate iterative feature imputation.
* Used 6 Machine Learning Models.
* For the Neural Network MLP Classifier they used the 'relu' activation layer.
* Used a Logistic Curve bounded from 0 to 1 for binary classification.
* For the SVM Classifier it creates an optimal hyperplane that separates data into different classes.
* Tree Decision Classifier was basic decision making that can grow complex easily.
* ADA Boost is an ensemble algorithm that looks at how each feature affects the outcome and changes the weights accordingly.
* For the primitive set they used strongly-typed and then they also added terminals.
* Spent a lot of time getting EMADE set-up.
* Then ended up using 1 computer and ran it overnight for 83 generations.
* There were 55 instances of AdaBoostLearner.
* DEAP had better final results but took a longer time to run.

### Image Processing

* Transfer learning is an optimization that allows rapid progress or improved performance when modeling.
* Multi-Class: only one label for an image, this was done for past semesters.
* Multi-Label: consider the cases that an image may have multiple diseases. This is what the ChexNet paper uses.
* Changed to Binary Cross Entropy. This measures the difference between the predicted probability of the positive class and the actual value.
* Many benefits of NNLearner like not interrupting with the main NNLearner method, making it safe to develop without issues.
* Fast Failing - Code Changes, checking every specified number to see if there is failing, and it will manually fail it.
  * Anh did not wanted to hard code it. Tried to import from Neural Network methods. Future work would be data-mining.

### Bootcamp Team 5

* Predicted age based on correlated variables such as "Sex".
* Members were able to get EMADE working.
* Difficulties with installing dependencies and SQL.
* Completed 30 generations on one machine over 2 days.
* Starts at 500 individuals and plateaus around 200.
* AdaBoost was dominated around generation 10.
  * Makes node where two leaves called stumps can be grouped with each one progressively learning.
* Second run of EMADE ran for about 200 generations locally, only 15 individuals were created.
* MOGP performed better for them.
* EMADE takes a while to run, but it is great to know which ML algorithms work the best.

### Bootcamp Team 1

* Dropped name, ticket number, cabin, and embarked columns.
* Creating the Pareto front for the machine models, sometimes they had to make one algorithm worse.
* None of the models had close to zero FPR or FNR.
* Primitives used in the GP tree do not require training which allows for quick evaluation.
* Succeeded in accessing the database remotely.
* Did 35 generations for EMADE.

### Stocks

* Sub-Teams
  * Literature Review and Overview
  * Price Trend Prediction
  * Reinforcement Learning
  * Future Work
* Sentiment Analysis
  * Ran into issues with data sources. FinViz and RSS feeds only include recent financial news for a given company. There is no way to access historical data.
  * Used snscrape which finally worked!
  * The data is super noisy.
* Fundamental Data
  * Original plan for stocks team last year.
  * Comes from quarterly-reported earnings/financial data.
  * This semester researched alternative sources for fundamental data.
  * Found a tool called SimFin that provides historical fundamentals through an API.
* Some of the EMADE runs did do better this semester, but some did not have a lot of individuals.
* EMADE Run with TA Lib Primitives
  * Worked on runs with previous stock team implementation of TA-Lib primitives.
* NNLearner
  * Fixing standalone/upgrades.
* Integration of portfoliodata in NNLearner.
  * Issues with Integration - Preprocessing.
    * Last semester, for the RLLearner used Keras flatten layer to flatten the last two dimensions of the input before running LTSM.
  * Issues with Integration - Miscellaneous.
    * Strings for CIFAR-10 primitives out of date.
  * Standalone Results
    * After fixing issues with portfolio data integration, they successfully tested multiple distinct individuals.
    * Early stopping stops the individuals at 2 generations on test dataset, still needs to be debugged.
* Starting RL EMADE runs.

### Bootcamp Team 3

* Identified the relevant features: passenger class, sex, age, siblings, parents/children, and survived or not.
* Set the passenger index column to be the row.
* Used K-Folds for the data with the dropped columns.
* For ML they used: AdaBoost Classifier, K Neighbors Classifier, Stochastic Gradient Descent Qualifier, Random Forest Qualifier, Nu-Support Vector Machines (RBF Kernel), and Neural Network Model.
* Evolutionary Algorithm
  * Considerations were strongly or loosely typed.
    * Used strongly typed.
  * For the evaluation they counted the number of false positives and false negatives.
* Ran 51 generations using empirically chosen values.
* Control EMADE Run was 67 Generation Pareto Front.
* ML algorithm was outperformed by EMADE. The evolutionary algorithm did slightly better than EMADE.

### Bootcamp Team 4

* Used the default <code>titanic_data_splitter</code> for data preparation.
* GP can result in a better AUC, has a better control of objectives, diverse Pareto front, but dine tuning GP is slower.
* In the long run ML is slower.
* EMADE with the processed dataset:
  * Steeper drops
  * Less frequent drops
  * Faster optimization
  * Faster training
* The processed dataset is a lot faster decrease over generations.
* EMADE compared to ML and GP
  * EMADE was run for 16 hours for 25 generations using a single machine.
  * EMADE takes a long time to run.
  * With more generations EMADE would have performed better.
* The Pareto front is dominated by AdaBoostLearner.
* The few successful individuals in the initial population have a large influence on later generations.

</details>

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### Scoliosis Sub-Team Meeting Thursday, March 16th

* Goal is to be organized with how far we are for everything.
* We need to update the onboarding document for new members.
* The presentation went well overall. Aaron and Dr. Zutty did not have many questions because we went at the back end.
* The biggest task is NNLearner, building it out, testing it, and running it through standalone.
  * It is a lot for a new member to get into.
  * New members can update the template file, and write terminals.
  * If we get 5 new members, 3 members getting comfortable with the structure would be nice.
* Need the primitives for the Decoder.
* In the Decoder primitive have a layer for how many layers back to take in.
  * Wrapping in a primitive to get standalone to work should be good for now.
  * In the future, we do not want to hard code it, we want it to be able to be inputted.
* The colors and the contrast could be tackled by new members.
* Daniel, Minseung, and my project update:
  * The testing and training script for PyTorch has a lot of dependencies on PyTorch. The training script, the way they do it is slightly different from what we would do with Keras. We need to write a new training script to test that.
  * The purpose of this is to test conversions. When we have standalone up, we will essentially be doing the same thing. However, this is useful because we will know our conversions are correct.
  * A goal date to shoot for is 1 to 1.5 weeks after break.

</details>

<details>
  <summary><b>Action Plan</b></summary>

### Action Plan

* Daniel, Minseung, and I have seemed to have formed a sub-sub-team. Our task is to make graphs comparing the loss and validation accuracies for the PyTorch and Keras models. The goal was to finish these graphs by Monday for the Midterm Presentations. 
  * However, the task turned out to be a lengthy and hard task. There are multiple steps to this task that I will now explain.
* The first step is to get the loss and validation accuracies for the PyTorch model. This model is built in Azure, so I will finish setting up EMADE on my compute node and then train the model to get the loss and validation accuracies.
* The next step is getting those loss and validation accuracies into a graph using <code>matplotlib</code>.
* While I work on this, Minseung and Daniel will start building the Keras model equivalent within Azure.
  * The training script for the VFL's PyTorch version depends on too many PyTorch dependencies.
  * The input shape for the Keras model will come from the base dataset in the <code>train.py</code>. Keras only needs the input shape to be passed in once at the beginning, and does not have a <code>input_channels</code> parameter like PyTorch.
* I will then join them, and then we will need to train that model to get its losses and accuracies.
* We will then plot them altogether, and make sure the errors in between the models are minimal and as close to zero as possibly.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - VIP Midterm Presentations | Completed | 03/13/2023 | 03/27/2023 | 03/13/2023 |
| Team Meeting Notes - Scoliosis | Completed | 03/13/2023 | 03/27/2023 | 03/16/2023 |
| Create an Action Plan for New Task | Completed | 03/13/2023 | 03/27/2023 | 03/27/2023 |

## 6 March 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, March 6th

* We are building out the NNLearner layer. This is the major bottle neck for progress right now. The finalized NNLearner will hold a layer list of layer primitives. We are anticipating a layer list size limit of approximately 100.
* Decoder side is essentially done. We are pending the completion of NNLearner. Also, we are finalizing tests for obtaining the dictionary of outputs to be given to the "final decoder."
* PyTorch to Keras conversions and testing are finished for DecNet layers.
* We discussed the difficulties surrounding the proposed individual structure.

### Scoliosis Team Meeting Thursday, March 9th

* The Bootcamp class is about 24 students. We are going to have 8 more people. We hope that we can convince the new members to start their own sub-team.
* Goals for this meeting:
  * See where we are at for getting an individual that we can feed an EMADE datapair to actual run EMADE.
  * Working on presentation.
* Actually building the model works now!
* We will only have one NNLearner per individual. We need terminals for all of the layer primitives. The terminals are the layer parameters.
* ResNet compiles all until it is trying to compare the output to a landmark. It is a dimensionality error. It is expecting a Cobb angle, but it is receiving a tensor.
* ResNet is outputting a tensor which is good. We cannot train it without DecNet and the Decoder.
* The reason ResNet is outputting 1 is because we have not changed the name of the unit test file for the <code>.npz</code>.
* A good task for new members would be to add terminals for the layer primitives. This is all within <code>gp_framework_helper.py</code>.
  * [Adding Terminals](https://github.gatech.edu/emade/emade/blob/athite3-nn-final/src/GPFramework/gp_framework_helper.py)
* Since the layers are primitives, they feed the layers that are terminals for the layer parameters. This is so they can get different activations and weights.
* Testing the terminal implementation would need to be an EMADE run, could not do unit tests, but we will have to see.
* Testing input and output would be an EMADE run.
* Our model does worse on the cropped images because there are screwed up Shriner Images. Those images we took out. Cropping shifts the landmarks, so Nathan had to alter the landmarks to update after cropping.
* The SMAPE is 11 which is better than the base model for Shriners Children's Hospital.

[Scoliosis Midterm Presentation 2023](https://docs.google.com/presentation/d/1AimgsfLxqCzFjXL5zdzpKi6f41qIMHYOWELXr0Jena4/edit?usp=sharing)

</details>

<details>
  <summary><b>Updated Tasking</b></summary>

### My Updated Task

* Daniel, Minseung, and I have seemed to have formed a sub-sub-team. Our task is to make graphs comparing the loss and validation accuracies for the PyTorch and Keras models. The goal is to finish these graphs by Monday for the Midterm Presentations.
  * This is a lengthy and hard task as the training script for the VFL's PyTorch version depends on too many PyTorch dependencies.
* This past month we have been working on the conversions from PyTorch to Keras in the <code>SpineNetEx.txt</code> file. We have completed ResNet and DecNet with the individual layer conversions and tested them. These conversions need to be put together in order to build the model.
* The Decoder Sub-Team people have completed the DecNet portion of the model.
* Daniel is going to put together our conversions for ResNet into a full Keras model.
  * The input shape for the Keras model will come from the base dataset in the <code>train.py</code>. Keras only needs the input shape to be passed in once at the beginning, and does not have a <code>input_channels</code> parameter like PyTorch.
* Minseung and I are going to understand the <code>train.py</code> file and figure out a way to get the loss and validation accuracies specifically for the PyTorch model and output them to a <code>.csv</code> file.
* We will then altogether make a graph for comparisons between the PyTorch and Keras models to present at the Midterm Presentation.
* In the VFL Model GitHub we can train the PyTorch model there, and modify the model in order to get losses and accuracies. The Keras model will need to be implemented into Azure and we will train the model there to get losses and accuracies.

</details>

<details>
  <summary><b><code>train.py</code> in Azure and GitHub</b></summary>

### <code>train.py</code> in Azure

* In Azure the <code>train.py</code> model it is all in Keras.
* We need to add the conversions made in the previous weeks into the the training file in Azure to be able to train it.
* However, within the training script for the PyTorch version depends on too many PyTorch dependencies. We converted all we could for the model, but when piecing them together we have to find the relative Keras dependencies.
* Daniel has worked hard to find these dependencies, but he was not able to find them all before the Midterm Presentation. Therefore, we will resume this task after presenting.
* We will most definitely have all of the graphs and statistics by the Final Presentation.
* Once I write the code to get the trained PyTorch model statistics into a <code>.csv</code> file I should be able to easily translate that within Azure for the Keras model.

### <code>train.py</code> in AASCE VFL Model Branch in GitHub

* Our goal is to train the PyTorch model to get losses and accuracies.
* Looking through the repo in order to run the model I will have to download all of their files.
* I did not have enough time to do this before Midterm Presentations, but next week I plan to create a new environment on my laptop and download all of their files and dependencies needed in that environment so I can train their model.
* After I train the model, I will modify their code in order to get the losses and accuracies and export them to a <code>.csv</code> file.

[**AASCE VFL Model Branch**](https://github.com/yijingru/Vertebra-Landmark-Detection/tree/master/models)

</details>

<details>
  <summary><b>Midterm Presentation Slide 21</b></summary>

### My Script

* I worked with Minseung and Daniel and converting the VFL Model from PyTorch to Keras.
* Keras is a neural network platform that runs on top of the open-source library TensorFlow, and PyTorch is a framework for building deep models.
* The VFL model has four PyTorch layers:
  * <code>nn.Conv2d</code>
  * <code>nn.BatchNorm2d</code>
  * <code>nn.ReLU</code>
  * <code>nn.MaxPool2d</code>
* Most of the parameters of the Keras equivalents are similar with a few key differences.
  * Passing as a parameter is not as flexible in Keras, so the <code>ZeroPadding2D()</code> layer must be used to have integer or tuple padding.
  * Keras is by default <code>channels last</code> while PyTorch is by default channels first, so that has to be explicitly specified.
* We also created a testing script to verify that comparison match.
  * The script takes in a random numpy array of any dimension, and feeds it to a Keras and a PyTorch model.
  * The output at the end is compared and checked to see that there is a small difference between them.
  * Both models must be in the same mode, PyTorch by default is in training mode, and Keras is by default in eval mode.

[Scoliosis Midterm Presentation 2023](https://docs.google.com/presentation/d/1AimgsfLxqCzFjXL5zdzpKi6f41qIMHYOWELXr0Jena4/edit?usp=sharing)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 03/06/2023 | 03/13/2023 | 03/09/2023 |
| Get an Updated Task | Completed | 03/06/2023 | 03/13/2023 | 03/08/2023 |
| Look into train.py in Azure and VFL Model GitHub | Completed | 03/06/2023 | 03/13/2023 | 03/12/2023 |
| Fill out Slide 21 for Midterm Presentation | Completed | 03/06/2023 | 03/13/2023 | 03/12/2023 |
| Make a Script for Slide 21 for the Midterm Presentation | Completed | 03/06/2023 | 03/13/2023 | 03/12/2023 |

## 27 February 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, February 27th

* Decoder side is done with converting the first decoder later (c_4), and only two more and the dict output is left.
* NNLearner has been used to "build" the VFL model, but the parameters have not been inputted. Other issues will probably necessitate a reworking and writing of the layer code.
* Testing has moved past <code>Conv2D()</code>; we are working on converting other components of <code>DecNet</code>. Running layers in the eval and training modes create various output disparities.
* Not entirely sure how EMADE handles model weight tuning with mating and mutation. Does it fit every individual it generates before evolving?
  * You would be refitting every time. There is no preservation of weights. Unless there is a deep copy maintained then it would be constructed for evaluation.
  * Every individual is evaluated uniquely and fits to a fresh weight.
  * Might have experimented with sharing weights, but it is not implemented.
* Nathan is suggesting taking the combination module class and using it as a primitive framework.
* There has been some overloading in the decoder. They are actual layers of the model. The layers are made of PyTorch layers which have weights that can be trained. They are trained on, but the final decoder that takes in the dictionary is not trained.
* Once we have all of the decoder layers working, we should be able to hook it up to ResNet and fill in the weights.
* My testing of PyTorch to Keras is beginning to overlap more and more with the other sub-team groups which is great because we are sharing work.
* The resizing issue worked with NNLearner. We just need to make a smarter way to resize images. We are not sure if the resizing will remove landmarks. There are functions that can be used for resizing. The truth data can be appropriately scaled by a factor as well.
* All sub-layers are a part of a parent layer, we could do more organization.
* Scoliosis poster is being turned in this week as the poster session is this Thursday, March 2nd.

### Poster Session Thursday, March 2nd

* We did not end up winning the poster session, but we presented to several people and judges.
* They thanked us for our work. The main thing we marketed was the impact our project would have on doctors work.
* Obviously, the doctors would still need to determine their Cobb angle prediction, but it would be a good measurement and tool for them to have. Eventually if it is a good enough model, doctors might not have to measure the angle.

</details>

<details>
  <summary><b>DecNet <code>SpineNetEx.txt</code> PyTorch to Keras Converting</b></summary>

### Overall Notes on Conversions
* Minseung and I worked this week on splitting up DecNet. Minseung took <code>dec_c2</code> and <code>dec_c3</code>, and I took <code>dec_c4</code>, <code>hm</code>, <code>reg</code>, and <code>wh</code>.
* Within the PyTorch to Keras Google Colab I placed the base conversions.
* Please refer back to my notebook entry from 13 January 2023 Base Network PyTorch to Keras Converting, as there is a more detailed description on the differences of each function as I converted them from PyTorch to Keras.
* Additional information I found on the <code>ReLU(inplace)</code> is shown below.

### <code>ReLU()</code> Function in PyTorch and Keras
* Originally when converting <code>ReLU(inplace)</code> from PyTorch to Keras, I thought it would be best to use Keras' <code>Activation('relu')</code>, but there is another function in Keras <code>ReLU()</code>.
* The <code>ReLU()</code> function in Keras with default values, it returns element-wise <code>max(x, 0)</code>.
* The <code>Activation('relu')</code> function applies the rectified linear unit activation function. With default values, this returns the standard ReLU activation: <code>max(x, 0)</code>, the element-wise maximum of 0 and the input tensor.
* These two are essentially the same. For readability purposes, it would be better to use the <code>ReLU()</code> function with no inputs.

[**PyTorch to Keras Testing Code**](https://colab.research.google.com/drive/1ULGVWYgeRKYzv48Wp79C_PNQLyGsj_N_?usp=sharing)

</details>

<details>
  <summary><b>ResNet <code>SpineNetEx.txt</code> PyTorch to Keras Testing</b></summary>

### Overall Notes on Testing
* Minseung and I worked this week on splitting up DecNet. Minseung took <code>dec_c2</code> and <code>dec_c3</code>, and I took <code>dec_c4</code>, <code>hm</code>, <code>reg</code>, and <code>wh</code>.
* Within the PyTorch to Keras Google Colab I placed the base conversions.
* We added in code that makes sure the weights get transferred properly for the different layers.
* Also, the code now has multiple different tests. For example, a <code># CONV2D Test</code>, <code># BATCHNORM Test</code>, <code># RELU Test</code>, and <code># MAXPOOL Test</code>. 
* Since all of the layers in the model fall into one of the categories listed above, we can just change the inputs within the test to verify the error is zero.

```
# Very hacky code for making sure the weights get transferred properly for the different layers
  # Will not work if there are multiple layers in torch_layers / keras_layers (excluding ZeroPadding2D)
  if len(keras_weights) > 0:
    contains_conv2d = any(isinstance(layer, keras.layers.Conv2D) for layer in keras_layers)
    contains_batchnorm = any(isinstance(layer, keras.layers.BatchNormalization) for layer in keras_layers)
    if contains_conv2d:
      # Transpose weights if conv2d
      pytorch_model[0].weight.data = torch.from_numpy(np.transpose(keras_weights[0], [3, 2, 0, 1]))
    
    if contains_batchnorm:
      pytorch_model[0].weight.data = torch.from_numpy(keras_weights[0])
      pytorch_model[0].bias.data = torch.from_numpy(keras_weights[1])
      pytorch_model[0].running_mean.data = torch.from_numpy(keras_weights[2])
      pytorch_model[0].running_var.data = torch.from_numpy(keras_weights[3])
```

* I tested the PyTorch <code>ReLU(inplace)</code> function with the Keras <code>ReLU()</code> function and was able to get an error of zero. Therefore, we can either use the <code>Activation('relu')</code> function or the <code>ReLU()</code> function.
* Overall, all of the conversions seem to be working. The same issues persist with padding, so we will have to be careful of the input shape and the padding parameters.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 02/27/2023 | 03/06/2023 | 03/06/2023 |
| DecNet PyTorch to Keras Converting | Completed | 02/27/2023 | 03/06/2023 | 03/06/2023 |
| DecNet PyTorch to Keras Testing | Completed | 02/27/2023 | 03/06/2023 | 03/06/2023 |

## 20 February 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, February 20th

* The final Decoder is implemented and tested. We are waiting for the remaining pipeline to be built.
* The ResBlock was broken for our purposes, so Patrick just rewrote it.
* The Decoder sub-sub-team is now working in implementing the DecNet from the back half of the model architecture.
* Currently, we are trying to get the weights for specific layers when loading pre-trained weights into the entire SpineNet model.
* The final Decoder implementation will take multiple NNLearner outputs as an input.

### Scoliosis Team Meeting Thursday, February 23rd

* Poster presentation is going to be like a competition.
  * Doctors are going to be going around the room and asking questions, making the prizes and placements.
* We want to do a high level presentation, speaking more technically to the judges.
* Shriner's people would be interested with how the technology could scale.
  * For example, practical applications of what we are doing.
* Dress somewhat formally, pants and a button down or blouse.
* We are still having trouble creating compute nodes for new members. This is a security issue on Shriner's side. Cole and Eric are currently working on resolving the issue.
* We want all inputs to the model to have the same size to prevent the <code>.npz</code> error.
* Truth data are the landmarks but our output is currently just a 500 x 500 array.
  * Only a single ResBlock has been implemented.
* Potential Issue: even if we can build individuals with NNLearner, we do not have a way to back propagate and update weights in the entire model. For example, how we are going to evolve the ResNet part of the model).
* We cannot update weights right now because we do not have the entire model built out.

</details>

<details>
  <summary><b>ResNet <code>SpineNetEx.txt</code> PyTorch to Keras Converting</b></summary>

### Overall Notes on Conversions

* In my Personal Scolosis Repository I added a file titled <code>base_conversions.py</code>. This file is the key towards the correct conversions for each layer. This is not the full built model, as the full model is much more complicated.
* When taking a closer look at the <code>SpineNetEx.txt</code> file, at a basic level there are only 12 different types of layers.
* When building these models there are different combinations of all 12.
* Within layer 2, 3, and 4 from the ResNet portion, there is a downsample functionality which is another layer composed of a <code>Conv2D()</code> layer and a <code>BatchNorm2d()</code> layer. I copied these to the <code>base_conversions.py</code> and placed them underneath their corresponding title like <code>### LAYER 2 DOWNSAMPLE</code>.
* Please refer back to my notebook entry from 13 January 2023 Base Network PyTorch to Keras Converting, as there is a more detailed description on the differences of each function as I converted them from PyTorch to Keras.
* In addition, the contents of <code>base_conversions.py</code> has been copied to the Google Colab linked in the ResNet <code>SpineNetEx.txt</code> PyTorch to Keras Testing and Resources.

### A Relook at Tasking

* From what I understand with the VIP meetings and our own Scoliosis meetings, we are mainly looking at making sure these conversions work with tests.
* We want to create a unit test that can be applied if you pass in the PyTorch and the Keras and an input shape if needed. 
* Other people on the Scoliosis team have been tasked with these conversions of DecNet and ResNet and are running into issues. This is when we can jump in and test our unit test functionality.
* I am going to get their conversions they made to the DecNet model specifically, and test them or give them the unit test to use to ensure they are converting from PyTorch to Keras properly.
* Also, now that we are starting to build the model, we can show to the team what we have found with the ResNet portion so people are not doing double work. Because as we were testing our functionality, we used several layers at the beginning as an example.

[**Notes with Minseung on Each Function in SpineNetEx.txt**](https://docs.google.com/document/d/1AGOvWl5OaJKkOgrKNid8n2eyBDf10t2YjGRFQll6JxI/edit?usp=sharing)

</details>

<details>
  <summary><b>ResNet <code>SpineNetEx.txt</code> PyTorch to Keras Testing</b></summary>

### Overall Notes

* Daniel, Minseung, and I worked on the testing functionality and making sure all four main layers can be tested.
* The <code>BatchNormalization()</code> weights are two dimensional while the weights in <code>Conv2D()</code> are four dimensional.
* <code>ReLU()</code> does not have any weights.
  * If the numbers are less than zero, it makes them zero.
  * If the numbers are greater than zero, it does not touch them.
* The weights for <code>Conv2D()</code> need to be transposed.
* Whenever a layer gets trained, it does a lot of math. The weights are the coefficients to the math being done.
  * If the weights are not copied over than the results are going to be completely different.
* The model will infer the input shape based on the call.

### Challenges with Unit Test

* When calling the <code>test_torch_keras_conversion(torch_layers, keras_layers, input_shape)</code> function with passed in layers you are testing, we had to deal with the weight issue discussed in the Overall Notes section.
* The function will look at the passed in layers and their corresponding weights. From there the function will complete a specific test based on the amount of weights associated with the layers.
* This functionality will not work if there are multiple layers in the <code>torch_layers</code> or <code>keras_layers</code> passed in. This excludes when <code>ZeroPadding2D</code> has to be passed in with <code>Conv2D</code> and <code>MaxPool2D</code>.

### Conclusions for Everyone

* When testing the <code>Conv2D()</code> and <code>MaxPooling2D()</code> layer, we found that in Keras the <code>padding</code> parameter only takes in a value of "same" or "valid" while PyTorch takes in a numerical value. We want to emulate the numerical value within Keras, so we can use the <code>ZeroPadding2D()</code> layer with the <code>padding</code> specified as a numerical value.
* The unit test can be used to test the conversions between the PyTorch and Keras layers. You will have to go one at a time.
* If the returned error is zero, then it will return the error as well as True, otherwise it will the return the error as well as False.

[**PyTorch to Keras Testing Code**](https://colab.research.google.com/drive/1ULGVWYgeRKYzv48Wp79C_PNQLyGsj_N_?usp=sharing)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 02/20/2023 | 02/27/2023 | 02/23/2023 |
| ResNet PyTorch to Keras Converting | Completed | 02/20/2023 | 02/27/2023 | 02/23/2023 |
| ResNet PyTorch to Keras Testing | Completed | 02/20/2023 | 02/27/2023 | 02/23/2023 |
| Peer Evaluations | Completed | 02/20/2023 | 02/24/2023 | 02/20/2023 |

## 13 February 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, February 13th

* The AASCE <code>.npz</code> was empty. Shriners set was fine. After the fix, NNLearner is still running into issues.
* Final decoder is moved in and wrapper is complete. The next task involves testing input and outputs to verify it works.
* Keras and PyTorch testing moving forward on DecNet portion of model pipeline.

### Scoliosis Team Meeting, February 16th

* The poster is going to be a mini-research paper that is very condensed.
  * Compare it to a very professional science fair.
  * It should be 4 by 3 feet.
  * If we get any images that would be great just do not put any patient information.
* EMADE is not working, still getting issues.
  * This is a roadblock for NNLearner right now, the <code>.npz</code> having issues.
  * This would be in NNLearner methods or a <code>.npz</code> issue.
  * The model fit is not working, everything works until the fitting of the model.
* ResBlock should be made out of individual layers.
* NNLearner works, we cannot fit anything, but we can build some blocks.
* Nathan got the final decoder to work.
  * He had to change the wrapper and primtiive.
  * It was breaking because he set up the unit tests wrong, but if it is set up correctly, the wrappers error. That is not a big issue because it is just fixing the order of parameters.
* Instead of having three EMADE datapairs we have 1, this will eliminate the roundabout way of going about things.
* Monday is the deadline for the abstract of the poster.

[Scoliosis Sub-Team Folder](https://drive.google.com/drive/folders/1-T1Fe77Qui8N--jE_5ABh1CTLRmhR2W4?usp=share_link)

[Scoliosis Poster](https://docs.google.com/presentation/d/1VO0NWRoIVfVfGY2EiNlYHyMd19153ap3JrzZfPHEBYs/edit?usp=sharing)

</details>

<details>
  <summary><b>Base Network PyTorch to Keras Converting</b></summary>

### Overview of Progress on Task

* Minseung and I have met and discussed the goals specifically for this week. Eric outlined that we should have the following layers converted from PyTorch to Keras with testing to ensure it works. The following lines of code below are in PyTorch:

  ```
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  ```
* Below, I will walk through each line and the parameters that are different. These are the parameters we changed when we went line by line before testing.
* In the Base Network PyTorch to Keras Testing section I will put the code of the model we have come up with so far, and the testing we have done to make sure it is equivalent.

### <code>Conv2D()</code> Layer

* The parameters that have a different meanings from PyTorch to Keras are: <code>in_channels</code>, <code>padding</code>, <code>stride</code>, and <code>bias</code>.
* Specifically for the <code>padding</code> parameter, PyTorch takes in an integer or tuple value while Keras takes in either "same" or "valid".
  * Keras does the padding automatically when the "same" value is set for this parameter.
* Another issue, we have run into is that Keras does not have an <code>in_channels</code> parameter. Keras has a <code>filters</code> parameter that corresponds to the PyTorch <code>out_channels</code> parameter.
  * A fix to this is by using the <code>input_shape</code> parameter which takes in the height, width and the number of input channels. Originally, we had the input channels for the <code>input_channel</code> parameter in place of height which was throwing off the number of params when we were testing the model. 
  * The <code>input_shape</code> parameter is used to specify the shape of the input to the first layer in the model. Therefore, this parameter does not need to be used in any of the other layers once specified. In this case the shape of <code>(height, width, 64)</code> is the same as an input channel of 64. I specified the height and width to both be 1.
* The final parameters <code>stride</code> and <code>bias</code> had easy changes into <code>strides</code> with the same values, and <code>use_bias</code> with the same value, False.

### <code>BatchNorm2d()</code> to <code>BatchNormalization()</code> Layer

* In the Keras implementation, the <code>axis</code> parameter is used to specify the channel axis. The default value is -1 meaning the last axis.
* The <code>momentum</code>, <code>epsilon</code>, <code>center</code>, and <code>scale</code> parameters have the same meaning as in PyTorch.
* Since we were converting the layers one by one, the <code>input_shape</code> parameter used to specify the shape of the input had to be used here as well. When we start building the model, we will not have to include this parameter. This is because it will be specified in the first layer.
* Keras by default computes the mean and variance over the batch axis, so the <code>axis</code> parameter has to be set up accordingly.

### <code>ReLU()</code> to <code>Activation()</code> Layer

* In Keras, you can use the <code>Activation</code> layer with the <code>"relu"</code> activation function to achieve the same effect as the <code>ReLU</code> activation in PyTorch.
* Since Keras does not have an <code>inplace</code> option, this does not need to be specified.

### <code>MaxPool2d()</code> to <code>MaxPooling2D()</code> Layer

* In Keras, the <code>MaxPooling2D</code> layer achieves the same effect as the <code>MaxPool2d</code> layer in PyTorch. 
* The <code>pool_size</code>, <code>strides</code>, and <code>padding</code> parameters have the same meaning in both frameworks.
* The <code>data_format</code> parameter is used to specify the channel axis, which by default is set to <code>channels_last</code>.
* Since we were converting the layers one by one, the <code>input_shape</code> parameter used to specify the shape of the input had to be used here as well. When we start building the model, we will not have to include this parameter. This is because it will be specified in the first layer.
* We think that Keras <code>MaxPool2d()</code> does padding automatically while PyTorch MaxPool2D() allows you to give a custom number. This is not true and I will explain more in the Base Network PyTorch to Keras Testing along with the solution.

[Notes with Minseung on Each Function in SpineNetEx.txt](https://docs.google.com/document/d/1AGOvWl5OaJKkOgrKNid8n2eyBDf10t2YjGRFQll6JxI/edit?usp=sharing)

</details>

<details>
  <summary><b>Base Network PyTorch to Keras Testing</b></summary>

### PyTorch Model

```
import torch.nn as nn
from torchinfo import summary

class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.hidden1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding = (3, 3), bias = False)
    self.hidden2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.hidden3 = nn.ReLU(inplace=True)
    self.hidden4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

model = NeuralNet()
summary(model)
```

### Keras Model

```
from tensorflow import keras

height = 1
width = 1

# we do not know the input shape
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", use_bias=False, input_shape = (height, width, 64)))
model.add(keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05, center=True, scale=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", data_format="channels_last"))

print(model.summary())
```

### Testing

* Minseung, Daniel, and I worked on testing once we researched into the functions and found the original differences.
* Below is the Google Colab we were using to work together.
* We used the <code>summary()</code> function to get the summary for the PyTorch model, and <code>model.summary()</code> function for the Keras model.
  * We used these functions to determine if the models were the same in the beginning comparing the Param # in the model output. However, that is just a measure of the shape, and not the actual error between the models.
* We had to change the approach of testing for the individual layer outputs.
* We give it an input, create the models, setup the input, make predictions, and print the outputs, errors, and the summaries.
  * We use the error and output shape to determine if our conversion was correct.
  * We want the error to be 0.
* When testing the <code>Conv2D()</code> and <code>MaxPooling2D()</code> layer, we found that in Keras the <code>padding</code> parameter only takes in a value of "same" or "valid" while PyTorch takes in a numerical value. We want to emulate the numerical value within Keras, so we can use the <code>ZeroPadding2D()</code> layer with the <code>padding</code> specified as a numerical value. When using this new layer to specify the padding amount, the error went from 1.4 to 0.

[PyTorch to Keras Testing Code](https://colab.research.google.com/drive/1ULGVWYgeRKYzv48Wp79C_PNQLyGsj_N_?usp=sharing)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 02/13/2023 | 02/20/2023 | 02/20/2023 |
| Base Network PyTorch to Keras Converting| Completed | 02/13/2023 | 02/20/2023 | 02/20/2023 |
| Base Network PyTorch to Keras Testing | Completed | 02/13/2023 | 02/20/2023 | 02/20/2023 |

## 6 February 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, February 6th

* NNLearner works up to compiling to the layer list.
  * The unpacking <code>.npz</code> resulted in empty instances.
  * We are still trying to verify if the NNLearner or <code>.npz</code> is responsible.
  * This is the AASCE dataset and Shriner's dataset we are using.
  * We should be able to load the <code>.npz</code> files and see if it is there.
* Decoder code is essentially fully converted from torch to Keras.
  * We are working on packing it into a primitive and verifying that the changes still have valid input and output.
* There's a saved .txt file of the entire pre-trained model structure. We'll use that to base our first seeded individual.
  * The decoder layers are included in the diagram, and the <code>decoder.py</code> takes output from the network and gets corner points.
* NNLearner code and EMADE uses Keras, so that is why we are using Keras.
* Eye on the prize, making sure the implementation matches AASCE implementation.
* We want to run standalone on it. 
* Adding a level of uncertainty on it. Seeing if Keras results match PyTorch. The sooner we can have something outside of EMADE to make sure both implementations match would be a great checkbox.
* If making a big leap by using Keras, want to make sure the results can be reproduced in Keras before handing it over to the architecture. Want to do the sanity check.
* There is a lot of uncertainty because this has never been done before.
* Not sure how easy Keras and PyTorch can be used interchangeably. Dr. Zutty has never done that, so he does not know if it is easy to do.
* The initialization of the model changes the pre-trained model. We will have to do multiple runs and sampling. Should be as simple as changing the seed to 5 instead of 1. Taking the random seed and changing it to 5.
  * This has nothing to do with EMADE, it is purely from the model. 

### Scoliosis Team Meeting, February 9th

* NNLearner has a <code>.npz</code> issue.
* They are running NNLearner on the Shriners dataset Monday.
* Hopefully, we will have it running on some dataset for the upcoming presentations.
* <code>emade-athi</code> is the new cloned branch Lucas made, <code>athi</code> was a mistake and is now deleted.
* We need the size of the original image, right now it is taking in the image size from the terminal, not in the EMADE datapair.
* Not sure if ResNet needs the size of the image too.
* We did a bunch of cropping, to get the images all the same size. We want to tackle the different sizes.
* We are not sure how to pass up image size through the primitive layers up to the decoder.
* If all the images are the same size, we could code it through the terminal and stick it in.
* The EMADE datapair is the neural network output, not sure how the image size would get moved in.
* Data Augmentation for EMADE work as a cross-sub-team with image processing since we are both presenting posters.
* Eric is moving his changes in.
* Nathan has 1 set of inputs and it has been tested against the old set of inputs.
  * It works, but he has not written testing for EMADE.
* This is separate from the decoder, but it will be helpful for everyone.
* The decoder and ResNet, the goal is to be converted by next Monday, February 13th.

</details>

<details>
  <summary><b>Tasking Updates</b></summary>

### PyTorch to Keras with Minseung

* My new task is to write test code for comparing layer primitive outputs between the PyTorch and Keras layers. The tests can take in an EMADE datapair or some PyTorch or Keras tensors with the same underlying values.
* Conversions would have to be done if necessary. Then, I would have to feed them into the two small primitives and verify that their outputs are the same.
* I started with the <code>/home/azureuser/cloudfiles/code/Users/echen89/Vertebra-Landmark-Detection-changed/SpineNetEx.txt</code> path.
  * The text file holds the anatomy of the model.
  * It is organized like a json file.
* This is a daunting task, so going line by line and verifying that the Keras and PyTorch functions have the same parameters is important.
* The PyTorch should not be changed, but the model needs to be converted into Keras.
  * I am taking notes on the side for the specific parameters for each function.
  * I will ask before I made real edits in Azure to make sure I am making the correct edits.
* If the same numbers put in will get spit out for the PyTorch and Keras equivalent. If they are not, then we will play with the Keras parameters, so they are the same as PyTorch.
* Right now, the backbone of the model is being implemented in Azure. Then the actual model would need to be implemented and it has to be converted from PyTorch to Keras. That is the task that Minseung and I are currently working on. We are basically working one step ahead of the group to make sure what we are doing is feasible.

</details>

<details>
  <summary><b>PyTorch to Keras Converting</b></summary>

### Converting PyTorch Model to Keras

* To convert a PyTorch model to Keras, we need to first convert the model architecture, including all the layers, to the equivalent Keras structure, and then convert the model’s weight values to be compatible with the Keras format. 
* Once we have completed these two steps, we can use the same data as in PyTorch to test the converted model in Keras and verify that the results are equivalent.

1. Convert the PyTorch model architecture to Keras.
2. Convert the Pytorch weights to Keras. Extract the weight values from the PyTorch model. Convert these weights to be compatible with the Keras format. Assign the converted weight values to the equivalent layers in the Keras model.
3. Test the converted model in Keras. Load the test data in Keras. Use the "predict" method to generate predictions with the converted Keras model. Compare the predictions generated by the Keras model with the predictions generated by the original PyTorch model to make sure they match.

### Testing Conversion and Comparing Outputs

* To test the conversion from Keras to PyTorch, we can compare the outputs of the two models on the same inputs and verify that they match. Additionally, we can compare the accuracy or other metrics of the two models on a test dataset to confirm that the conversion has been done correctly. You can use ONYX to convert PyTorch models to Keras.

1. **Test Accuracy:** One way to test the correctness of the conversion is to compare the accuracy of the models. We can train the same model in both Keras and PyTorch, and compare the accuracy of the models on the same test dataset.
2. **Visualize the Model Prediction:** Another way to test the conversion is to visualize the model’s prediction. We can visualize the prediction of the model in Keras and PyTorch with the same sample data and see if the results match.
3. **Model Parameter Comparison:** You can also compare the model parameters of the models trained in Keras and PyTorch. We can compare the weight and bias values of the model layers to make sure they match.
4. **Numerical Gradient Check:** We can also perform a numerical gradient check to see if the gradients of the model parameters are computed correctly. This can be done by comparing the gradients of the models in Keras and PyTorch, and making sure that they are close to each other.

### Overall Notes

* I looked through the PyTorch and Keras documentation for each function in the <code>/home/azureuser/cloudfiles/code/Users/echen89/Vertebra-Landmark-Detection-changed/SpineNetEx.txt</code> path.
* It is not a 1-to-1 conversion between PyTorch and Keras. This is currently the struggle that we are dealing with. For now I am still researching each function.
* I found there are models that convert PyTorch to Keras, but an ultimate goal could be to use or create one of our own models for converting PyTorch to Keras.
* In the Google Document below, details about each function are listed. Also, within my personal repository, I have uploaded the <code>.txt</code> file that I converted to <code>.py</code> to work on implementing the PyTorch to Keras. 
* This following week we will be working on continually converting PyTorch to Keras in the file by inputing parameters and making sure they have the same output. As long as we have a few layers working, we can start stacking them together.

[Notes with Minseung on Each Function in SpineNetEx.txt](https://docs.google.com/document/d/1AGOvWl5OaJKkOgrKNid8n2eyBDf10t2YjGRFQll6JxI/edit?usp=sharing)

</details>

<details>
  <summary><b>Self Grading Assessment</b></summary>

### Self Grading Assessment

| Task | Score | Comments |
| ---- | --------------- | ---------------|
| _**Notebook Maintenance**_ | | |
| Name & Contact Info | 5/5 | | 
| Teammate Names and Contact Info Easy to Find | 5/5 | Easy to locate in Resources. |
| Organization | 5/5 | |
| Updated at Least Weekly | 10/10 | |
| _**Meeting Notes**_ | | |
| Main Meeting Notes | 5/5 | |
| Sub-teams' Efforts | 10/10 | I have documented extensively the efforts of the sub-team.| 
| _**Personal Work and Accomplishments**_ | | |
| To-Do Items: Clarity, Easy to Find | 5/5 | |
| To-Do List Consistency (Weekly or More) | 8/10 | I need to be more detailed with the tasks. |
| To-Dos & Cancellations Checked & Dated | 5/5 | | 
| Level of Detail: Personal Work & Accomplishments | 5/15 | Explanations and justifications are present, but my reflections could be improved. Also, I have not been tasked well. Now that I have a task I will have more personal contributions to document. |
| _**Useful Resource**_ | | |
| References (internal, external) | 10/10 | Scoliosis Repository has all of my code files uploaded. Also, Azure is used mainly for our team. |
| Useful Resource for the Team | 15/15 | I supply detailed meeting notes, and notes on all the papers. |

* **Total out of 100:** 88

</details>


### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 02/06/2023 | 02/13/2023 | 02/13/2023 |
| Tasking Update | Completed | 02/06/2023 | 02/13/2023 | 02/13/2023 |
| Begin to Convert PyTorch to Keras | Completed | 02/06/2023 | 02/13/2023 | 02/09/2023 |
| Create Personal Repository for Scoliosis Work | Completed | 02/06/2023 | 02/13/2023 | 02/06/2023 |
| Self Grading Assessment for Notebook | Completed | 02/06/2023 | 02/13/2023 | 02/06/2023 |

## 30 January 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Meeting Monday, January 30th

* We will need to put together some posters for the Shriner’s Children Hospital workshop and poster session, on campus, March 2nd.
* We decided to use Keras for NNLearner implementation.
* **Pull Request Etiquette**
  * Do not remove datasets, it is great to do locally, but as soon as you check it in, we cannot break up any commits.
  * If you have already removed datasets please revert those commits.
  * If you want to keep branches, open PR and it is very easy to merge in.
  * When committing things, pay attention to what you are committing.
  * Try to pay attention as you are going to keep changes clean and organized. Then we can be a lot better about pull requests.
  * Image Processing pulled most recently from Cameron’s branch, but after a certain number of years, the university clears your account in GitHub. We need to find a way to get it back.
  * Try to keep changes atomic. One team removed all primitives they did not want to use, and made changes to the wrappers. Now there are a bunch of primitives that don’t work if the wrappers are reverted.
    * A new wrapper is a good solution or reverting to the way the wrapper is being used currently.
  * It would be great if we could get code back into the Cache_V2 branch.
* Mainly spent our time clarifying how granular we want EMADE to search the neural architecture.
  * The VFL model was broken up into 5 ResNet layers and the Decoder.
    * Original proposition is to move the decoder layer to the eval function so EMADE will just evaluate on the 5 ResNet layers.
   * Still working on getting our version of NNLearner up.
   * We need to convert the VFL repo from PyTorch to Keras. Then we can feed our eval function a valid input.
   * By the time we have NNLearner ready we can move in Keras versions of the VFL model.
   * In 2 weeks time we should be able to have the VFL model in EMADE made out of keras components to evolve on it.
* What is the motivation in moving decoder to evaluation functions?
  * We figured it would be easier. The decoder is taking output from ResNet layers, not as much to be gained from evolving on it as opposed to evolving the ResNet layers.
* Why not put in a decoder that gets coupled to an encoder?
  * Two lists make up the tree. If we include the decoder, the decoder should only end up as the last primitive in the tree.
  * A problem - we are getting out of how EMADE works. EMADE will run whatever function the tree is, send the resulting predictions to the evaluation functions.
  * EMADE will just get the predictions.
  * Another way to approach is fail fast mentality, specific decoder primitive required, in <code>EMADE.py</code> goes to evaluate individual, looks for specific things to be in the tree structure based on the type of data in the input file, xml tag that gets read out of xml file, no learner in tree structure, it becomes invalid, gets marked fatal, if you created an NNEncoder and NNDecoder. We can require that the tree contains a decoder and encoder, we can fail it otherwise.
* Why can't we put the decoder in the eval function?
  * Workers initialize objective functions, check tree for common things, then the tree is built (ties all primitives together into a function). They then iterate through datapairs, and creates a function and sends it against the datapair. At this point it starts to run the primitives. When it comes out at the end, it gets back our <code>returned_dict_result</code>. Pulled off of target for each dataset, the shared dictionary between the two is getting back the classes.
   * For us that result would be the landmark formations. Our evaluation function would have to take in something that is not there same with the truth data.
* We need to try to get a branch set up, if we are better with configuration then there could be better commits to the main branch.
* Goal is to have everyone get EMADE running, so everyone can test their work.

### Scoliosis Meeting Thursday, February 2nd

* We presentation at the beginning of March. We are on a clock now between that and Midterm presentations.
* Eric and Nathan have been converting PyTorch into Keras. They go line by line and find Keras equivalents.
* The tasking has been unclear, so we pinned a To-Do List for both the ResNet and Decoder sub-teams in the channel. People have reacted to which one they are working on. This way we can all know what is getting done.
* The decoder team is converting PyTorch to Keras.
  * Nathan is testing each step, 
   * A problem is that PyTorch expects a numerical value and Keras has two set actions, set and validate.
  * For testing, you make up the inputs and throw it into PyTorch and Keras.
  * There should not be anything that does not have Keras equivalents.
* ResNet
  * NNLearner should be able to hold multiple layers
  * In <code>resnet.py</code> file for the VFL model. There is a list of ResNet models pulled from PyTorch. We will need to pull Keras versions.
    * We are trying to go for layers
  * <code>neuralnetworkmethods.py</code>
    * ResBlock and input layers, we are thinking of using those methods, wrapping them into primitives, and putting them together. We will begin unit testing the primitives after they are created. All of this happens after NNLearner is set up.
  * We need to make sure the layers of NNLearner all connect in ways we expect.

### Sub-Team Meeting with Brian, February 3rd

* AASCE model has a decoder and ResNet.
* We are trying to work on ResNet.
  * Going for layers with a neural network.
  * We cannot wrap them in a primitive boiler plate. They have to go into the NNLearner. Then once the layered list is registered, it can be added to EMADE.
* First we are trying to get NNLearner to work. While Patrick works on debugging NNLearner, it is better to help by making primitives.
* The idea of unit tests, would be to have a set of expected outputs for inputs, neural network layers are matrix multiplication.
  * A list of numbers to a list of numbers.
* The two main tasks are NNLearner and primitive development.
* The input for NNLearner, we are still discussing then we would be able to put it all together.
* Cache_V2 is the beginning of EMADE, and the NNLearner vehicle is best. It makes primitives work in the context of neural network.
* Our current setup is a pipeline, the stuff at the front needs to work in order to work to the end.
* I should now choose a layer, understand it, and create a wrapper for a primitive.

</details>

<details>
  <summary><b>ResNet Implementation in VFL/AASCE Model</b></summary>

### ResNet in VFL/AASCE Model

* They start by importing URLs for resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, and wide_resnet101_2.
  * For resnet18 through resnet152:
    * They are used and loaded as Deep Residual Learning for Image Recognition.
    * If pretrained bool is True, it returns a model pre-trained on ImageNet.
    * If progress bool is True, it displays a progress bar of the download to stderr.
  * For resnet50_32x4d and resnext101_32x8d:
    * These are used and loaded as Aggregated Residual Transformation for Deep Neural Networks.
    * If pretrained bool is True, it returns a model pre-trained on ImageNet.
    * If progress bool is True, it displays a progress bar of the download to stderr.
  * For wide_resnet50_2 and wide_resnet101_2:
    * These are Wide Residual Networks. 
    * These models are the same as ResNet except for the bottleneck number of channels which is twice larger in every block. The number of channels in outer 1x1 convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048 channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    * If pretrained bool is True, it returns a model pre-trained on ImageNet.
    * If progress bool is True, it displays a progress bar of the download to stderr.
* There are three main classes <code>BasicBlock</code>, <code>Bottleneck</code>, and <code>ResNet</code>.
  * In all of these classes there is a <code>forward</code> function which changes the output for each class. Within the <code>ResNet</code> class it appends layers and returns the <code>feat</code>.
  * The <code>Bottleneck</code> class seems to create channels for each layer, and there is a limit which is outputted within the <code>forward</code> function and the <code>out</code> variable.

[ResNet Code in VFL Repo](https://github.com/yijingru/Vertebra-Landmark-Detection/blob/master/models/resnet.py)

</details>

<details>
  <summary><b>Layer from nn-resnet Branch in EMADE</b></summary>

### Chosen Layers

* Brain is working on the <code>ResBlock</code> layer.
  ```
     def ResBlock(filters, layerlist):
      # empty_layerlist = LayerList()
      # addlist = [Add()]
      # addlist = addlist + [arg.mylist for arg in argv]
      # empty_layerlist.mylist.append(addlist)
      # return empty_layerlist

      layerlist.mylist.append(('resblock',filters))
      return layerlist
  ```

* I chose to work on the <code>ConcatenateLayer</code>. 
  ```
     def ConcatenateLayer(*argv):
       empty_layerlist = LayerList()
       concatlist = [Concatenate(axis=-1)]
       concatlist = concatlist + [arg.mylist for arg in argv]
       empty_layerlist.mylist.append(concatlist)
       return empty_layerlist
  ```
  * The goal of the above method is to concatenate the argument passed in. It returns <code>empty_layerlist</code> which is used in almost every method.
  * Inside the method it first creates an <code>empty_layerlist</code> which is a <code>LayerList()</code>.
  * <code>Concatenate()<code> is imported from Keras. It is a layer that concatenates a list of inputs.
    * It takes as input a list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.
  * The method then loops through the <code>argv</code> and adds it to the <code>concatlist</code>.
  * Finally, the <code>concatList</code> is appended onto the <code>empty_layerlist</code> which is what is then returned.

[nn-resnet Branch from EMADE](https://github.gatech.edu/emade/emade/blob/nn-resnet/src/GPFramework/neural_network_methods.py)
  
</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 01/30/2023 | 02/06/2023 | 02/05/2023 |
| Research on ResNet34 Implementation in VFL/AASCE Model | Completed | 01/30/2023 | 02/06/2023 | 02/05/2023 |
| Pick a Layer of ResNet | Completed | 01/30/2023 | 02/05/2023 | 02/05/2023 |

## 23 January 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### Scoliosis Meeting Thursday, January 26th

* Scoliosis has 11 people on the team now.
* The VFL branch, if basing off of this we should do layered list.
* The **nn-resnet** branch.
  * Traditional EMADE datapair way of adding primitives.
  * Last semester, they were adding primitives in <code>gp_framework_helper.py</code>.
  * If basing our ResNet branch off of **nn-resnet** we should do the EMADE datapair.
  * This way is more flexible. Each layer is a primitive, no layered lists.
* Image processing does a 4 dimensional layered list.
* We must hook up these neural net layers to a decoder.
* The output of the ResNet layer, not sure if it is compatible with the decoder, but it is something to think about going forward.
* Short-term goal is to improve on ResNet, and to make the primitives and throw in the VFL model as a primitive. EMADE will develop it.
* We are trying to make layers of ResNet.
* Inside the master branch of the VFL Git Repository:
  * <code>spinalnet.py</code>
    * decnet is a network not a decoder.
  * <code>resnet.py</code>
    * Pull existing layers already available.
    * Ideally we can do that.
    * Forward functions and different classes.
      * Forwarding through individual locks.
      * There are components in ResNet18.
      * It would make sense for those to be primitives we are working with to look like, line 147. I will go into more detail in the next section in my notebook.
        * There are some invalid values, but you can expose these things to the genome, if you go by the layers that is fine.
    * Need to find out what the distinction is between ResNet18 and ResNet34, if we end up doing that and things break, we would not know how much that would change it.
    * If not interested in modifying the architecture of ResNet we could block layers into primitives.
    * The <code>Bottleneck</code> section could be its own primitive, line 78.
* We need to look for an implementation using Keras, the way that they are doing it we can do it as well.
* Which components are we trying to isolate as a primitive?
  * The analog for a primitive in the VFL model would be each individual ResNet34 layer.
  * The complete VFL-like-primitive would have 5 ResNet players and 3 decoder layers.
* When choosing primitives, they should correspond to layers or sections of the neural network.
* We could split it out into blocks of layers and that would expose different layers themselves.
* The more you expose the more likely it will break.

</details>

<details>
  <summary><b>ResNet and Keras Research</b></summary>

### ResNet
* DecNet Portion
  * The Combination Module Class:
    ```
       self.up = nn.Sequential
         nn.Conv2D(c_low, c_up, kernel_size=3, padding=1, stride=1)
         nn.BatchNorm2D(c,up)
         nn.ReLU(inplace=True)
       self.cat_conv = nn.Sequential
         nn.Conv2D(c_up*2, c_up, kernel_size=1, stride=1)
         nn.BatchNorm2D(c_up)
         nn.ReLU(inplace=True)
     ```
  * DecNet has 3 decoder layers, specified as Combination Modules.
    ```
       self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
       self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
       self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
    ```
  * Network makes some initializations based on "heads". 
    * "heads" specified in the Network Class of <code>train.py</code> as:
      ```
         heads = {'hm': args.num_classes,
                  'reg': 2*args.num_classes,
                  'wh': 2*4,}
      ```
  * The two branches we are looking at have different methods of adding their neural network primitives. The image processing branch uses the 4dimLayerList, but the EMADE ResNet branch is using datapairs and is registering primitives in the traditional way.
  * We think we should be basing it off of Patricks branch.

[AASCE VFL Model Branch](https://github.com/yijingru/Vertebra-Landmark-Detection/tree/master/models)

[EMADE ResNet Branch](https://github.gatech.edu/emade/emade/blob/nn-resnet/src/GPFramework/neural_network_methods.py)

### Keras

* Keras is an API designed for humans. Offers consistent and simple APIs, minimizing the number of user actions required for common use cases and provides clear and actionable error messages.
* The most used deep learning framework.
* Built on top of TensorFlow 2, can scale large clusters of GPUs or an entire TPU pod.
* Used by CERN, NASA, NIH, and other scientific organizations around the world.

[Keras](https://keras.io/)

### Keras vs. PyTorch

* Deep learning and machine learning are a part of artificial intelligence. Deep learning is a subset of machine learning.
* Deep learning imitates the human brain's neural pathways in processing data, using it for decision-making, detecting objects, and translating languages.
* Deep learning processes machine learning by using a hierarchical level of artificial neural networks. This lets machines process data using a nonlinear approach.
* Keras is an effective high-level neural network Application Program Interface (API) written in Python. This open-source neural network library is designed to provide fast experimentation with deep neural networks, and it can run on top of CNTK, TensorFlow, and Theano.
* PyTorch is a relatively new deep learning framework based on Torch. Used mainly for natural language processing applications.
* TensorFlow is an end-to-end open-source deep learning framework developed by Google and released in 2015. Used for neural networks and is best suited for dataflow programming across a range of tasks. Offers multiple abstraction levels for building and training models
* Both PyTorch and Keras are good if you’re just starting to work with deep learning frameworks. Mathematicians and experienced researchers will find PyTorch more to their liking. Keras is better suited for developers who want a plug-and-play framework that lets them build, train, and evaluate their models quickly. Keras also offers more deployment options and easier model export.

[PyTorch vs. Keras](https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article#:~:text=PyTorch%20vs%20Keras,-Both%20of%20these&text=Keras%20is%20better%20suited%20for,and%20has%20better%20debugging%20capabilities.)

</details>

<details>
  <summary><b>Defining Goal and Understanding Project</b></summary>

### Goal with ResNet and Decoder Implementation
  <img width="622" alt="VIP Image of Goal" src="https://github.gatech.edu/storage/user/60851/files/78709813-c21c-41c1-8d06-adf89d6754af">

* The goal is to put the ResNet portion and Decoder portion into EMADE and have EMADE develop the layers in order to improve the already working VFL model from the AASCE challenge.
* The next step for me is to choose a specific layer and start developing it into a primitive.
* Patrick and Lucas are working on testing the functionality of NNLearner to get it operational.
* The rest of us on the sub-team need to get an understanding of the ResNet architecture, how the VFL/AASCE model uses ResNet34 to get meaningful information for the decoder. 
* Also, start breaking apart the VFL/AASCE model and make it into something EMADE can evolve on. This will meaningfully provide information for the decoder to use.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 01/23/2023 | 01/30/2023 | 01/29/2023 |
| More Research on ResNet and Keras | Completed | 01/23/2023 | 01/30/2023 | 01/29/2023 |
| Goal for Scoliosis | Completed | 01/23/2023 | 01/30/2023 | 01/29/2023 |

## 16 January 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### Scoliosis Meeting Thursday, January 19th

* This was a discussion on how we move forward.
* We are shifting gears far away from what we were doing last semester. Still we are making sure everyone has azure access for this semester.
* We looked back at the VFL (Vertebra Focused Landmark) model, and ResNet (Residual Networks) is inside of it already.
  * The VFL model uses ResNet to get the landmarks and feeds the output into a decoder to get the final output.
  * The layers of the neural network are very well documented within the paper.
* The Image Processing GPFramework helper has the neural network methods.
  * From what Eric understands the way they set it up, they used Keras, and the VFL model uses PyTorch.
  * The structure is weird, instead of having each layer be a primitive, each primitive takes in a list of layers, what the neural net is up until that point.
  * A bunch of primitives that pass in a growing neural net and at the top they give it the image to perform what it needs to do.
  * We are not sure why they have a layered list wrapper.
    * Looks like there is a 3 and 4 layer list.
    * Since we are only working with image data, the output is not an image.
* We discussed preferences between torch and Keras.
  * We have Keras and PyTorch in Azure right now. Most of the EMADE branches use Keras, so we decided to use Keras.
* **Short-Term Goals**
  * First thing to do is build the VFL model from the ground up and duplicate their results.
    * The VFL model is structured as a ResNet portion and then it is fed into the decoder portion. We decided to split the work up with those two portions.
    * We are assuming what the Image Processing team worked at some point.
    * The part of our team that works on ResNet will need more people.
  * We want to focus on the primitives, and neural network methods for Image Processing. We will probably lift a lot of those and use them in our model.
* What are we trying to evolve with EMADE?
  * Change ResNet and decoder, build both out of neural network layers, EMADE would add more, and we would try to improve on the model that way.
* ResNet
  * The neural network ResNet branch already has a ResNet builder class used to build custom ResNet architecture.
    * We do not know if it works, but it is a good starting point.
* ResNet: Lucas, Patrick, Charlotte, Ruchi, and Brian
  * Goal is not to come up with the full primitive, set it as a 2 week goal, ideally by then having a working ResNet primitive.
* Decoder: Minseung, Daniel, Nathan, Pranav, and Eric
  * Ideally two weeks for the decoder.
* Similar process to last semester, writing unit tests to make sure the primitive works.
* We need to be sure to fit the ResNet layers together, and we need to make sure there are compatible unit tests.
* The Google Doc below is where we will add ideas in the upcoming week.

[Scoliosis Team Google Doc](https://docs.google.com/document/d/1bxkiiBsz4bTi3snZy1-0ATkDZpc1rev_Wg49HKiJ5GE/edit#heading=h.c5mtf9g5ai9j)

</details>

<details>
  <summary><b>Revisiting Vertebra Focused Landmark Model</b></summary>

### Key Summary

* Existing regression-based methods for the vertebra landmark detection typically suffer from large dense mapping parameters and inaccurate landmark localization.
* Segmentation-based methods predict connected or corrupted vertebra masks.
* Their vertebra-focused landmark detection method first localizes the vertebra centers, based on which it then traces the four corner landmarks of the vertebra through the learned corner offset. The comparison results demonstrate the merits of their method in both the Cobb angle measurement and the landmark detection on low-contrast and ambiguous X-ray images.
* To deal with the large variability and the low tissue contrast in X-ray images, supervised learning-based methods are developed, they use structured Support Vector Regression (SVR) to regress the landmarks and the Cobb angles directly based on the extracted hand-crafted features.
* The backbone of the vertebra-focused landmark detection network is from ResNet34. The sizes of the feature maps are presented as height * width * channels.
* Cobb angles are determined by the location of the landmarks.
* The X-ray image contains 17 vertebrae from the thoracic and the lumbar spine. Each vertebra has 4 corner landmarks, and each image has 68 landmarks in total.
* They used the training data, 580 images, of the public AASCE MIC-CAI 2019 challenge as their dataset. The images are all anterior-posterior X-ray images.
* The strategy of predicting center heatmaps enables their model to identify different vertebrae and allows it to detect landmarks robustly from the low-contrast images and ambiguous boundaries.

### Implementation

* They implemented their method in PyTorch with NVIDIA K40 GPUs (Computing Processor Graphic Cards). The backbone network ResNet34 is pre-trained on ImageNet.
* To reduce overfitting, they adopted standard data augmentation, including random expanding, cropping, contrast and brightness distortion.
* The network is optimized with Adam with an initial learning rate of 2.5 * 10^-4. They trained the network for 100 epochs and stop when the validation loss does not decrease significantly.

[VFL Model](https://arxiv.org/pdf/2001.03187.pdf)

[AASCE VFL Model Implementation](https://github.com/yijingru/Vertebra-Landmark-Detection/blob/master/models/resnet.py)

[Image Processing ResNet Implementation](https://github.gatech.edu/emade/emade/blob/nn-resnet/src/GPFramework/neural_network_methods.py)

</details>

<details>
  <summary><b>ResNet</b></summary>

### Residual Networks (ResNet) - Deep Learning

* When we increase the number of layers, there is a common problem in deep learning associated with that called Vanishing/Exploding gradient. This causes the gradient to become 0 or too large. Increasing the number of layers, the training and test error rate also increases.
* In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Blocks.
* There is a technique called skip connections. 
  * It connects activations of a layer to further layers by skipping some layers in between forming a residual block.
* ResNets are made of stacking residual blocks together.
* If any layer hurt the performance of architecture then it will be skipped by regularization, so this results in training a very deep neural network without the problems caused by vanishing/exploding gradient.
* Using the Tensorflow and Keras API the ResNet architecture can be designed from scratch. 
* In the GeeksforGeeks website below there is an implementation of ResNet using Keras.

[ResNet GeeksforGeeks](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

### ResNet34 Roboflow

* The ResNet34 model was released December 10, 2015.
* ResNet34 is a state-of-the-art image classification model, structured as a 34 layer convolutional neural network and defined in "Deep Residual Learning for Image Recognition". 
* ResNet34 is pre-trained on the ImageNet dataset which contains 100,000+ images across 200 different classes.
* ResNet is different from traditional neural networks in the sense that it takes residuals from each layer and uses them in the subsequent connected layers.

[ResNet34 Roboflow](https://roboflow.com/model/resnet-34#:~:text=What%20is%20Resnet34%3F,images%20across%20200%20different%20classes.)

### ResNet34 Kaggle

* Deeper neural networks are harder to train. A residual learning framework eases the training of networks that are deeper than those used previously.
* The residual networks are easier to optimize and gain accuracy from considerably increased depth. 
* On the ImageNet dataset residual nets were evaluated with a depth up to 152 layers.
  * The ensemble of these residual nets achieved 3.57% error on the ImageNet test set and won first place on the ILSVRC 2015 classification task.
* A pre-trained model has been previously trained on a dataset and contains the weights and biases that represent the features of whichever dataset it was trained on.
* Pre-trained models are beneficial, by using it you can save time. Someone else already spent the time and compute resources to learn a lot of features and your model will benefit from it.

[ResNet34 Kaggle](https://www.kaggle.com/datasets/pytorch/resnet34)

  <img width="310" alt="ResNet34" src="https://github.gatech.edu/storage/user/60851/files/926ebad9-51a5-4eb2-9adf-eda0a0431365">

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 01/16/2023 | 01/23/2023 | 01/21/2023 |
| Revist the Vertebra Focused Landmark Model | Completed | 01/16/2023 | 01/23/2023 | 01/22/2023 |
| Research on ResNet | Completed | 01/16/2023 | 01/23/2023 | 01/23/2023 |

## 9 January 2023

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Team Monday, January 9th

* This week is the beginning of the semester, and we are discussing the future of the Scoliosis team.
* We decided our weekly meeting time will be Thursdays at 5-6pm or 6-7pm. Also, we will have our first meeting this next week.
* From what we discussed during class on Monday, January 9th, I believe we are still going into depth in the Scoliosis field, but looking to get results this semester. We want to basically complete the AASCE challenge to get a Cobb angle and landmarks by improving an already existing model.

</details>

<details>
  <summary><b>Image Processing Branch</b></summary>

* We are looking to explore the neural network parts of the image processing team and apply that to our project having to do with x-ray images of scoliosis and finding landmarks to calculate the Cobb angle based off of them.
* From what I can see by exploring the Image Processing Branch, they are using a PACE dataset/model.
* In the datasets folder, there are many added programs to gather images from the internet. For example, in the chestxnet folder there is a python file that downloads the images and checks the checksums.
* I will be asking my teammates this week about how to properly explore the Image Processing Branch and what to look for.

[Image Processing Branch](https://github.gatech.edu/emade/emade/tree/Image-Processing(nn-vip))

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 01/09/2023 | 01/16/2023 | 01/16/2023 |
| Look at the Image Processing Branch | Completed | 01/09/2023 | 01/16/2023 | 01/16/2023 |

# Resources

| Sub-team Members - Scoliosis |
| -------------------------- |
| [Tariq Almawash](https://github.gatech.edu/emade/emade/wiki/Notebook-Tariq-Almawash) |
| [Aditya Neeraj Chandaliya](https://github.gatech.edu/emade/emade/wiki/Notebook-Aditya-Neeraj-Chandaliya) |
| [Eric Chen](https://github.gatech.edu/emade/emade/wiki/Notebook-Eric-Chen) |
| [Brandon Davis Conner](https://github.gatech.edu/emade/emade/wiki/Notebook-Brandon-Davis-Conner) |
| [Yahya Hassan](https://github.gatech.edu/emade/emade/wiki/Notebook-Yahya-Hassan) |
| [Minseung Jung](https://github.gatech.edu/emade/emade/wiki/Notebook-Minseung-Jung) |
| [Pranav Malireddy](https://github.gatech.edu/emade/emade/wiki/Notebook-Pranav-Malireddy) |
| [Ruchi Patel](https://github.gatech.edu/emade/emade/wiki/Notebook-Ruchi-Patel) |
| [Aarav Aeraj Shah](https://github.gatech.edu/emade/emade/wiki/Notebook-Aarav-Aeraj-Shah) |
| [Adil Shaik](https://github.gatech.edu/emade/emade/wiki/Notebook-Adil-Shaik) |
| [Patrick Haoran Weng](https://github.gatech.edu/emade/emade/wiki/Notebook-Patrick-Haoran-Weng) |
| [Lucas Dong Hyuk Yim](https://github.gatech.edu/emade/emade/wiki/Notebook-Lucas-Dong-Hyuk-Yim) |
| [Daniel You](https://github.gatech.edu/emade/emade/wiki/Notebook-Daniel-L-You) |
| [Pujith Veeravelli](https://github.gatech.edu/emade/emade/wiki/Notebook-Pujith-Veeravelli) |
| [Brian G Zhang](https://github.gatech.edu/emade/emade/wiki/Notebook-Brian-G-Zhang) |
| [Nathan Zhong](https://github.gatech.edu/emade/emade/wiki/Notebook-Nathan-Zhong) |

[**Personal Scoliosis Repository**](https://github.gatech.edu/chettrich3/chettrich3Scoliosis.git)

[**Scoliosis Team Google Doc Spring 2023**](https://docs.google.com/document/d/1bxkiiBsz4bTi3snZy1-0ATkDZpc1rev_Wg49HKiJ5GE/edit#heading=h.c5mtf9g5ai9j)

[**VFL Model**](https://arxiv.org/pdf/2001.03187.pdf)

[**AASCE VFL Model Branch**](https://github.com/yijingru/Vertebra-Landmark-Detection/tree/master/models)

[**EMADE ResNet Branch**](https://github.gatech.edu/emade/emade/blob/nn-resnet/src/GPFramework/neural_network_methods.py)

[**Notes with Minseung on Each Function in SpineNetEx.txt**](https://docs.google.com/document/d/1AGOvWl5OaJKkOgrKNid8n2eyBDf10t2YjGRFQll6JxI/edit?usp=sharing)

[**PyTorch to Keras Testing Code**](https://colab.research.google.com/drive/1ULGVWYgeRKYzv48Wp79C_PNQLyGsj_N_?usp=sharing)

[**Scoliosis Sub-Team Folder**](https://drive.google.com/drive/folders/1-T1Fe77Qui8N--jE_5ABh1CTLRmhR2W4?usp=share_link)

[**Scoliosis Poster for the Shriner's Poster Session**](https://docs.google.com/presentation/d/1VO0NWRoIVfVfGY2EiNlYHyMd19153ap3JrzZfPHEBYs/edit?usp=sharing)

[**Scoliosis Midterm Presentation Spring 2023**](https://docs.google.com/presentation/d/1AimgsfLxqCzFjXL5zdzpKi6f41qIMHYOWELXr0Jena4/edit?usp=sharing)

[**Scoliosis Final Presentation Spring 2023**](https://docs.google.com/presentation/d/1PyhmQvF3Y53qIJXIGeG8v7GHpvdVqYhZPSQS6dpgAs4/edit?usp=sharing)

</details>

# 2022 Fall Notebook Entries

<details>
  <summary><b>2022 Fall Notebook Entries</b></summary>

## 5 December 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Meeting Monday, December 5th

* Model primitives are not a feasible goal before finals; we're realistically going to be able to only optimize our image primitive set for the Vertebra Focused Landmark model.
* Unit testing also revealed some inconsistencies in our .npz file packing; this has all been resolved.
* Image/Model primitive compatibility is facing some difficulties in that different primitives require different conda environments. Our current band-aid solution under consideration for addressing this issue is to save and reference primitive outputs so that the primitives can access the data. Creating a consolidated environment would be ideal but will likely be reserved as a next semester task.
* Our priority tasks are:
  * Writing unit tests for feeding an image through our custom primitives and verifying the input/output at each step.
  * Creating the .xml for actually running EMADE.
  * Perform the rudimentary EMADE run for single model primitive this week.

</details>

<details>
  <summary><b>Personal Notes on Presentation</b></summary>

### Script for Slide 32 of Scoliosis Final Presentation

* In Method-2, they implemented a two-stage system, which extracts 68 rough points along the spine curves using a Simple Baseline. They then generated patches and made sure each of them contained three vertebrae at most based on ground truth. Then they trained a second Simple Baseline to predict the exact key points of these patches. A post process of clustering is used to deal with dense key point predictions due to the freedom of patch selections. 
* Simple Baseline I can grasp the global implicit sequentiality of key points through generating corresponding heat maps simultaneously with fixed order. We use them as the outline sketches of spines to generate patches so as to force the model to focus on local information in a certain range of vertebrae.
* A patch includes n points for 1 to 3 vertebrae.
* Patches acquired by the patch process are used to detect key points by Simple Baseline II.
* Key points predicted by Simple Baseline II ranges from 4 to 12.
* To summarize the final 68 key points from 4 vertex groups of points, they proposed a post process to cluster and remove outliers.
* The highly robust Method-2 grasps the global implicit sequentiality of key points and makes it much easier to match a template for any adjacent vertebrae because of various patch samples.
* Method-2 avoids the disadvantages of Method-1.
* For the Fusion method, they chose one result from Method-1 and two results from Method-2 to fuse which led to the competitive result in the AASCE 2019 challenge.

[Final Scoliosis Presentation](https://docs.google.com/presentation/d/1FVFZFrdj3Xgtz5NXpPXL5gO6i-7ftTiqRhQwHypXXTY/edit#slide=id.p)

</details>

<details>
  <summary><b>Final Presentation Notes</b></summary>

### NLP

* NLP is a subfield of ML and AI. 
* QA is a subset of an NLP task called Information Retrieval.
  * Fact-based
* NNLearner QA is a custom version of NNLearner.
  * Two input layers
* Goal is to see if EMADE can evolve an existing state of the art model.
  * High performance Keras model implemented by the Keras team.
* Ran EMADE and they could not increase the accuracy of the individual.
* For this semester they wanted to expand search space via refinement of GP parameters and add more capabilities within NL.
* Database Analysis
  * There were lots of errored individuals.
* Functionality Testing
  * Manually generated four high-performing individuals, used all as seeded individuals.
  * Tested multi-seeding capabilities of EMADE, first in standalone evaluator, then in database evaluator.
  * Problem arises when a run dies and some individuals are left incompletely evaluated.
* On the first full run of EMADE, 24 hour run (3 8 hour runs), 19 Pareto optimal individuals were generated.
* Integrate state-of-the-art pre-trained models from Hugging Face into EMADE as primitives.
* Created a Generalized Data Converter for BERT-like models.
* ALBERT Model
  * Extension to the traditional BERT model with performance increase.
  * Reduction in memory usage by parameter reduction techniques.
* Tested primitive results with standalone tree evaluator.
* RoBERTa
  * A Robustly Optimized BERT Pre-trained Approach
  * Also, tested with the standalone tree evaluator.
* PyTorch Models
  * PyTorch models are not compatible with NNLearnerQA
  * NNLearnerQA creates and compiles a Keras/TensorFlow model.
  * PyTorch modules cannot be added into a TensorFlow without converting architecture first.
  * This is a possible next semester task.
* GloVe + Naqanet
  * Naqanet uses a more recent version of <code>numpy</code> than what they were working on which led to dependency issues.
  * Were able to make an arbitrary layer because of Naqanet. 
  * For the future, need to implement custom layer (Variable or Lambda) using Naqanet or custom GloVe model.
* DeBERTa
  * NLP model based on transformer architecture.
  * Future BERT model.
  * Used a tiny version of DeBERTa because of PACE constraints.
* Experiments
  * Ran several trials of EMADE to see if AutoML could produce a Pareto from and improved on seeded individuals.
  * Albert individual dominated RoBERTa.
  * EMADE produced 13 Pareto individuals.

### Stocks

* New Sources of Data
  * Original data set exclusively consisted of our price data
  * Want to buy at low point and sell at high point.
  * Added technical indicators, ratios, and accounting data.
  * Settled on including Volume, RSI, and OBV.
    * Volume is the number of shares of stock traded that day.
    * RSI measures whether a stock is overbought or oversold.
    * OBV is a cumulative total on volume.
* Price Trend Prediction
  * Data Pre-Processing
    * Original data consisted of entries at different scales.
  * Big improvement over midterms, AUCs were generally lower. Some graphs look really good, with lots of individuals and a significant gap between the random chance line and the Pareto front.
  * They need to evaluate an individual's profit and loss rates.
* Reinforcement Learning
  * Develop a reinforcement learning primitive, had to migrate a model to EMADE.
  * The weights represent allocations of money across the universe of stocks.
  * Worked on building evolution functions.
  * For the primitive they are working on a proof of concept. 
  * Needed NPZ files, created new pickled data that copies our data to the target.
  * Using "Deep Learning for Portfolio Optimization" Keras model.
    * Consists of an LSTM layer, followed by a dense layer with a softmax activation.
    * Model is designed to train on all of the data in the dataset at the same time.
  * Want evaluation functions to be able to judge different individuals.
  * Implemented the Sortino Ratio, similar to the Sharpe Ratio. Key difference is only using the standard deviation of the downside risk.
* Looking into the Treynor Ratio Exploration.
  * The Beta measures how a single stock is doing relative to the stock market as a whole.
* Future Work
  * Price Trend Prediction
    * Deep dive into results, and look into why some sectors perform better than others.
  * Implement more evaluation functions such as the Treynor Ratio.
  * Complete a run of EMADE on the portfolio optimization tasks.

### Image Processing

* Problem
  * Chest x-ray set with instances of 14 different diseases as well as x-rays with no problems. They are trying to train models through EMADE to correctly identify diseases without input from a radiologist.
* Score of 0.5 means that the algorithm cannot differentiate between class at all.
* Recap from Last Semester
  * Multi-objective problems are still difficult especially with 15 objectives.
  * Data leakage in the old datasets.
  * Goal is to get away from 0.5 on all metrics.
* New Primitives
* Wrote the script to parse ADFs to regenerate trees.
* ADF script can take in a Pareto front CSV, parse trees into format compatible with standalone.
  * Removes the ADFs.
* In a previous semester, added DEAP's implementation of Lexicase to EMADE. A problem, DEAP's implementation is different from the pure form of Lexicase.
  * Lexicase is a selection method designed to help solve highly modal problems.
  * Modal problems require qualitatively different solution methods for different inputs.
* Dataset
  * Dataset was really large, there were a lot of classes that were unbalanced.
  * Before running EMADE, there was a pre-processing script to sample and reprocess images. This ensures the same image does not end up in the train and testing sets.
* Goal is with the old code, wanted to see if increasing the mutation rate will help with the evolution.
  * In Old Code, Got to 39 generations. Trees did not perform exceedingly well. 
  * In New Code, maybe performed slightly better than the Old Code.
* Originally an elite pool of individuals were added to the gene pool every generation. 
* Replaced elite pool with parents to prevent evolution from getting stuck.
* ResNet
  * ResNet uses skip layers to help provide an alternative path for the gradient, allowing them to add more layers and get better performance.
  * Works well for classifying images.
  * Using neural network methods in EMADE and some residual block code to build a ResNet tree.
* Transformer in Image Processing
  * Build custom patch encoder for Transformers.
  * Implementing different layers in EMADE.
  * In the future, use Chexpert to train the model before fine-tuning it on Chexnet to do transfer learning.
* Created a new analysis script, looks at how trees in the Pareto front change over generations.
  

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 12/05/2022 | 12/11/2022 | 12/05/2022 |
| Create Script for Final Presentation | Completed | 12/05/2022 | 12/11/2022 | 12/09/2022 |
| Final Presentation Notes | Completed | 12/05/2022 | 12/11/2022 | 12/09/2022 |

## 28 November 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Meeting Monday, November 28th

* EMADE has come to a stopping point. We need more permissions for the SQL Server. Right now we can write the tables, but we cannot delete the tables. This is a problem because EMADE writes and deletes the table every time. 
* Changes from the original Cache-V2 EMADE branch will be moved over to ensure future compatibility without Azure support in the next week.
* Model primitives are still moving slowly, as we are facing various issues with how models accept data.
* Image primitives are moving faster, and they are expected to be able to be tested with EMADE datapairs for unit tests within this week.
* We have primitives that we can write the unit tests for and test them this week.
* The Vertebra-Focused Landmark Detection for Scoliosis Assessment has pre-trained weights and is more promising, so we will be building this up this week.
* Look for other open database tools, if we do not get the Azure SQL stuff up and running.
* Testing the image primitives will be our first look into verifying our data was packed correctly.

### Scoliosis Meeting Friday, December 2nd

* We need to let EMADE run and let it build.
* Our first task as a group is to start with the Presentation.
* Patrick moved the files into EMADE rather than creating a path.
* Brian is working on the primitive and is almost done.
* We just want to get a Cobb angle.
* Method that is being called by some of the other unit tests is not working on ours because it is expecting a numpy array.
  * We need to repackage a numpy array or we write a new data loading method for the unit tests.
* Right now, Nathan is running one singular image through the method and after it finishes cropping, EMADE gets saved as data pair object and he converts to image to see if the cropping worked.

</details>

<details>
  <summary><b>Working on the Presentation</b></summary>

### Current Goals with Presentation

* Minseung and I have been assigned slides 31 and 31 to talk about the Accurate Automated Keypoint Detection paper and future steps on how we could make a primitive from this paper next semester.
* Attached below are links with our compiled notes, the old presentation they had from the Midterm, and the newest presentation.
* Creating a primitive is very hard, and other team members are struggling with it and they have been working in this group longer than us. Therefore, my main task is to have a clear understanding of Method-2 in the Accurate Automated Keypoint Detection paper and to be able to explain that to the group during the final presentation.
* We both found the GitHub for this paper Accurate and Lightweight Keypoint Detection and Descriptor Extraction. It is slightly different from the Accurate Automated Keypoint Detection paper, but we can probably look at their code for inspiration. We have yet to find the GitHub for the Accurate Automated Keypoint Detection paper, I checked the contributions section, and the specific authors. They probably took it down or archived it. I will continue searching; however for now we can use the other similar GitHub we found.
* Currently, I have finished my slide. I put in a picture and an overall description of the steps within Method-2. I will work on a script as a task for next week to prepare for the final presentation.

[Accurate and Lightweight Keypoint Detection and Descriptor Extraction](https://github.com/Shiaoming/ALIKE)

[Compilation of Notes on Accurate Automated Keypoint Detection Paper](https://docs.google.com/document/d/1sIFtVmSCMcqt47dRMHxCqr3cIYEmV3wDf9dBwPghcss/edit?usp=sharing)

[Old Scoliosis Presentation](https://docs.google.com/presentation/d/1FVFZFrdj3Xgtz5NXpPXL5gO6i-7ftTiqRhQwHypXXTY/edit#slide=id.p)

[New Scoliosis Presentation](https://docs.google.com/presentation/d/1FVFZFrdj3Xgtz5NXpPXL5gO6i-7ftTiqRhQwHypXXTY/edit#slide=id.p)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 11/28/2022 | 12/05/2022 | 12/02/2022 |
| Compile Notes on Accurate Automated Keypoint Detection with Minsung | Completed | 11/28/2022 | 12/05/2022 | 12/03/2022 |
| Work on Presentation for Final | Completed | 11/28/2022 | 12/05/2022 | 12/05/2022 |
| VIP Final Peer Evals | Completed | 11/28/2022 | 12/05/2022 | 11/30/2022 |

## 21 November 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### VIP Meeting Monday, November 21st

* The SQL team decided that the lost MSSQL compatibility with the new Cache-V2 branches of EMADE means we should move over to traditional MySQL for running EMADE on Azure.
* Model team members are training and implementing pre-trained models to act as primitives.
* Locating the repositories of certain papers are difficult. We may need to implement based off of purely paper description of model architecture.
* Austin noticed some ImageNet work done by the Seg4Reg people, PSPNet + DenseNet pipeline is trained on ImageNet because AASCE is too small.

</details>

<details>
  <summary><b>Notes on Guide to Adding Primitives</b></summary>

### How to Add Primitives into EMADE by Austin Dunn

* First, we need recreate the wrapper classes EMADE uses.
* The class cannot be instantiated on its own, so in order to run the line of code below we need to make a subclass of RegistryWrapper and implement the abstract method "register".
  * <code>my_wrapper = RegistryWrapper()</code>
* We need a wrapper for registering the primitives into the primitive set AND a wrapper for returning the results of a primitive.
* Below is the custom registry wrapper.
  ```
  class MyRegistryWrapper(RegistryWrapper):
    """
    This wrapper is a standard registry wrapper for primitives
    Used by signal_methods, spatial_methods, and feature_extraction_methods

    Stores a mapping of primitives used in generating the PrimitiveSet

    The first object of input_types must be a EmadeDataPair

    Args:
        input_types: common inputs for every primitive stored in the wrapper
                     used to create a mapping between arg and index
                     example mapping created: {'EmadeDataPair0': 0, 'TriState1': 1}

    """
    def __init__(self, input_types=[]):
        super().__init__(input_types)

    def register(self, name, p_fn, s_fn, input_types):
        # create wrapped method
        wrapped_fn = partial(primitive_wrapper, name, p_fn, s_fn)

        # create a mapping for adding primitive to pset
        self.registry[name] = {"function": wrapped_fn,
                               "types": input_types}

        # return wrapped method
        return wrapped_fn
  ```
* The method is now implemented and the superclass (RegistryWrapper) handles the constructor method.
* **primitive_wrapper** is the wrapper method we defined to process the numpy arrays.
* **name** is a simple string of the primitive name.
* **p_fn** stands for primitive function or method, method modifying passed in numpy arrays.
* **s_fn** stands for setup function or method, optional method.
* Most important part is the <code>partial</code> method call. It returns a new callable method with the name, p_fn, and s_fn arguments constant every time the method is called.

* **Standard Primitives**
  * Helper Method
    * register: told the registry this primitive has no setup method, and old the registry this primitive requires an integer.
    * doc : required for EMADE documentation. This method does not have the typical def <code>my_add():</code> most python methods have, so we have to add our documentation to the <code>__doc__</code> attribute of the method for our documentation to work.
    * my_add_helper: EMADE's registry wrapper allows us to use the same helper method for both primitives defined above. The primitive_wrapper used by both primitives will call <code>my_add_helper</code> during EMADE's individual evaluation.
  * Utilizing a Setup Method
    * This primitive uses both a helper method and a setup method.
    * All primitives are required to have a helper method.
    * Setup methods only run once, so we do not have to recalculate the value of the fraction for every numpy array in the data list.
  * Multiple Data Objects
    * In order to operate on more than one data object, we use this method. 

* **Abnormal Primitives**
  * Machine Learning Methods
    * LearnerType: This is a custom object the learner wrapper requires. It selects a random learner from a list of valid ones with a constant set of initial parameters.
    * ModifyLearner: This is a special primitive which modifies the parameters of a LearnerType.
    * Estimator Objects: these are the classifier objects you typically see in scikit-learn such as <code>DecisionTreeClassifier</code>. EMADE creates this object based on the parameters stored in <code>LearnerType</code>.

* **Fit-Transform Methods**
  * Some primitives such as PCA, K-means clustering, and feature selection methods fit and transform on an entire dataset.
  * A new registry wrapper and primitive wrapper is needed.

[How to Add Primitives into EMADE by Austin Dunn](https://github.gatech.edu/emade/emade/blob/CacheV2/notebooks/How%20to%20Add%20Primitives%20into%20EMADE.ipynb)

[Advanced EMADE Primitives by Dr. Jason Zutty](https://github.gatech.edu/emade/emade/blob/CacheV2/docs/advanced-EMADE/writing-primitives.html)

[Guide to NLP Primitives](https://github.gatech.edu/emade/emade/wiki/Guide-to-NLP-Primitives)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 11/21/2022 | 11/28/2022 | 11/28/2022 |
| Read Guide to Adding Primitives | Completed | 11/21/2022 | 11/28/2022 | 11/28/2022 |

## 14 November 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### Sub-Team in Scoliosis Meeting Thursday, November 17th

* This meeting was about understanding the goal of what we are trying to accomplish with writing primitives and the direction we are going.
* Seg4Reg research paper, implementing parts of the paper.
  * Ways they extract images.
  * Breaking that into primitives we can use in EMADE for genetic programming.
* Two Approaches:
  * Take the model and use that as a primitive, takes in an image input and Cobb angle output.
  * Segmentation (vertebra) and regression (Cobb angle).
* Before the midterm presentation, we were able to get the code base for EMADE into Azure, able to adapt the evaluation functions.
  * Must set up SQL primitives.
  * Write the primitives.
    * We want to keep the basic primitives and add on more complex primitives.
  * How to Develop Individuals
    * Primitives that do image cropping.
    * Also, develop models that serve a certain paradigm.
    * Finding the four corner points on every vertebrae using trigonometry.
    * Have individuals just output the Cobb angle.
* Primitives
  * Choice on what we want to implement.
    * Take pieces and sizes and figure out how we want to put it all together.
  * Lay the groundwork for a future better EMADE run.
  * Take a model that inputs an image and outputs a Cobb angle.
    * Image preprocessing images, may improve the chances of getting more accurate Cobb angle measurements.
    * More options on how we want to alter hyper-parameters.
    * Train models outside of EMADE and bring these into EMADE.
    * Primitives layer in neural networks, similar to image processing.
  * Want to build toward a neural network.
  * Selecting now is how to preprocess the input, how we want to do the segmentation and regression.
  * Probably will end up continuing with the broad approach because the data we have been provided is only the Cobb angle.
    * We do not have the truth data to properly pursue the segmentation and landmark approach.
  * More granularity with the primitives.
* Trying to create the most basic primitives, incorporate a primitive that takes in the Seg4Reg model.
* Seg4Reg has one for segmentation and one for regression, must take entire model for a primitive.
  * Instead of having two primitives, to ensure capability, going straight from image to Cobb angle in a compound approach.
* Eric posted in the slack channel different papers on different models.
  * New Task: look into the Accurate Automated Keypoint Detection paper.
* Some models that take in the x-ray image of the spine, get the four corner landmarks. Then they go from the corner landmarks to the Cobb angle using trigonometry.
* Two networks combined to get the final image and there is an angle calculation.
* How do we know if the Cobb angle calculated is accurate?
  * Shriners marked each of the x-rays with what they believe the Cobb angle to be.
* The goal is to give Shriners a model that will work on their images and give them a Cobb angle.

### Scoliosis Meeting Friday, November 18th

* In order to run EMADE in your compute node, download <code>git</code> and <code>git lfs</code> in your personal compute node.
* The Seg4Reg model does not have pretrained weights.
* Some models might already be trained.
* When writing primitives, we should compare the learner methods wrapper classes approach.
* The three main approaches:
  * Tutorial written by Austin to make boiler plate code and at the very end make a method that makes calls to your method (CacheV2Branch), uses ResNet.
* Image processing primitives done seem to be entire models but layers of neural networks.
* Long term stretch goal: make modes more granular, a lot of models we found are all neural network based, do a similar approach to image processing.
* We do not know what our exact neural network inputs and outputs would be.
* We do not have the intermediate truth data with the landmarks identified by the radiologist.
* Try to get the predetermined weights and use those with the existing model to make some function calls.
* Brian's model is also an AASCE model, should be a lot less painful than the landmark detection model.
* Patrick is still trying to build a UNet model, but thinking of transitioning to Versa instead. He can add a raw UNet primitive, but it is functionally obsolete. Trying to replicate what the papers did for scoliosis.
* Lucas is still working on the SQL server, but will probably shift to MySQL for better compatibility.
* Already a pre-trained model for Versa in the Docker image. The entire Versa is not built for x-ray images. Flattened all of the 3D images. Took the largest vertebra and made it all into a 3D cube. Might be useful to test data.
* Regression model takes in an image and outputs a Cobb angle. Essentially a model for angle calculation. Segmentation does not change the size of the image. If there is a segmentation base model it inputs an image and outputs another image but labels which pixels are vertebra and which are not. Possibly implement a pair of primitives but would essentially be a model. We do not have any truth data for segmentation.
* If trained on a public dataset, the models might work, we need to find a public dataset.
* New goal is to look into the Accurate Automated Keypoint Detection for a new primitive, might need the pre-trained weights.

</details>

<details>
  <summary><b>Notes on Seg4Reg and Accurate Automated Keypoint Detection</b></summary>

### Seg4Reg Networks for Automated Spinal Curvature Estimation

* **Abstract and Introduction**
  * Seg4Reg contains two deep neural networks focusing on segmentation and regression. Based on the results generated by the segmentation model, the regression network directly predicts the Cobb angles from segmentation masks. 
  * To alleviate the domain shift problem appeared between training and testing sets, they conduct a domain adaptation module into network structures.
  * Deep neural networks have got amazing achievements in various image classification tasks.
  * BoostNet is proposed as a novel framework for automated landmark estimation, which integrates the robust feature extraction capabilities of Convolutional Neural Networks (ConvNet) with statistical methodologies to adapt to the variability in X-ray images.
  * To mitigate the occlusion problem, MVC-Net (Multi-View Correlation Network) and MVE-Net(Multi-View Extrapolation Net) have been developed to make use of features of multi-view X-rays.
  * Currently there are two ways to estimate the Cobb angles:
    * Predicting landmarks and then angles.
    * Regressing angle values.
  * The first way is able to produce high-precision angle results but relies heavily on the landmark predictions.
  * The second way is more stable but may lack the ability to generate precise predictions.

* **Proposed Method**
  * Their process is constructed by two networks: one for segmentation and the other for regression. The architecture of the segmentation network is similar to PSPNet while the regression part employs traditional classification models.
  * They apply histogram equilization to both sets to make them visually similar. Considering the limited number of testing images, they manually cropped the X-rays to remove the skill and keep the spine in the appropriate scope.
  * For segmentation, they built the groundtruth masks on top of offered landmarks' coordinates. Adding another class "gap between bones" helped the segmentation model perform the best. 
  * Followed the instructions to design the segmentation network. After the feature extractor, PSPNet utilized different pooling kernels to capture various receptive fields. 
  * Also, append the dilated convolution with different dilation rates to the pooling pyramid.
  * They used ResNet-40 and ResNet-101 as the basic feature extractor.
  * They directly employed recent classification networks to perform the regression task. ImageNet based pretraining was used because they found it helped a lot under limited training samples.

* **Experimental Results**
  * They reported their experimental results in both local validation and online testing sets. They did not use cross validation.
  * Adding a dilation pyramid thus improves the performance of previous PSPNet.
  * During the model ensemble stage, they assigned different weights to different weights to different model outputs considering their validation scores. They mainly ensemble ResNet series, DenseNet series and EfficientNet series.

[Seg4Reg Paper](https://link.springer.com/chapter/10.1007/978-3-030-39752-4_7)

### Accurate Automated Keypoint Detection for Spinal Curvature Estimation

* **Abstract and Introduction**
  * In order to estimate the spinal curvature, they proposed two methods to detect the spinal key points at first.
    * In Method-1, they used a RetinaNet to predict the bounding box of each vertebra followed by a HR-Net to refine the key point detections.
    * In Method-2, they implemented a similar two-stage system, which first extracts 68 rough points along the spine curves using a Simple Baseline. Then they generate patches and make sure each of them contains three vertebrae at most based on ground truth. They train a second Simple Baseline to predict the exact key points of these patches, which are not fixed numbers.

* **Methodology**
  * Method-1 is based on RetinaNet and HR-Net. It has two stages, the first is to detect vertebrae and to generate corresponding bounding boxes using RetinaNet. Those bounding boxes are used to train the RetinaNet from the 68 key points demanding each box covers 4 key points of an individual vertebra.
  * In the second stage, they use HR-Net to detect the 4 key points of the bounding box.
  * In Method-2, they train Simple Baseline I with spinal images to detect all 68 key points in a spinal sample. Simple Baseline I can grasp the global implicit sequentiality of key points through generating corresponding heat maps simultaneously with fixed order.
  * The predicted key points can smoothly trace the curvature of most spines, even though their landmarks are not accurate enough to calculate Cobb angles. They use them as the outline sketches of spines to generate patches so as to force the model focus on local information in a certain range of vertebrae.
  * A patch includes n points, for one to three vertebrae. Patches are randomly captured in multiple times within a certain vertebrae range. An image sample produces hundreds of patches.
  * Patches acquired by Patch Process are used to detect key points by Simple Baseline II. For each patch, the number of key points predicted by Simple Baseline II varies from 4 to 12, corresponding to the patch range from 1 to 3.
  * The final 68 key points from 4 vertex groups of points, they proposed a post-process to cluster and remove outliers.
  * DBSCAN clustering is applied respectively to 4 vertex groups of points. Cluster within the same groups are sorted in ascending order based on the Y-axis coordinate of cluster centers. The first and the last cluster centers of the 4 vertex groups are then assessed.
    * If the Y-axis coordinate order or the angles of adjacent centers are un-normal, the corresponding clusters are excluded. 

* **Experiments and Results**
  * Method-1
    * For RetinaNet, they follow to experiment with ResNet-50-FPN backbone.
    * Strategies are proposed for the output of RetinaNet to drop the outlier boxes and keep the remaining boxes stay in the spine line.
    * After bounding box detection, the two IoU of current box with the previous adjacent box and posterior adjacent box will be calculated. If they are larger than the threshold, the current box will be dropped.
    * The distance between the upper left vertex of current box and previous box is also compared with the distance between the upper left and upper right vertexes of current box, if larger, the current box will also be dropped.
  * Method-2
    * They use ResNet-152 backbone for Simple Baseline I.
    * The Simple Baseline II result from ResNet-153 for further research.
  * Different from the usual clinical method, we noticed that the angle measurement method provided by organizers is so sensitive and complex that deviation in few pixels can cause great error of angles.
  * They tried different angle measurement ways, including fixing the first and the last vertebrae for Proximal-Thoracic (PT) and Thoracic-Lumbar (Tl) angles.
  * They chose one result from Method-1 and two results from Method-2 to fuse. The results from Method-2 differ from each other only in the post-processing squeezed vertebrae assessment. 

* **Conclusion**
  * Method-1 implemented a RetinaNet for the vertebrae bounding box detection, with an HR-Net for local key point detection. The vertebrae detection based workflow is easy to implement and works efficiently for FOV with unfixed number of vertebrae. Without sequential modeling, it is hard to balance the precision and the recall.
  * Method-2 cleverly avoids the disadvantages in Method-1 with a highly robust system, but it needs to design a suitable post-processing and therefore not easy to implement. They first use a Simple Baseline for rough key point detection to sketch the spine curve globally. Then they train a Simple Baseline on patches to predict exact local points with an unfixed number. The highly robust Method-2 grasps the global implicit sequentially of key points and makes it much easier to match a template for any adjacent vertebrae.

[Accurate Automated Keypoint Detection](https://link.springer.com/chapter/10.1007/978-3-030-39752-4_6)

### Discussion of Future Tasks

* Below is a link to the full paper. I am currently trying to find the GitHub repository they used. I will check section 1.2 in the paper, the contributions section and see if I can find it this next week.
  * [Scoliosis Detection using Deep Neural Network](https://arxiv.org/pdf/2210.17269.pdf)
* After finding the repo, I will attempt to make a primitive to add to the EMADE run.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 11/14/2022 | 11/21/2022 | 11/19/2022 |
| Read Seg4Reg | Completed | 11/14/2022 | 11/21/2022 | 11/21/2022 |
| Read Accurate Automated Keypoint Detection for Spinal Curvature Estimation | Completed | 11/14/2022 | 11/21/2022 | 11/21/2022 |

## 7 November 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### Scoliosis Meeting Friday, November 11th

* Seg4Reg model, following the tutorial about building the wrappers.
  * [Seg4Reg](https://link.springer.com/chapter/10.1007/978-3-030-39752-4_7)
  * Only really use model from scikitlearn, trying to find other models.
* Set a testing flag for every primitive and save the output somewhere.
  * Only if we get EMADE running.
* Better plan might be to take a dozen primitives and manually run the primitives on an individual we create themselves.
  * There is a folder in GPFramework/UnitTests which is a flag we can set to check the primitives.
* Visualizing the output is important.
* We need to write the UnitTests in the first place, do not need a full EMADE run.
* A lot of models have intermediate steps.
  * Example: centerline and then landmarks found from the centerline.
* Other groups used NN-Learner and used other models as primitives as layers to NN-Learner.
* Would need to import AASCE model. We do not have to train it, just need the waits and the model itself.

### Implementing Model-Based Primitives

* Monday and throughout this week we will discuss next steps in implementing model-based primitives into EMADE for our team, Scoliosis.
* All code is from the scoliosis-CacheV2 branch.
* The links below have been resources Austin Peng and Brian Zhang have been using as a part of their research.
  * [Seg4Reg Research Paper](https://link.springer.com/content/pdf/10.1007/978-3-030-39752-4.pdf)
  * [Guide to Adding Primitives Dr. Zutty](https://github.gatech.edu/emade/emade/blob/CacheV2/notebooks/How%20to%20Add%20Primitives%20into%20EMADE.ipynb)
  * [Example NLP Primitives](https://github.gatech.edu/emade/emade/wiki/Guide-to-NLP-Primitives)
  * [Primitive Examples](https://github.gatech.edu/emade/emade/blob/scoliosis-CacheV2/src/GPFramework/learner_methods.py)
  * [Related UnitTests](https://github.gatech.edu/emade/emade/blob/scoliosis-CacheV2/src/GPFramework/UnitTests/learner_methods_unit_test.py)

</details>

<details>
  <summary><b>About Azure and Fixing Access</b></summary>

### Meeting with Coleman Hilton Wednesday, November 9th

* We had a meeting with Cole our contact with Shriner's Hospital for Children to get Azure fixed.
* An overzealous IT person removed us from the directory which is a good problem to have for them. However, we were not able to access any of the resources.
* He resent us the invitation for access and added us to the resources and now all the new members have full access to Azure.

### About Azure

* Microsoft Azure is Microsoft's public cloud computing platform.
* It can be used for services like analytics, virtual computing, storage, networking, and more.
* Azure allows for flexibility, advanced site recovery, and built-in integration.
* It backs up data in any language, on any OS, in any location.
* Shriners Children's Hospital uses Azure to hold their data and we specifically use the Machine Learning Studio to house our project.
* Also, Azure allows for two factor authentication to ensure security which was initially part of our problem with maintaining access on Azure.

### How-To Access Files

* In order to access files, within Azure click on "RohlingML". From there we click on the "Studio web URL".
* Now the ML Studio is launched, and from the tabs on the left Notebooks and the Compute Nodes can be accessed.
* Eric walked us through how to create Compute Nodes as these allow us to make edits to data. We have to be careful how long we leave them on as it is $0.90 cents an hour to run them.
* The Notebooks are where everybody's files are, and where I uploaded the xml file Minseung and I worked on.
* When the Compute Node is running, it is much easier to work on code within VSCode. One of my first tasks was to copy Eric's files to my User within Azure. 
  * I started my Compute Node, clicked on edit with VSCode and ran this command within the terminal <code>cp –r Users/echen89/Vertebra-Landmark-Detection-changed Users/chettrich3/Vertebra-Landmark-Detection</code>. 
  * This took about 20 minutes, but it copied all of the files over to my folder.

</details>

<details>
  <summary><b>xml File Creation for Scoliosis Team</b></summary>

* Minseung and I worked on creating the xml file for our EMADE run.
* We have a SQL Server within Azure now, and under Networking we were able to find the settings to add into our xml file.
  * Rule Name: ClientIPAddress_2022-10-31_15-48-40
  * Start IPv4 Address 128.61.2.170
* Also, we had access to an email chain with Cole where certain user logins were created.
  * Lyim: s7MA5dBcLegTiTN 
  * Echen: 1qAz2WsXDE3
  * Apeng: plm0oKn9ijB
  * Nzhong: ijn9Uhb7ygv
  * Bzhang: Wsx3edcRfv
* The IP address and the user logins were helpful with the dpconfig section within the xml file.
* We did not know the selection algorithm or how many splits the data would need and the titles of each file, so we did not complete those sections yet. 
* We spoke to Eric about the completion of the xml file and some questions we had, and he said that what we did was great and to work on a new project now.
* I uploaded the xml file in my personal compute node in Azure now.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 11/07/2022 | 11/14/2022 | 11/11/2022 |
| Get Azure Working and Familiarize with Layout | Completed | 11/07/2022 | 11/11/2022 | 11/11/2022 |
| Create xml File for Scoliosis Team | Completed | 11/07/2022 | 11/14/2022 | 11/11/2022 |

## 31 October 2022

<details>
  <summary><b>Team Meeting Notes - Scoliosis</b></summary>

### General Summary

* I joined the Scoliosis team within Automated Algorithm Design.
* The first task was to get Azure access within Shriners Children's Hospital Organization.
  * I was able to gain access on Tuesday, November 1st; however, during our sub-team meeting Friday, November 4th I lost access and the link no longer works.
  * We are currently working through this issue, and we may have to email Cole if the problem persists until Monday, November 7th.
  * We started working on this issue during the sub-team meeting.
* The second task was to go through the ScoliosisDoc and familiarize myself with the New Member portion.
* I read through the Vertebra-Focused Landmark Detection for Scoliosis Assessment and took notes as well as looked at the AASCE challenge website.
* Once I get access to Azure, I will be able to help with adding primitives to EMADE from other models.
* Since, the new members just finished their 10 week Bootcamp and working with EMADE, we should be able to help with adding primitives.

[ScoliosisDoc](https://docs.google.com/document/d/1C2O5fehahl7I6nq9i8tg8LO3b3wMiWS4-hyHCIJk1gc/edit)

</details>

<details>
  <summary><b>Notes on Vertebra-Focused Landmark Detection for Scoliosis Assessment and Accurate Automated Spinal Curvature Estimation</b></summary>

### Vertebra-Focused Landmark Detection for Scoliosis Assessment

* **Introduction**
  * Adolescent idiopathic scoliosis (AIS) is a lifetime disease that arises in children. 
    * AIS is a lateral deviation and axial rotation of the spine that arises in children at or around puberty.
  * Estimation of Cobb angles of the scoliosis is essential for doctors to make diagnosis and treatment decisions.
    * It would also decrease the need for surgery.
  * Cobb angles are measured according to the vertebrae landmarks.
    * Cobb angles are used for assessment, diagnosis, and treatment. It is measured based on the anterior-posterior (AP) radiography (X-ray) by selecting the most tilted vertebra at the top and bottom of the spine.
    * Measurement is challenging due to the ambiguity and variability in the scoliosis AP X-ray images.
  * Regression-based methods for the vertebra landmark detection typically suffer from large dense mapping parameters and inaccurate landmark localization.
  * Segmentation-based methods tend to predict connected or corrupted vertebra masks.
  * They propose a novel vertebra-focused landmark detection method.
    * Their model localizes the vertebra centers, based on which it then traces the four corner landmarks of the vertebra through the learned corner offset.
    * The method can keep the order of the landmarks.
    * Comparison of the results demonstrate the merits of this method in Cobb angle measurement and landmark detection on low-contrast and ambiguous X-ray images.
  * S-squared VR uses structured Support Vector Regression (SVR) to regress the landmarks and the Cobb angles directly based on the extracted hand-crafted features.
  * Due to the dense mapping between the regressed points and the latent features the input image has to be downsampled to a very small resolution to enable training and inference.
  * Keypoint-based methods localize the points without dense mapping.
    * Simplifies the network and is able to consume the higher-resolution input image.

* **Method**
  * To first localize the vertebrae by detecting their center points.
  * Capture the 4 corner landmarks of each vertebra from its center point, so they can keep the order of the landmarks.
  * They use ResNet34 conv1-5 to extract the high-level semantic features of the input image.
  * Then they combine the deep features with the shallow ones to exploit both high-level semantic information and low-level fine details.
  * At layer D2, the heatmap is constructed, center offset and corner offset maps using convolutional layers for landmark localization.

* **Heatmap of Center Points**
  * The keypoint heatmap is generally used to pose joint localization and object detection. 
  * For each point k, its ground-truth is an unnormalized 2D Gaussian disk.
  * The radius is determined by the size of the vertebrae.
  * The variant of the focal loss is used to optimize the parameters.

* **Center Offset**
  * The output feature map of the network is downsized compared to the input images.
  * Reduces the imbalance problem between the positive and negative points due to the reduced output resolution.
  * Use the center offset to map the points back to the original input image.
  * The center offsets at the center points are trained with L1 loss.

* **Corner Offset**
  * When the center points of each vertebra are localized, we trace the 4 corner landmarks from the vertebra using corner offsets.
  * Corner offsets are defined as vectors that start from the center and point to the vertebra corners.
  * The corner offset map has 4 x 2 channels.
  * L1 loss is used to train the corner offsets at the center points.

* **Implementation**
  * Used the training data of the public AASCE MICCAI 2019 challenge. All images are AP X-ray images.
  * 60% of dataset was used for training, 20% for validation, and 20% for testing.
  * The Cobb angle is calculated using the algorithm provided by AASCE.
  * Implemented their method in PyTorch with NVIDIA K40 GPUs. The backbone network ResNet34 is pre-trained on ImageNet.
  * To reduce overfitting, we adopt the standard data augmentation, including random expanding, cropping, contrast and brightness distortion.
  * Following the AASCE Challenge, we used the symmetric mean absolute percentage error (SMAPE) to evaluate the accuracy of the measured Cobb angles.

* **Results and Conclusion**
  * The vertebra-focused method achieved the best performance in both Cobb angle measurement (SMAPE) and the landmark detection.
  * The strategy of predicting center heatmaps enables the model to identify different vertebrae and allow it to detect landmarks robustly from the low-contrast images and ambiguous boundaries.

[Vertebra-Focused Landmark Detection](https://arxiv.org/pdf/2001.03187.pdf)

### Accurate Automated Spinal Curvature Estimation (AASCE)

* This challenge was used to investigate (semi-)automatic spinal curvature estimation algorithms.
* Evaluation was based on the symmetric mean absolute percentage error (SMAPE).
* The Vertebra-Focused Landmark Detection method used the AASCE data set for their training and testing. Their data consisted of 580 public images.

[AASCE](https://aasce19.grand-challenge.org/)

[AAD-Vertebra-Focused-Landmark](https://github.gatech.edu/echen89/AAD-Vertebra-Focused-Landmark/tree/echen89)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Scoliosis | Completed | 10/31/2022 | 11/07/2022 | 11/04/2022 |
| ScoliosisDoc - New Member Onboarding | Completed | 10/31/2022 | 11/07/2022 | 11/04/2022 |

## 24 October 2022

<details>
  <summary><b>Team Meeting Notes - Midterm Presentations</b></summary>

### Bootcamp Team 5

* Data Pre-Processing
  * Assigned an integer to the prefix of each name, ordered them by their own preference on who survived.
  * Standardized the data.
* Folded the data using cross_validator_predict().
* Used a strongly typed approach where inputs were floats, outputs were floats.
* Algorithm processes the data and creates function that uses these inputs.
* Lower output values result in higher likelihood.
* FPR and FNR were used as fitness values.
* Stuck with the functions that scaled consistently with input size.
* Mate, Mutation, and Tournament → experimented with different types that made the model more optimal.
* Used tournament selection of size 4, after saw a setback because it is not multi-objective.
* Used the same preprocessing in EMADE.
* Used the number of false positives and number of false negatives.
* Pareto front comparison, EMADE is an improvement to the MOGP evolutionary algorithm approach.
* Used 32 generations in EMADE, MOGP had around 60 generations.

### NLP

* NLP is Natural Language Processing, subfield in ML
  * Want a computer to understand the contents of the document and the answer to the question being posed.
* QA (Question Answering)
  * Context: passage
  * Question: can find answer to in the context
* Closed domain and extracted because you can find the answer within the message
  * Gave EMADE ability to replicate an existing state of the art model.
    * High performance keras model was used.
* No valid results, individuals were failing completely.
* Looking into the GP parameters and adding more capabilities within ML.
* Push EMADE towards creating individuals that outperform the seeded individual through matings and mutations.
* Looked through database of last semester’s runs.
  * Lots of errored individuals.
  * NNLearner’s to mate/mutate with.
* Mutation functions were not exploring the search space in the right directions.
  * Wrote a mutation script to test the functionality of different mutation functions manually.
* GloVe
  * Another technique developed by Stanford
  * Unsupervised learning algorithm for obtaining vector representations/embeddings for words.
  * Trying to use GloVe to load the pre-trained word vectors, and converting QA passages to those embedding.
  * GloVe does not fit, dimensions are off, when trying to run last years model it got stick on connection setup.
  * Troubleshooting by comparing model summaries of last semester’s seeded individual and identifying dimension mismatches.
* Current goal is to Integrate Hugging Face Transformers as EMADE Primitives
* ALBERT Model
  * Extension to traditional BERT model with performance increases
* Continue to work on final mutation functions, new students will add new Huggingface primitives and help run their experiments.

### Scoliosis

* Adolescent Idiopathic Scoliosis (AIS) - most common type; affects children ages 10-18.
* Angle of 10 degrees is defined as scoliosis, want to make it easier to find the angle because there are different treatments for different angles.
* Localize each vertebrae, segmentation sensitive to image quality, difficulty separating attached vertebrae.
* Shriner’s Dataset is the ultimate goal to run EMADE using AASCE dataset and pull the generated model out and run Shriner’s dataset on that to get Cobb Angle predictions.
* Cropping
  * Initial Cropping, manually altered the images.
  * Does a lot better job of taking out the extra information, wanted to improve the cropping more as it improves the model.
  * Chose the top, left, bottom, and right most bits and took the average, so the Shriner’s data can be cropped in a similar way.
  * Additional change because the image was so zoomed out it would identify the jaw as the edge-most vertebrae, excluded the top three and. bottom three vertebrae.
  * Cobb angle was 55.4 degrees and the model gave 51 degrees.
* Edge Sharpening
  * Shriner’s images had much less defined spine and vertebrae when compared to AASCE, edge sharpening techniques improved it.
  * Computes the histogram for the region around each pixel, so it enlarges the regions overall.
* Used SMAPE as the Evaluation Function
  * Preferred over MAPE for scoliosis measurements on Cobb Angles which creates an upper bound and lower bound.
* Azure
  * Using Azure Compute to perform our preprocessing
  * Created an Azure SQL Database under advice of contact at Shriners
  * Based on Microsoft SQL rather than MariaDB
* U-Net
  * Popular deep convolution network
  * Builds off of fully convolutional networks
  * Initially made for cell segmentation, adapted for vertebrae segmentation.
  * Employs technique to segment objects of same class via their border.
* Seg4Reg
  * Uses Neural Network to find the segmentation masts, and then perform vertebrae segmentation followed by the Cobb Angle
* Future Work
  * Adding Primitives to EMADE
  * Shriner is providing 700 more images, many images need additional metrics, if possible after getting Cobb Angles there are more metrics used to diagnose scoliosis.

### Image Processing

* Problem → chest x-ray set of 14 different diseases as well as x-rays with a variety of problems.
* Goal is to train models through EMADE to correctly identify diseases.
* Last Semester
  * Beat results from the ChestXNet Paper with EMADE result.
  * Could not reproduce dataset, ROC was 0.5 everywhere.
    * Found out that the hashes in the test set were the same as the train set, now figuring out how to solve it the natural way.
* Ran into conflicts with running EMADE Standalone Evaluator
* CPU vs GPU Testing
  * Tested whether or not using the GPU would change the results of the standalone runs, found that it did not make much of a difference.
  * Going to make a Virtual Environment to double check.
* Grad-CAM Automation
  * Generating Grad-CAM images was a cumbersome process involving saving models, extracting test images, and running a script.
  * Wanted to streamline things by automatically generating images when running standalone.
  * Add a new command line flag.
* ADF Tree Parser
  * Tree visualization script creates better visual when there are no excess parentheses.
    * Gives a bunch of blank blocks.
  * Recursively substitute ADF definitions in the tree string.
* Preprocessing
  * Before running EMADE, the images need to be preprocessed, includes sampling, resizing, and making sure images do not appear in both training and testing sets.
* ResNet in Image Processing
  * Known to work extremely well for classifying images.
  * Trying to use it with EMADE to process images.
* Transformers in Image Processing
  * Mainly used for NLP tasks but recently got good scores in image processing as well.

### Bootcamp Team 4

* Data Preprocessing
  * Identified features by impact on Survival rate.
  * Removed Siblings and Parents/Children.
  * One-Hot encoded features.
* Correlation Matrix
  * Showed the most important features relating to survival.
    * Sex has most effect on survival.
    * Military did not have as much of an effect on survival.
* MOGP
  * Decided to make every feature into a Boolean.
  * Used greater than median as a deciding factor.
  * A lot of these points are clustered, over 200-300 points because they did not check the threshold of the distances of the points from each other.
    * MOGP was better than ML.
* EMADE
  * Most of the individuals had fitness infinity, but as it kept running individuals the fitness was decreasing.
  * Python 3.10 did not work, but 3.7 was most stable.
  * A lot of AdaBoost Learners throughout the EMADE database result.
    * Ensemble learning technique.
      * Uses stumps instead of decision trees.
    * Hard to tune hyper-parameters.
  * There were some non co-dominant individuals within the original Pareto frontier.
  * Ran EMADE multiple times, did not change the train and test data with their preprocessed data originally.
  * Ran for 33 generations, the last generation Pareto front with all of the Pareto fronts were very similar.
* MOGP yielded the best results, EMADE is hard to run, AdaBoost is a good classifier.

### Bootcamp Team 1 - My Group

* The number of Pareto optimal individuals decreased over time at some points and this makes sense because in some generations an individual is generated that beats out all of the other individuals.
* Possibly change the input_titanic.xml to calculate the area under the curve (AUC).

### Stocks

* Returning team, started the literature review, made a decision on how to approach the problem. Got everything into EMADE.
* Completely new stock team, previous team published a paper that was accepted to GECCO.
* Model worked on “trading point prediction”.
* Portfolio Optimization Problem
 * New team was interested in Portfolio Optimization
   * Did an intense literature review.
 * Reinforcement learning is difficult to replicate in EMADE.
* Goal is to predict the thread of the next day based off data from the look-back window of the stock price data for the past 50 days.
  * Trained models on each sector, since stocks in different sectors have distinct behaviors.
* Got free data from yahooFianance
  * Paywall meant for Wall Street firms
* Preprocessing
  * For every 50 Day Windows of EOD
    * Open/Close/Adjustments
    * High/Low
* Labeling/Implementation Detail
  * Need to add labels to the data, 0 or 1 depending on whether or not the price went up or down.
  * Convert to EMADE Stream Data
  * KFold Procedure
* Results
  * The Pareto front learners do not seem to perform much better than random chance.
  * AUC’s between 0.5 and 0.7 is in the high sector.
* Short-Term Future
  * Runs have been a starting point, passing in basic data and get our individuals that might not be extremely effective.
  * Add new learner types to EMADE.
  * Add in EMADE Primitives.
  * Improve and implement preprocessing.
  * Experiment with time scales.
  * Improve the data.
* Long-Term Future
  * Implement a regression-based or reinforcement learning model.
  * Target “true” portfolio optimization - risk-adjusted return, not price prediction.
  * Train with alternative data.

### Bootcamp Team 2.5

* Preprocessing
  * Names have a lot of information.
  * Visualization of the data, missing the age and cabin data.
    * Did research, SAVE WOMEN AND CHILDREN became the motto.
  * Correlation Heat Map
    * Women were found to be saved the most.
  * Changed to One-Hot Encoder, they normalized the scalar data.
* Machine Learning
  * Ran a lot of ML Algorithms.
  * Did not all select their own model, chose 20 of the most popular ML algorithms.
* Genetic Programming
  * Made custom primitives.
  * Max value was 17 which might have still been large.
  * Ran loop over 80 generations.
  * Added in tree height.
* Developed a SVM Model
* EMADE
  * Data Splitter, generated zipped data files.
  * Connected with SQL, they had a really good run, and it was less than the ML and MOGP models. EMADE had the lowest area under the curve (AUC).
  * Did not use the actual FPR and FNR but used a rate by dividing it into the total number of samples.
  * A lot faster in calculating the AUC.

### What I am Interested In

* I have a background in finance, so I could bring a lot to the table with the Stocks team. Also, with my forecasting experience, I have some equations that might be helpful.
* Also, I think that what the Scoliosis team is doing is super cool. Especially their partnership with Schneider's Children's Hospital, I like that you can immediately see the results and give them something physical to help them.
* The other teams Natural Language Processing and Image Processing are also very interesting, but I was not as compelled to join with their presentations.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Midterm Presentations | Completed | 10/24/2022 | 10/31/2022 | 10/24/2022 |
| Sub-Team Preference Quiz | Completed | 10/24/2022 | 10/31/2022 | 10/25/2022 |

## 19 October 2022

<details>
  <summary><b>Team Meeting Notes - EMADE Project</b></summary>

### Bootcamp Group 1 Meeting

* In this meeting we normalized our data, fixed our ML algorithms, fixed our MOGP graph, and worked on the EMADE project.

### More Preprocessing of the Data

* Normalizing our data is important, and to do this we divided each column by the mean.
* Due to standardizing our data, our ML algorithms were not Pareto optimal, so we had to change them.

### Changes to ML Algorithms

* ML Algorithms:
  * <code>KNeighborsClassifier()</code>
  * <code>SVC()</code>
  * <code>GaussianNB()</code>
  * <code>GaussianProcessClassifier()</code>
  * <code>RandomForestClassifier()</code>
* The <code>RandomForestClassifier()</code> is RANDOM, and the <code>GaussianNB()</code> varies because we activated the shuffle parameter in the <code>KFolds()</code> which can changed based on each time we run the code. However, multiple times the average FPR (false positive rates) and FNR (false negative rates) comes out to be Pareto optimal.

### Changes to the MOGP Evolutionary Algorithm and Graph

* We added in more primitives to the set. We added a locally defined <code>pass_(a)</code> function.

  ```
  pset = gp.PrimitiveSetTyped("main", [float, float, float, float, float, float, float], bool)

  # math ops
  pset.addPrimitive(operator.add, in_types=[float, float], ret_type=float)
  pset.addPrimitive(operator.sub, in_types=[float, float], ret_type=float)
  pset.addPrimitive(operator.mul, in_types=[float, float], ret_type=float)

  # comparators
  pset.addPrimitive(operator.lt, in_types=[float, float], ret_type=bool)
  pset.addPrimitive(operator.le, in_types=[float, float], ret_type=bool)
  pset.addPrimitive(operator.gt, in_types=[float, float], ret_type=bool)
  pset.addPrimitive(operator.ge, in_types=[float, float], ret_type=bool)
  pset.addPrimitive(operator.eq, in_types=[float, float], ret_type=bool)

  # pass
  def pass_(a):
    return a
  pset.addPrimitive(pass_, in_types=[float], ret_type=float)

  pset.renameArguments(ARG0="Pclass", ARG1="Sex", ARG2="Age", ARG3="SibSp", ARG4="Parch", ARG5="Fare", ARG6="Embarked")
  ```
* We also had an overall population that took the best individuals from each fold and made a Pareto front. This was instead of having graphs for each fold because it was hard to make comparisons with the ML algorithms for each fold since they were not Pareto optimal for each fold.
* In addition, we made the points of the graph extend to the limit of the graph [0, 1] due to the area under the curve calculations.
* The MOGP Evolutionary Algorithm was clearly better than the ML Pareto front due to the graph itself as well as the area under the curve.
  * ML's area under the curve was twice as much as the MOGP area under the curve.
* Below are the pictures of the ML, MOGP, and both Pareto fronts together as well as the area under the curves.
  <img src="files/chettrich3/ML Pareto Optimal.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/MOGP Pareto Optimal.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/ML + MOGP Graphs.png" alt="drawing" width="300"/>

### EMADE as a Group

* We had to connect to Daniel's SQL Server using his credentials.
  * Host: Daniel's IP Address
  * Username: His Username
  * Password: His Password
  * Port: 3306
* Also, we had to change the script to change the dataset files to use our preprocessed data when EMADE ran.
* I was off campus this weekend, and even though I was using the VPN I could not access the database via Sequel Pro or MSQL Workbench.
* Coming back on campus, I was able to join as a worker process by running the script we made with <code>python group_1_data_splitter.py</code>, and using this to start EMADE <code>python src/GPFramework/launchGTMOEP.py templates/input_titanic.xml -w</code>. We had to make sure to change <code>input_titanic.xml</code> to Daniel's credential and set the reuse bit to 1 in order to run the process and not change any of the data that was running before.
* At one point we had over 100 generations; however, we did not set the reuse bit so it wiped all of the data. We truly learned the hard way.
* Since some of us have different Python versions, we cannot run as a cluster of computers. I now have to figure out how to change my Python version to 3.7 which is not possible with the M1 Mac arm64 structure.
* We ran for 100 generations overnight as a cluster of computers.
* We found that the number of Pareto individuals increased as the number of generations increased, generation 0 only had 3 Pareto individuals while generation 100 had 150 Pareto individuals.
  * The average would not be helpful as it increases steadily as the number of generations increases.
* We imported <code>pymysql</code> as well as <code>create_engine</code>. Below is the code that we used to analyze the data from EMADE.

  ```
  # Can only connect to DB through colab when this ngrok tunnel to Daniel's laptop is on
  engine = create_engine("mysql+pymysql://ROUSER:roro@6.tcp.ngrok.io:12877/vip")

  with engine.connect() as conn:
  pareto_front = conn.execute("SELECT * from vip.paretofront")

  generations = {i: [] for i in range(0, 101)}
  for row in pareto_front:
    if (row["generation"] <= 100):
      generations[row["generation"]].append(row["hash"])
    
  for gen, hashes in generations.items():
    if gen == 100:
      fpr_list = []
      fnr_list = []

      for hash in hashes:
        ind = conn.execute(f"SELECT * FROM vip.individuals WHERE hash='{hash}'").fetchone()
        fpr = ind["FullDataSet False Positives"] / TN
        fnr = ind["FullDataSet False Negatives"] / TP

        fpr_list.append(fpr)
        fnr_list.append(fnr)

      fpr_list, fnr_list = (list(l) for l in zip(*sorted(zip(fpr_list, fnr_list))))

      # Fix pareto front
      new_fpr, new_fnr = [], []
      min_fnr = 10000
      for i in range(len(fpr_list)):
        min_fnr = min(min_fnr, fnr_list[i])
        if fnr_list[i] <= min_fnr:
          new_fpr.append(fpr_list[i])
          new_fnr.append(fnr_list[i])

      new_fpr = [new_fpr[0]] + new_fpr + [1]
      new_fnr = [1] + new_fnr + [new_fnr[len(new_fnr) - 1]]
      
      # Calculate EMADE AUC
      f1 = np.array(new_fpr)
      f2 = np.array(new_fnr)
      EMADE_AUC = f"EMADE Area Under Curve: {(np.sum(np.abs(np.diff(f1))*f2[:-1]))}"
  ```

* Below is the Pareto front for EMADE as well as the combined graph of the ML, MOGP, and EMADE Pareto frontiers.
  <img src="files/chettrich3/EMADE Pareto Frontier.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/ML + MOGP + EMADE Pareto Frontier.png" alt="drawing" width="300"/>

[Bootcamp Team 1 - Midterm Presentation Slides](https://docs.google.com/presentation/d/1YB6Ebc-ghr86Fm3_Evw8EsbBS3zByQGmwH0rwu7C9LE/edit?usp=sharing)

[Bootcamp Team 1 - Titanic Colab Notebook](https://colab.research.google.com/drive/1xLTwAyai275zJ02qhR5QDsvrAevK_ea7?usp=sharing)

</details>

<details>
  <summary><b>EMADE - M1 Mac Issues and Solutions</b></summary>

### Worker Process Issues

* In order to fix the issue with the <code>zip</code> file not being able to be opened dealt with the deap and setup tools versions.
* However, if you are on Python version 3.10.0 or greater and change the deap version to be 1.2.2 and the setuptools version to be 57, you will receive this error.
  * <code>Import Error: cannot import name 'Sequence' from 'collections'</code>
  * The <code>Sequence</code> module used to be imported straight from <code>collections</code>, but it was changed to <code>collections.abc</code> with an update to Python 3.10.0 from 3.9.0.
  * The quick fix to this problem is to change your Python version to 3.9.0 which you can do on an M1 Mac. The longer fix would be to do <code>pip install requests --upgrade</code> where requests would be changed out with the package you are trying to install. However, this would not work in this case because we do not want to update the deap version to 1.3.3 or setuptools to version 65.
* Once I changed the Python version to be 3.9.0 using <code>conda install python=3.9</code>, I made sure all of the packages were installed by going through the README instructions again, ran <code>bash reinstall.sh</code>, and it worked.
* Make sure to only change your Python version once and to ensure everything else is installed properly, no warnings should be generated when running <code>bash reinstall.sh</code>. Previously, I was getting warnings when I ran the script, and even though my master and worker process ran there were some underlying issues. This would cause the worker process to stall out with some packages when running. My M1 Mac would also not terminate processes if they took more than 30% of memory or over 9000 seconds, so there were underlying issues. A quick fix to this was to delete the row out of the database; however, you could be deleting Pareto optimal individuals, so this is not a valid fix. The longer fix and how I got my master and worker process running to generation 5 in 5 minutes was to delete EVERYTHING. More is shown in the next section "The Longer Fix".

### The Longer Fix

* I uninstalled every package, using <code>conda uninstall package_name</code> or <code>pip uninstall package_name</code>. I created a new environment called mlp, using <code>conda install --name myenv</code>, and I activated this environment with <code>conda activate mlp</code>. The website below goes into more details about how to manage environments.
  * [Managing Environments in Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)
* From this point I went through the README instructions. I had a lot of trouble with the tensorflow version. The second website below really walks through step by step. In addition to installing tensorflow this way, you also need to do <code>conda install tensorflow</code> as shown in the later README instructions. I had to play around a bit to get it to work, but ultimately it does if you have patience.
  * [Apple Website for Tensorflow](https://developer.apple.com/metal/tensorflow-plugin/)
  * [Helpful Website to Install Tensorflow](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)
* After resolving the tensorflow issues, I ran into the MyPool issue. This is fixed by updating the Python version with these two commits.
  * [Changing Python Version Part 1](https://github.gatech.edu/emade/emade/commit/35e53cd957c05f140f7df8b991cb3b8d80a99a10)
  * [Changing Python Version Part 2](https://github.gatech.edu/emade/emade/commit/eb5b009d7997016c8a4178b8d143ea2c633fb680)
* Then I ran into the deap version issue and the setuptools version issue. I fixed this by uninstalling both and reinstalling in the appropriate versions mentioned in the "Worker Process Issues" section.
* Lastly, I ran into the issue with the Python version and having to change to 3.9.0 in order for the master and worker process to start. The website helped me make the decision to change the Python version, and provides more information on the issue as does the section above.
  * [ImportError](https://bobbyhadz.com/blog/python-importerror-cannot-import-name-sequence-from-collections)
* Make sure to run the <code>bash reinstall.sh</code> and make sure there are no warnings before starting the EMADE process.
* Ultimately getting EMADE to work took A LOT of time, but in the end it is worth it, do not lose patience or give up!

### The Future

* Depending on what Sub-Team I end up on, when we use EMADE I will wait to see what version of Python they are using.
* If I need to switch my M1 Mac to Python 3.7 I will do that. I have found some resources online to help. Till then, I do not want to alter what works locally.
  * [Python to 3.7 on M1 Mac](https://diewland.medium.com/how-to-install-python-3-7-on-macbook-m1-87c5b0fcb3b5)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - EMADE Project | Completed | 10/19/2022 | 10/24/2022 | 10/24/2022 |
| EMADE - M1 Mac Issues | Completed | 10/19/2022 | 10/24/2022 | 10/23/2022 |

## 12 October 2022

<details>
  <summary><b>Team Meeting Notes - EMADE Set-Up</b></summary>

### Problems Setting Up EMADE

[EMADE Set-Up README](https://github.gatech.edu/emade/emade)

* I have had a lot of correspondence with the Professor regarding installation and setting up EMADE on my M1 Mac.
* Having an M1 Mac has created a lot of issues for installation, but once I got access to the M1 Installation Guide it made my life a lot easier.
  * [M1 Installation Guide](https://github.gatech.edu/emade/emade/wiki/M1-Installation-Guide)
* Homebrew has proved to be very helpful with downloading MariaDB, but it is a different configuration.
  * Installing miniforge and MariaDB through homebrew is useful.
  * There are websites specifically on commands through MariaDB.
  * [Installing Maria DB](https://mariadb.com/kb/en/installing-mariadb-on-macos-using-homebrew/)
* After resolving these issues, I needed a program in order to work with the database, so I downloaded Sequel Pro.
  * [Sequel Pro](https://sequelpro.com/)
* I ended up having to create a database titled "titanic" as well as 127.0.0.1 as the Host. The Username and Password I created by making a user title "vipuser1" and a password to go along with it. I had to grant all privileges to "vipuser1" through the admin user MariaDB creates.
* As mentioned in class, I had to change the version of Python to 3.8 or greater specifically for M1 Macs using the committed changes, this is also known as the MyPool problem.
  * [Changing Python Version Part 1](https://github.gatech.edu/emade/emade/commit/35e53cd957c05f140f7df8b991cb3b8d80a99a10)
  * [Changing Python Version Part 2](https://github.gatech.edu/emade/emade/commit/eb5b009d7997016c8a4178b8d143ea2c633fb680)
* You must remember to run the command <code>bash reinstall.sh</code> in the terminal if the file changed is an imported file.
* However, after all of these changes, I am now getting errors in the Worker Processes files, and will have to work on fixing this issue during class this week.
  * The problem is related to [Link to DEAP Issues](https://github.com/DEAP/deap/issues/671). 
  * The Professor made a script and ran it through <code>2to3.py</code> and fixed the line. 
  * The recommendation is to uninstall deap and setuptools, verifying they are gone, and then <code>pip install setuptools==57</code>, and <code>pip install deap==1.2.2</code>.
    * I did find that pip install setuptools should be == 57.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Team Meeting Notes - Set-Up EMADE: Fix Worker Processes | Completed | 10/12/2022 | 10/19/2022 | 10/23/2022 |

## 5 October 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 6</b></summary>

### Introduction to EMADE

* EMADE is the Evolutionary Multi-Objective Algorithm Design Engine.
* It combines a multi-objective evolutionary search with high-level primitives to automate the process of designing machine learning algorithms.
* To start a run of EMADE, navigate to the top level directory and run 

  ```
  python src/GPFramework/launchGTMOEP.py templates/input_titanic.xml
  ```
### What is in the Input File?

* The input file is an xml document that configures all the moving parts in EMADE.
* EMADE automatically detects cluster management software for gridengine and SLURM.
* Make sure to configure the Database, change the SERVERHOSTNAME to localhost or 127.0.0.1.
* EMADE can run across multiple datasets.
* The data is preprocessed into gzipped csv files.
* It is cross folded 5 times, this will create 5 Monte Carlo trials that algorithms can be scored with.
* Each train and test file create a DataPair object in EMADE.
* Objectives
  * The names are the columns in the database.
  * Weight specifies if it should be minimized (-1.0) or maximized (1.0).
  * Achievable and goal are used for steering the optimization, lower and upper are used for bounding.
  * Evaluation specifies where evaluation functions specified in the objectives section live, and how much memory each worker is allowed to use before marking an individual as "fatal".
* <code>\<workersPerHost\></code> specifies how many evaluations to run in parallel.
  * EMADE is resource intensive, keep this value low on laptops (2-3).

### Connecting a Worker Process to a Peer

* You can use the -w flag along with your peer's server information in the dbconfig in order to allow your computer to act as a worker process for your peer's master process.

  ```
  python src/GPFramework/launchGTMOEP.py templates/input_titanic.xml -w
  ```
* Make sure that the dbconfig in the input file specifies their IP address and not localhost.

### Understanding EMADE Output

* The best place to look at the outputs of EMADE are in the MySQL databases.
* Connect to a mysql server from the command line:

  ```
  mysql -h hostname -u username -p
  ```

* You can select the database, and use queries to select certain individuals.

### EMADE Structure

* src/GPFramework is the main body of code
  * gtMOEP.py is the main EMADE engine, most of the evolutionary loop is here, including the evaluation method.
  * gp_framework_helper.py is where the primitive set is built for EMADE, this function points to where the primitives live.
  * data.py provisions the DataPair object that is passed from primitive to primitive.
* datasets/ is where some test datasets live.
* templates/ is where the input files live.

</details>

<details>
  <summary><b>EMADE Assignment Overview</b></summary>

### EMADE Assignment

* Run EMADE as a group.
  * 1 person is the master
  * The rest should connect their workers
* Run for a substantial number of generations.
* Learn some SQL.
* Make a plot of your non-dominated frontier at the end of the run, compare with ML and MOGP assignments.
* Make any other plots and figures to show your analysis of EMADE running on the Titanic problem, make successful trees.
* Present findings on Monday, October 25th.
* My team has decided to start setting up the EMADE master and worker processes this Wednesday, October 12th during our Workday.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Download EMADE - Master Process Fix | Completed | 10/05/2022 | 10/12/2022 | 10/09/2022 |
| Lecture 6 Review - "Introduction to EMADE" | Completed | 10/05/2022 | 10/12/2022 | 10/09/2022 |
| EMADE Project Overview | Completed | 10/05/2022 | 10/12/2022 | 10/09/2022 |

## 28 September 2022

<details>
  <summary><b>Team Meeting Notes - Presentations ML/GP</b></summary>

### Presentation Notes

* My group presented first, as my group was Group 1.
* **Future Considerations:**
  * Do more pre-processing of the data, leaving in the Name column and grouping people into classes based on who is most likely to survive.
  * Test more ML models and choose the ones that create a Pareto front instead of using brute force.
  * Improve the Evolutionary Algorithm by understanding what it does.
  * Average all the folds for ML and for MOGP and put them on the same graph to compare them.
  * Do the minimization graph to see which model is the best.
  * Normalize the data!
  * Compare correlations of what actually makes a difference in determining who survives.
  * Make sure to add a group of individuals rather than a single individual.
  * Possibly plot out the tree.

* If the ML model does better than the MOGP model, probably something is wrong with the Evolutionary Algorithm.
* Make sure to understand Single Objective Genetic Programming vs. Multi-Objective Genetic Programming.
* Ultimately, I need to learn more about cross validation and what is included in DEAP and scikit-learn to be able to use the packages in improving the Titanic Project and future projects.

</details>

<details>
  <summary><b>Machine Learning Notes</b></summary>

### What is Machine Learning?

* Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
* Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects.

### How Machine Learning Works

* A Decision Process
  * ML algorithms are used to make a prediction or classification. Based on input data, which can be labeled or unlabeled, the algorithm will produce an estimate about a pattern in the data.
* An Error Function
  * An error function evaluates the prediction of the model. If there are known examples, an error function can make a comparison to assess the accuracy of the model.
* A Model Optimization Process
  * If the model can fit better to the data points in the training set, then weights are adjusted to reduce the discrepancy between the known example and the model estimate.
  * The algorithm will repeat the evaluate and optimize process, updating weights autonomously until a threshold of accuracy has been met.

### Supervised Machine Learning

* Supervised learning, is defined by its use of labeled datasets to train algorithms to classify data or predict outcomes accurately. 
* As input data is fed into the model, the model adjusts its weights until it has been fitted appropriately. This occurs as part of the cross validation process to ensure that the model avoids overfitting or under-fitting.
* Helps organizations solve a variety of real-world problems at scale.
* Some Methods: neural networks, linear regression, logistic regression, random forest, and support vector machine (SVM)

### Unsupervised Machine Learning

* Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets.
* These algorithms discover hidden patterns or data groupings without the ned for human intervention.
* Used to reduce the number of features in a model through the process of dimensionality reduction.
* Principal component analysis (PCA) and singular value decomposition (SVD) are two common approaches.

[IBM - Machine Learning](https://www.ibm.com/cloud/learn/machine-learning)

### DEAP

* DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas.
* DEAP includes genetic algorithms using any imaginable representation and genetic programming using prefix trees.

[DEAP Documentation](https://github.com/DEAP/deap)

### Genetic Programming

* Genetic programming is a technique to create algorithms that can program themselves by simulating biological breeding and Darwinian evolution.
* Basic approach is to let the machine automatically test various simple evolutionary algorithms and breed the most successful programs in new generations.

[Genetic Programming](https://deepai.org/machine-learning-glossary-and-terms/genetic-programming)

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Download EMADE - Begin | Completed | 09/28/2022 | 10/05/2022 | 10/02/2022 |
| Team Meeting Notes - Presentations ML/GP | Completed | 09/28/2022 | 10/05/2022 | 09/29/2022 |
| Learn About DEAP and ML | Completed | 09/28/2022 | 10/05/2022 | 10/02/2022 |

## 21 September 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 5</b></summary>

### Overview of Titanic MOGP

* **Strongly Typed Genetic Programming**
  * Putting more constriction on the structure of the trees.
  * How primitive set of strongly typed works in DEAP?
  * Must have enough primitives to connect all of the dots.
  * DEAP decides how deep the tree is, and then fills them in.
  * The area under the curve, Riemann sum, is a good way of scoring the algorithms.
    * Multi-objective
    * Measure time and generations decreases the area under the curve.
  * Should put a constant in the primitive set.
    * Helps to pre-compute some columns.
    * Means of a column
      * Example: are you above the average age on the boat?
  * Hall of Fame that keeps track of things over time.

* **Presentations**
  * Pareto front from ML and MOGP on the same plot.
  * Tell which outperformed which, in other words which line is closer to the origin on every level.
  * Tell about pre-processing steps.
  * How machine algorithms were co-dominant?
  * Compare the results from ML and MOGP.
  * Tell anything you discovered along the way.
  * Presentation should be about 7 minutes long.
  * Post on the team page in the wiki.

[Bootcamp Team 1 - Titanic Presentation](https://docs.google.com/presentation/d/1YB6Ebc-ghr86Fm3_Evw8EsbBS3zByQGmwH0rwu7C9LE/edit?usp=sharing)

[Bootcamp Team 1 - Colab Notebook](https://colab.research.google.com/drive/1xLTwAyai275zJ02qhR5QDsvrAevK_ea7?authuser=1)

### Team Meeting 2 for Titanic Project

* We shared which machine learning models we used and found out if they were co-dominant.
  * Charlotte: <code>Perceptron(max_iter=5)</code>
  * Daniel: <code>SVC()</code>
  * Pranav: <code>LogisticRegression()</code>
  * Aditya: <code>GaussianProcessClassifier()</code>
  * Dhruv: <code>GaussianNB()</code>
* Together with the Average False Positive Rates (FPR) and Average False Negative Rates (FNR) over each fold averaged separately for each model we were able to create a Pareto Front.
* We had to change models because the rates would not be able to create a Pareto Front. For example, previously I used Random Forest, but I had to change to <code>Perceptron()</code> due to the FPR and the FNR.
* We then discussed the MOGP evolutionary algorithm and what we wanted to complete. Also, we started creating the presentation and putting our information inside. More information on the evolutionary algorithm creation is shown in the Titanic MOGP Lab Notes Tab.

</details>

<details>
  <summary><b>Titanic MOGP Lab Notes</b></summary>

### Titanic Lab - Fixing Machine Learning

* I had to change my machine learning model to <code>Perceptron()</code> and average the rates over the five splits. 
* When I originally used <code>RandomForestClassifier()</code> I did not test it over the folds
* Within the for loop for the <code>KFold()</code> method, I accumulated the individual false positives, false negatives, true positives, and true negatives to average them outside of the for loop.

  ```
  kf = KFold(n_splits=5, shuffle=True, random_state=10)
  model = Perceptron(max_iter=5)

  avg_fpr = 0
  avg_fnr = 0

  for train_index, test_index in kf.split(X_train_split):
      X_train, X_test = X_train_split.iloc[train_index,:], X_train_split.iloc[test_index,:]
      y_train, y_test = y_train_split.iloc[train_index], y_train_split.iloc[test_index]

      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
      fpr = fp / (fp + tn)
      fnr = fn / (fn + tp)
    
      avg_fpr += fpr / 5
      avg_fnr += fnr / 5
    
      print(f"FPR: {fpr}", f"FNR: {fnr}")
    
  print(f"Average FPR: {avg_fpr}", f"Average FNR: {avg_fnr}")
  ```
* With the <code>Perceptron()</code> algorithm, we were able to generate a proper Pareto front in the picture below for our ML models.

  <img src="files/chettrich3/ML Pareto Front.png" alt="drawing" width="300"/>

### Titanic Lab MOGP Notes

* We started by creating the Population with DEAP and plotted the objective space.
* Then we went into create the algorithm, we used previous code from Lab 1 and 2 to help us create it. 
* Also, we used the algorithms.py file under the DEAP documentation to help us go in the right direction. We tried to create an algorithm to mimic <code>varOr()</code> algorithm.
* Of course we had a different function to evaluate the entire population, a for loop for beginning the evolution, a crossover portion, a mutation portion, and finally an evaluate offspring portion. These are all different based on our values and the way we calculated each value, and the way we evaluated our offspring and population.

  ```
  # Evaluate entire population
  fitnesses = [toolbox.evaluate(ind, X_test, y_test) for ind in pop]
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

  # Begin evolution
  for generation in range(NGEN):
    # Select the next generation 
    offspring = toolbox.select(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in pop]

    # Apply crossover on offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    # Apply mutation on offspring
    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # Evaluate offpsring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = [toolbox.evaluate(ind, X_test, y_test) for ind in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(offspring)

    pop[:] = offspring
  ```

* In the end, we generated Pareto fronts for each fold. We added the ML Pareto plot for each different algorithm we chose onto the MOGP plot to show the differences.
* The Pareto front was not optimal in each fold, only in fold 2 and the average false positive and false negative rates.
* The pictures are shown below for each fold:

  <img src="files/chettrich3/Pareto Front Fold 1.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/Pareto Front Fold 2.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/Pareto Front Fold 3.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/Pareto Front Fold 4.png" alt="drawing" width="300"/>
  <img src="files/chettrich3/Pareto Front Fold 5.png" alt="drawing" width="300"/>

* Overall, the MOGP seemed to be the better algorithm compared to the ML. Except when we get into the better ML algorithms like Pranav's <code>LogisticRegression()</code> and Aditya's <code>GaussianProcessClassifier()</code>.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Add Bootcamp Team Repository in GitHub | Completed | 09/21/2022 | 09/28/2022 | 09/28/2022 |
| Team Meeting Notes - Lecture 5 and Titanic Group | Completed | 09/21/2022 | 09/28/2022 | 09/25/2022 |
| Titanic MOGP | Completed | 09/21/2022 | 09/28/2022 | 09/25/2022 |

## 14 September 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 4</b></summary>

### Overview of the Titanic Project

Dataset: https://www.kaggle.com/competitions/titanic/data?select=test.csv

* Goal 1: find out whether or not a passenger survived the Titanic through machine learning techniques.
* Goal 2: find ways to digest and pre-process data.
  * Pre-Process Data: remove the noise.
* Come up with different algorithms to take each row to see whether or not they survived.
* Create a Pareto Frontier of Algorithms
  * All solutions must be co-dominant.
  * Must have 4-5 co-dominant solutions.
* Make a model using the train.csv dataset and use test.csv dataset to test the model.
* train.csv Data
  * Contains passenger data along with whether or not they survived.
  * **Cross Folding**
    * Fit on one portion of the data, and predict on the other portion of the data.
    * Score on the part the model did not fit on.
    * Rotate around.
* test.csv Data
  * Contains passenger data, but not whether or not they survived.
* gender_submission.csv Data
  * This is a sample submission that only contains the passenger_id, survived.

### Team Meeting 1 for Titanic Project

* We decided to use Google Colab to work together on the project.
* **Pre-Processing the Data**
  * We pre-processed the data by dropping the Name, Ticket, and Cabin columns and setting the index to be the PassengerID column on both the train.csv dataset and the test.csv dataset.
  * We then replaced the NaN values in several columns:
    * Age column replaced with the mean Age
    * Fare column replaced with the mean Fare
    * Embarked column replaced with the mode value from Embarked
  * We changed the strings to integers in the Embarked and Sex columns to make it easier to work with.

  ```
  train_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True) # drop Name, Ticket, Cabin columns
  train_data.set_index(keys=['PassengerId'], drop=True, inplace=True) # replace index with PassengerId

  test_data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
  test_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

  train_nan_map = {'Age': train_data['Age'].mean(), 'Fare': 
  train_data['Fare'].mean(), 'Embarked': train_data['Embarked'].mode()[0]}
  test_nan_map = {'Age': test_data['Age'].mean(), 'Fare': 
  test_data['Fare'].mean(), 'Embarked': test_data['Embarked'].mode()[0]}

  train_data.fillna(value=train_nan_map, inplace=True)
  test_data.fillna(value=test_nan_map, inplace=True)

  columns_map = {'Embarked': {'C': 0, 'Q': 1, 'S': 2}, 'Sex': {'male': 0, 'female': 1}}
  train_data.replace(columns_map, inplace=True)
  test_data.replace(columns_map, inplace=True)
  ```

* **Folding the Data**
  * We decided to fold the data 5 ways using the <code>KFold</code> method to split the dataset into 5 consecutive folds.
  
  ```
  X_train_split = train_data.loc[:, train_data.columns != 'Survived']
  y_train_split = train_data.loc[:, 'Survived']

  kf = KFold(n_splits=5, shuffle=True, random_state=10)

  for train_index, test_index in kf.split(X_train):
      X_train, X_test = X_train_split.iloc[train_index,:], X_train_split.iloc[test_index,:]
      y_train, y_test = y_train_split.iloc[train_index], y_train_split.iloc[test_index]
  ```

</details>

<details>
  <summary><b>Titanic ML Lab Notes</b></summary>

### Charlotte's Titanic Machine Learning Model

* To start with, I copied all of the code my team made in our first meeting to get the KFolds.
* I decided to test my model using the Random Forest algorithm
* **Random Forest**
  * Random Forest is a supervised learning algorithm. It creates a forest and makes it random like the name.
  * The forest is an ensemble of Decision Trees, trained with the bagging method.
    * The bagging method is that a combination of learning models increases the overall result.
  * The Random Forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.
  * This can be used for both classification and regression problems.
  * The Random-Forest algorithm brings extra randomness into the model, when growing its trees.
  * Instead of searching for the best feature while splitting a node, it searches for the best feature among a random subset of features. This process creates a wide diversity.
* General rule for models, the more features you have, the more likely your model will suffer from overfitting.

  ```
  random_forest = RandomForestClassifier(n_estimators = 100, oob_score = True)
  random_forest.fit(X_train.values, y_train.values)
  y_pred = random_forest.predict(X_test.values)
  y_truth = y_test.values

  random_forest.score(X_train.values, y_train.values)

  acc_random_forest = round(random_forest.score(X_train.values, y_train.values) * 100, 2)
  ```
* Then, I created a confusion matrix using the Random Forest model. I first printed the values.

  <img src="files/chettrich3/Confusion Matrix Values.png" alt="drawing" width="300"/>

* After printing the values, I plotted the confusion matrix.

  <img src="files/chettrich3/Confusion Matrix Plotted.png" alt="drawing" width="300"/>

* From the predictions using the Random Forest algorithm, I created a .csv file with a PassenderId and a Survived column entitled random_forest_predictions.csv.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Add Bootcamp Team Members in GitHub | Completed | 09/14/2022 | 09/21/2022 | 09/17/2022 |
| Team Meeting Notes - Lecture 4 and Titanic Group | Completed | 09/14/2022 | 09/21/2022 | 09/17/2022 |
| Titanic Data Format and ML Models | Completed | 09/14/2022 | 09/21/2022 | 09/18/2022 |

## 7 September 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 3</b></summary>

### Multiple Objectives

* What is an algorithm looking for in a mate?
  * High Fitness
  * Efficiency
  * Level or Error and Accuracy
    * Type I or Type II Error
  * Precision and Recall
  * Better Objective Performance
  * Complexity
    * Want something simpler
* If algorithms are structurally similar, there is a higher possibility in producing a better offspring.

* Gene pool is the set of genome to be evaluated during the current generation.
  * Genome
    * Genotypic description of individuals
      * DNA
      * Set of values
      * Tree structure, string
    * Search Space
      * Set of all possible genomes
  * The Evaluation of a Genome associates a genome/individual with a set of scores.
    * True Positive - TP
      * How often we are identifying the desired object
    * False Positive - FP
      * How often are we identifying something else as the desired object
  * Objectives
    * Set of measurements each genome is scored against
    * Phenotype
  * Objective Space - Set of objectives
  * Evaluation - Maps a Genome
    * From a location in search space
      * Genotypic description
    * To a location in objective space
      * Phenotype description

### Different Measures

* **Classification Measures**
  * Binary Classifiers: for every entry in dataset either something happened or it did not happen
  * Confusion Matrix: Truth Column, Predicted: Positive Column, and Predicted: Negative Column
    * Assesses event by event
    * No right answer on which is the worst error, there is a tradeoff space.

<img src="files/chettrich3/Measures.png" alt="drawing" width="500"/>

* **Maximization Measures**
  * Bigger is Better
  * Sensitivity or True Positive Rate (TPR)
    * AKA hit rate or recall
    * TPR = TP/P = TP/(TP+FN)
  * Specificity (SPC) or True Negative Rate (TNR)
    * TNR = TN/N = TN/(TN+FP)

* **Minimization Measures**
  * Smaller is Better
  * False Negative Rate (FNR)
    * FNR = FN/P = FN/(TP+FN)
    * FNR = 1 - TPR
  * Fallout or False Positive Rate (FPR)
    * FPR = FP/N = TN/(FP+TN)
    * FPR = 1 - TNR = 1 - SPC

* **Other Measures**
  * Precision or Positive Predictive Value (PPV)
    * PPV = TP/(TP+FP)
    * Bigger is Better
  * False Discovery Rate
    * FDR = FP/(TP+FP)
    * FDR = 1 - PPV
    * Smaller is Better
  * Negative Predictive Value (NPV)
    * NPV = TN/(TN+FN)
    * Bigger is Better
  * Accuracy (ACC)
    * ACC = (TP+TN)/(P+N)
    * ACC = (TP+TN)/(TP+FP+FN+TN)
    * Bigger is Better

### Objective Space

* Each individual is evaluated using objective functions.
  * Mean Squared Error
  * Cost
  * Complexity
  * True Positive Rate
  * False Positive Rate
* Objective scores give each individual a point in objective space.
  * Referred to as the phenotype of the individual.

### Pareto Optimality

* An individual is Pareto optimal if there is no other individual in the population that outperforms the individual on all objectives.
* The set of all Pareto individuals is known as the Pareto frontier.
* These individuals represent unique contributions.
* We drive selection by favoring Pareto individuals.
  * We maintain diversity by giving all individuals some probability of mating.

### Nondominated Sorting Genetic Algorithm II (NSGA II)

* Population is separated into nondomination ranks.
* Individuals are selected using a binary tournament.
* Lower Pareto ranks beat higher Pareto ranks.
* Ties on the same front are broken by crowding distance.
  * Summation of normalized Euclidian distances to all points within the front.
  * Higher crowding distance wins.

### Strength Pareto Evolutionary Algorithm 2 (SPEA2)

* Each individual is given a strength _S_.
  * _S_ is how many others in the population it dominates.
* Each individual receives a rank _R_.
  * _R_ is the sum of _S_'s of the individuals that dominate it.
  * Pareto individuals are nondominated and receive an _R_ of 0.

</details>

<details>
  <summary><b>Lab 2 - Multiple Objective Optimization</b></summary>

### Multiple Objective Optimization

* Goal: Minimize two objectives, mean squared error and the size of the tree, and have an individual as close to the bottom right of the graph as possible.
* There are few extra functions like the Pareto dominance function that shows a representation of the objective space. 
* Individuals dominating the Pareto front are closer to the bottom right because they have the lowest MSE and tree size.
* DEAP's Pareto front hall of fame gives us a list of the nondominated individuals after our evolution. This list and the last population creates a visualization of the Pareto front.
* Used DEAP's <code>eaMuPlusLambda</code> function for multi-variable genetic programming using tournament selection.
  * This adds two new parameters, mu and lambda.
    * Mu is the number of individuals to select for the next generation.
    * Lambda is the number of children to produce at each generation.
* To measure the performance of the Pareto front, we use the area under the curve of the Pareto front. The lower area under the curve or AUC is, the better the Pareto front is.

  <img src="files/chettrich3/Lab 2 - Pareto.png" alt="drawing" width="300"/>

### Assignment: Decrease 25% in the AUC

* I decreased the AUC from 2.46 to 1.84 by;
  * I changed the mutation algorithm to <code>mutInsert</code> rather than <code>mutUniform</code>.
    ```
    toolbox.register("mutate", gp.mutInsert, pset=pset)
    ```
  * Also, I changed the evolutionary algorithm from <code>eaMuPlusLambda</code> to <code>eaMuCommaLamba</code>.
    ```
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    ```
  * I switched the selection algorithm to <code>gp.NGSA2</code>, known as the Nondominated Sorting Genetic Algorithm II we learned in Lecture 3.
    ```
    toolbox.register("select", tools.selNSGA2)
    ```
  * Lastly, I added the primitives <code>power2</code>, and <code>power4</code>. I originally added the primitive <code>np.exp</code>, but it did not optimize the AUC like the other primitives do.
    ```
    def power2(x):
      return np.power(x, 2)

    def power4(x):
      return np.power(x, 4)
    ```

    <img src="files/chettrich3/Lab 2 - Assignment.png" alt="drawing" width="300"/>

</details>

<details>
  <summary><b>Self Grading Assessment</b></summary>

### Notebook Grading

| Task | Score | Comments |
| ---- | --------------- | ---------------|
| _**Notebook Maintenance**_ | | |
| Name & Contact Info | 5/5 | | 
| Teammate Names and Contact Info Easy to Find | 0/5 | I do not have teammates yet. |
| Organization | 5/5 | |
| Updated at Least Weekly | 10/10 | |
| _**Meeting Notes**_ | | |
| Main Meeting Notes | 5/5 | |
| Sub-teams' Efforts | 5/10 | I do not have a team, but my personal contributions are documented.| 
| _**Personal Work and Accomplishments**_ | | |
| To-Do Items: Clarity, Easy to Find | 5/5 | |
| To-Do List Consistency (Weekly or More) | 10/10 | |
| To-Dos & Cancellations Checked & Dated | 5/5 | | 
| Level of Detail: Personal Work & Accomplishments | 12/15 | Explanations and justifications are present, but my reflections could be improved. |
| _**Useful Resource**_ | | |
| References (internal, external) | 10/10 | Bootcamp Repository has all of my code files uploaded. |
| Useful Resource for the Team | 15/15 | I supply my code, and I supply explanations for my code in the corresponding Notebook section. |

* **Total out of 100:** 87

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Learn How to Create a Repository in GitHub | Completed | 09/07/2022 | 09/14/2022 | 09/09/2022 |
| Team Meeting Notes - Lecture 3 Review: "Multiple Objectives" | Completed | 09/07/2022 | 09/14/2022 | 09/10/2022 |
| Lab 2 - Multiple Objective Optimization  | Completed | 09/07/2022 | 09/14/2022 | 09/10/2022 |
| Self Grading Assessment | Completed | 09/07/2022 | 09/14/2022 | 09/10/2022 |

## 31 August 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 2</b></summary>

### Genetic Programming

* Instead of taking an individual and having a function evaluator to obtain objective scores → the individual is the function itself.
* Creating the function itself:
  * The individual consumes some inputs and gives out some outputs.
  * The evaluator asses how good it is to fit the target.

* **Tree Representation**
  * We can represent a program as a tree structure.
    * **Nodes** are called **primitives** and represent functions.
    * **Leaves** are called **terminals** and represent parameters.
      * The input can be thought of as a particular type of terminal.
      * The output is produced at the root of the tree.
  * How is the Tree Stored?
    * The tree is converted to a **lisp preordered parse tree**.
      * The operator is followed by inputs.
    * Order matters
  * What’s the parse tree?
    * As you build the preorders tree, it dictates the function.

* **Crossover in Genetic Programming**
  * Crossover in tree-based genetic programming is simply exchanging subtrees.
  * Start by randomly picking a point in each tree.
  * These points and everything below create subtrees.
  * The subtrees are exchanged to produce children.

* **Mutation in Genetic Programming**
  * Mutation can involve
    * Inserting a node or subtree
    * Deleting a node or subtree
    * Changing a node
  * Any permutation of the above changes can alter the program

</details>

<details>
  <summary><b>Lab 2 - Genetic Programming</b></summary>

### Symbolic Regression

* **Goal:** to minimize the mean squared error, since we are trying to optimize the function.
* Individual class inherits from DEAP's PrimitiveTree class instead of a list like in the One Max Problem and the N Queens Problem.
* Individuals are represented as a tree structure.
* Trees are the most common data structure used in genetic programming. Trees are made of functions and variables called primitives. Each primitive is a node in the tree where the leaves of a node are the inputs to the parent node. Evaluating an individual means compiling the primitive tree from its leaves to its root node.
* Arity specifies the amount of arguments each primitive takes.
* To evaluate the primitive tree, the tree is compiled into a function. Then the function is calculated and the mean squared error is determined between the function compiled and the actual function generated. 
* Genetic programming is finding the best combination of primitives given objectives to minimize or maximize.

* The primitives already added were <code>np.add</code>, <code>np.subtract</code>, <code>np.multiply</code>, and <code>np.negative</code>.
* Using the primitives already added, generated the best individual possible <code>add(add(add(negative(x), add(multiply(x, x), x)), x), multiply(x, add(multiply(x, x), multiply(x, add(subtract(x, x), multiply(x, x))))))</code> in 25 generations.

  <img src="files/chettrich3/Lab 2 - First.png" alt="drawing" width="300"/>

* Next, I added two of my own primitive functions.

  ```
  pset.addPrimitive(np.sin, arity=1)
  pset.addPrimitive(np.exp, arity=1)
  ```
* After using <code>np.sin</code> and <code>np.exp</code>, another best individual, <code>multiply(add(sin(add(x, sin(add(x, sin(sin(add(x, sin(add(x, x))))))))), exp(sin(add(sin(add(sin(add(sin(add(x, x)), x)), multiply(x, x))), sin(add(x, add(sin(x), x))))))), multiply(x, x))</code> was generated after 32 generations.

  <img src="files/chettrich3/Lab 2 - More Primitives.png" alt="drawing" width="300"/>

* Finally, I experimented with changing the mutation function. Instead of using <code>mutUniform</code> with <code>genFull</code> replacement, which replaces a subtree at a random point with a subtree generated by <code>genFull</code>, I used <code>mutInsert</code>. The mutation function I used inserts a random branch at some random point in the tree. Using this mutation function along with <code>np.sin</code> and <code>np.exp</code> an best individual, <code>add(add(multiply(sin(multiply(x, x)), exp(x)), x), multiply(x, multiply(sin(sin(multiply(multiply(sin(multiply(x, x)), x), x))), x)))</code>, was generated after 33 generations.

  <img src="files/chettrich3/Lab 2 - Mutation.png" alt="drawing" width="300"/>

* In conclusion, by using the original primitives: <code>np.add</code>, <code>np.subtract</code>, <code>np.multiply</code>, and <code>np.negative</code>, and by using <code>mutUniform</code> rather than <code>mutInsert</code> the best individual was generated in the least amount of generations.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Learn How to Add a Picture in GitHub | Completed | 08/31/2022 | 09/07/2022 | 09/02/2022  |
| Team Meeting Notes - Lecture 2 Review: "Genetic Programming" | Completed | 08/31/2022 | 09/07/2022 | 09/02/2022 |
| Lab 2 - Genetic Programming | Completed | 08/31/2022 | 09/07/2022 | 09/05/2022 |


## 24 August 2022

<details>
  <summary><b>Team Meeting Notes - Lecture 1</b></summary>

### Genetic Algorithms

* Each new generation is created through mating/mutation of individuals in the previous population. This is a process that eventually produces the best individual.
* By best meaning the one whose fitness is better than everyone else's in the population and cannot get better.
* Genetic algorithms are bio-inspired

  <img src="files/chettrich3/Genetic Algorithm Flow Chart.png" alt="drawing" width="300"/>

### Definitions

* **Individual:** one specific candidate in the population
* **Population:** group of individuals whose properties will be altered
* **Objective:** a value used to characterize individuals you are trying to maximize or minimize
  * Ex: individual score
* **Fitness:** relative comparison to other individuals
  * Ex: score on curve
* **Evaluation:** a function that computes the objective of an individual
* **Selection:** represents ‘survival of the fittest’; gives preference to better individuals
* **Fitness Proportionate:** the greater the fitness value, the higher the probability of being selected for mating
* **Tournament:** several tournaments among individuals; winners are selected for mating
* **Mate/Crossover:** represents mating between individuals
  * There can be n-point crossovers
* **Mutate:** introduces random modifications; purpose is to maintain diversity
  * Any small change to a gene without combination of two parents would be a mutation
* **Algorithms:** various evolutionary algorithms to create a solution or best individual

</details>

<details>
  <summary><b>Lab 1 - Genetic Algorithms with DEAP</b></summary>

### Overall Notes

* The DEAP module is used for rapid prototyping and testing of ideas. Specifically, it seeks to make algorithms explicit and data structures transparent.

### One Max Problem

* **Goal:** to mutate and evolve a list of all ones [1,1,...,1] from a random starting population
* The tuple, (1.0,) was defined which represents a single objective we want to minimize.
* An "Individual" class was defined as a list with a fitness defined as the tuple objective.
* The string of individuals are a list of Booleans represented as 1s and 0s.
* Toolbox Functions
  * Evaluation
    * This function returns the sum of the Boolean integers of an individual.
    * Individuals with more 1s will receive a higher fitness score with the maximum being 100. 
    * The sum is returned as a tuple to match the fitness objective.
  * Mate
    * Defined as a two-point crossover function.
  * Mutate
    * Defined as flipping a bit in the bitstring to either 1 or 0 with an independent probability of flipping each individual bit of 5%.
  * Select
    * Defined as a tournament selection of 3 individuals. Tournament is a common genetic operator where individuals are sampled and competed against each other.
* With a population size of 300 and 40 generations:

  ```
  -- Generation 39 --
    Min 88.0
    Max 98.0
    Avg 96.48
    Std 2.0839705692099755
  -- End of (successful) evolution --
  ```

### The N Queens Problem

* **Goal:** to mutate and evolve permutations of queens on a size N chessboard to avoid one queen being able to take another
* We want to minimize the number of conflicts between two queens on the chessboard.
* The individual is the list of integers sampled from range(n) without replacement.
* The evaluation counts the number of conflicts along the diagonals.
* Partially matched crossover represents swapping around pairs of queen positions between two parent individuals. This is more effective than swapping individuals around like in a one or two point crossover.
* We will shuffle indexes for the mutation for this problem. We cannot mutate any of the values to be duplicate or outside of the set bounds.
* Another Mutation Function:
  ```
  def mutSwapIndexes(individual):
      '''
      Swaps a single random index in the individual with another random index of different size.
      :param individual: Individual to be mutated.
      :returns: a tuple of one individual.
      '''
      size = len(individual)
      index_1 = random.randint(0, size - 2)
      index_2 = random.randint(0, size - 4)
      if index_1 == index_2:
          index_2 += 1
          individual[index_1], individual[index_2] = individual[index_2], individual[index_1]
      return individual,
  ```

* Even after 100 generations the algorithm is not guaranteed to have a global minimum of 0.
* My mutation function, mutSwapIndexes, output had a global minimum of 3.0 shown below. When I ran it other times it typically has a global minimum greater than 0. This means that it is worse than the mutShuffleIndexes function but not exponentially worse.
  ```
  mutShuffleIndexes
  -- Generation 99 --
    Min 0.0
    Max 12.0
    Avg 0.9366666666666666
    Std 2.259643531375887
  -- End of (successful) evolution --
  Best individual is [4, 17, 10, 14, 9, 5, 2, 13, 18, 7, 1, 3, 11, 6, 16, 0, 19, 12, 15, 8], (0.0,)
  mutSwapIndexes
  -- Generation 99 --
    Min 3.0
    Max 7.0
    Avg 3.05
    Std 0.3476108935769064
  -- End of (successful) evolution --
  Best individual is [5, 7, 16, 15, 8, 12, 14, 18, 2, 4, 1, 3, 9, 13, 19, 11, 0, 10, 6, 17], (3.0,)
  ```
* With the provided settings, the evolutionary loop is able to reach a stable bend around the 20-30 generations mark into the algorithm.
* Altering the tournament size from 3 to 5 or 6 can help the algorithm reach a stable bend around 17 generations into the algorithm.

</details>

### Action Items

| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| ---- | --------------- | ---------------| ------------| -----------| 
| Set up Notebook and Learn GitHub | Completed | 08/24/2022 | 08/31/2022 | 08/28/2022  |
| Team Meeting Notes - Lecture 1 Review: "Genetic Algorithms" | Completed | 08/24/2022 | 08/31/2022 | 08/28/2022 |
| Lab 1 - Genetic Algorithms with DEAP | Completed | 08/24/2022 | 08/31/2022 | 08/28/2022 |

# Resources

| Sub-team Members - Bootcamp Team 1 |
| -------------------------- |
| [Aditya Chandaliya](https://github.gatech.edu/emade/emade/wiki/Notebook-Aditya-Neeraj-Chandaliya) |
| [Pranav Malireddy](https://github.gatech.edu/emade/emade/wiki/Notebook-Pranav-Malireddy) |
| [Dhruv Sharma](https://github.gatech.edu/emade/emade/wiki/Notebook-Dhruv-Sharma) |
| [Daniel You](https://github.gatech.edu/emade/emade/wiki/Notebook-Daniel-L-You) |

[**Personal Bootcamp Repository**](https://github.gatech.edu/chettrich3/chettrich3Bootcamp)

[**Bootcamp Team 1 - Titanic ML and MOGP Slides**](https://docs.google.com/presentation/d/1YB6Ebc-ghr86Fm3_Evw8EsbBS3zByQGmwH0rwu7C9LE/edit?usp=sharing)

[**Bootcamp Team 1 - Midterm Presentation Slides**](https://docs.google.com/presentation/d/1YB6Ebc-ghr86Fm3_Evw8EsbBS3zByQGmwH0rwu7C9LE/edit?usp=sharing)

[**Bootcamp Team 1 - Colab Notebook**](https://colab.research.google.com/drive/1xLTwAyai275zJ02qhR5QDsvrAevK_ea7?authuser=1)

</details>