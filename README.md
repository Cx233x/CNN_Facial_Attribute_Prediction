# CNN-Based Facial Attribute Prediction

**Deep Learning | Computer Vision | CNN | Multi-Task Learning | TensorFlow**

This is a deep learning project that investigates how **Convolutional Neural Networks (CNNs)** can estimate facial attributes from images, comparing single-task and multi-task learning approaches across different training datasets.

**Tech Stack:** Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, OpenCV, Matplotlib

---

## 1. Project Overview

This project explores whether a computer can learn visual patterns from facial images to predict three attributes:

| Prediction Task | Output | Problem Type |
|---|---|---|
| Age Prediction | Estimated age in years | Regression |
| Gender Classification | Male / Female dataset labels | Binary Classification |
| Ethnicity Classification | White, Black, Asian, Indian, Other dataset labels | Multi-class Classification |

These categories reflect the labels provided by the training datasets. They are not definitive descriptions of an individual's identity.

### The Main Research Question

**Can one neural network learn multiple related prediction tasks as effectively as separate specialized networks?**

To investigate this, we compare two approaches:

**Approach A: Single-Task Learning**

Build three independent CNN models. Each network specializes in predicting one attribute.

**Approach B: Multi-Task Learning**

Build one CNN with a shared feature extraction network and three task-specific output branches.

We also investigate whether expanding the training dataset affects model performance.

### Key Research Objectives

1. Develop CNN architectures for age regression and facial attribute classification.
2. Compare single-task and multi-task learning approaches.
3. Evaluate the effect of combining facial image datasets.
4. Analyze model performance using task-specific evaluation metrics.

---

## 2. How the Models Work

A **Convolutional Neural Network (CNN)** is a deep learning model designed to recognize patterns in images.

For example, a CNN can learn simple features such as edges and shapes, then combine them into more complex visual representations.

In this project, each input image is converted into a **48 × 48 grayscale image** before being processed by the model.

The general pipeline is:

```text
      Input Facial Image
              |
              v
       Image Preprocessing
     (48 × 48, Grayscale)
              |
              v
      Convolutional Layers
       (Feature Learning)
              |
              v
       Prediction Layers
              |
              v
      Predicted Attributes
```

### Architecture A: Single-Task CNN

In single-task learning, each model focuses on one prediction.

```text
                 Input Image
                      |
               Preprocessing
                      |
         +------------+------------+
         |            |            |
         v            v            v
      Age CNN     Gender CNN   Ethnicity CNN
         |            |            |
         v            v            v
    Age Estimate    Gender      Ethnicity
```

**Advantages:**

- Each model can be optimized for a specific task.
- Model performance can be evaluated independently.
- Changes to one model do not directly affect the other models.

**Trade-offs:**

- Three separate models must be trained and maintained.
- The models do not share their learned image features.

### Architecture B: Multi-Task CNN

In multi-task learning, the model shares its convolutional layers across the three prediction tasks.

```text
                  Input Image
                       |
                Preprocessing
                       |
                Shared CNN
             (Feature Extraction)
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Age Head    Gender Head  Ethnicity Head
          |            |            |
          v            v            v
     Age Estimate    Gender      Ethnicity
```

The shared CNN learns image features, while each prediction head specializes in a particular task.

**Advantages:**

- Different tasks can reuse the same learned features.
- Shared feature extraction may reduce duplicated computation.
- Related tasks may benefit from learning together.

**Trade-offs:**

- Training requires balancing the objectives of multiple tasks.
- Improvements in one task may not improve the others.
- Multi-task learning is not guaranteed to outperform separate models.

### Model Training Techniques

The architectures use several deep learning techniques:

| Technique | Purpose |
|---|---|
| Convolutional Layers | Learn visual patterns from images |
| Max Pooling | Reduce feature map dimensions |
| Batch Normalization | Help stabilize training |
| Dropout | Reduce the risk of overfitting |
| L2 Regularization | Discourage excessively large model weights |
| Adam Optimizer | Update model parameters during training |
| Early Stopping | Stop training when validation loss stops improving |
| ReduceLROnPlateau | Reduce the learning rate when monitored performance plateaus |

The multi-task models also use separate losses for regression and classification, combined into a weighted training objective.

---

## 3. Experimental Design

The project compares models across **two dimensions**:

1. **Model Architecture:** Single-task vs. Multi-task CNN
2. **Training Dataset:** Original Dataset vs. Combined Dataset

This creates four experimental configurations:

| Experiment | Architecture | Training Dataset |
|---|---|---|
| Experiment 1 | Single-task CNN | Original dataset |
| Experiment 2 | Multi-task CNN | Original dataset |
| Experiment 3 | Single-task CNN | Combined dataset |
| Experiment 4 | Multi-task CNN | Combined dataset |

### "Combined Dataset"

The original experiments use `age_gender.csv`.

The expanded-data experiments use `combined_dataset.csv`, created by combining:

- `age_gender.csv` — The original facial attribute dataset.
- `utkface_compact.csv` — Processed facial images from UTKFace, converted into a compatible tabular format.

The data preparation notebook, `dataprocess.ipynb`, performs this process.

```text
       Original UTKFace Images
                  |
                  v
        Image Preprocessing
        Resize to 48 × 48
                  |
                  v
        utkface_compact.csv
                  |
                  +------------------+
                  |                  |
                  v                  v
           Processed Data       age_gender.csv
                  |                  |
                  +---------+--------+
                            |
                            v
                  combined_dataset.csv
```

The datasets are combined by **adding rows of examples**, rather than adding new image features.

**Why try a combined dataset?**

Increasing the number and variety of training examples may help models learn more generalizable patterns.

However, additional data can also introduce differences in label definitions, sample distributions, or data quality.

Therefore, we evaluate the combined dataset experimentally rather than assuming it produces better results.

---

## 4. Technologies Used

| Technology | What It Does |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Builds, trains, and evaluates CNN models |
| NumPy | Performs numerical operations |
| Pandas | Loads, cleans, and combines datasets |
| Scikit-learn | Splits data into training and testing sets |
| OpenCV | Processes and resizes facial images |
| Pillow | Opens and transforms images |
| Matplotlib | Visualizes images and model results |
| Jupyter Notebook | Organizes code, experiments, and results |
| VS Code | Local development environment |

---

## 5. Repository Structure

```text
CNN-Facial-Attribute-Prediction/
|
|-- README.md
|
|-- dataprocess.ipynb
|   # Prepares and combines facial image datasets
|
|-- single_predictor/
|   |-- models.ipynb
|   |-- age_weights.h5
|   |-- gender_weights.h5
|   |-- ethnicity_weights.h5
|
|-- Muti-feature/
|   |-- Multi_Feature_Model.ipynb
|   |-- multi_feature_model.h5
|
|-- single_predictor_with_new_data/
|   |-- models_new_dataipynb.ipynb
|   |-- age_weights_2.h5
|   |-- gender_weights_2.h5
|   |-- ethnicity_weights_2.h5
|
|-- Muti-feature_new_data/
|   |-- Multi_Feature_Combined_Data.ipynb
|   |-- multi_feature_model_combined_data.h5
|
|-- testimage/
    |-- testimage1.jpg
    |-- testimage2.jpg
    |-- testimage3.jpg
    |-- testimage4.jpg
```

### File Extensions

- **`.ipynb`** — A Jupyter Notebook containing executable Python code and its experiment outputs.
- **`.h5`** — A saved neural network model or a collection of trained model weights.
- **`.csv`** — A tabular data file, similar to a spreadsheet.
- **`.jpg`** — An image file.
- **`.md`** — A Markdown document used for project documentation.

---
## 6. Getting Started

### Step 1: Install Python and VS Code

1. Install [Python](https://www.python.org/downloads/) and [VS Code](https://code.visualstudio.com/).
2. In VS Code, install the **Python** and **Jupyter** extensions by Microsoft.
3. Verify your Python installation in Terminal:

```bash
python3 --version
```

*Windows users can use `py --version` instead.*

### Step 2: Download the Repository

Clone the project using Git:

```bash
git clone https://github.com/Cx233x/CNN_Facial_Attribute_Predicion.git
cd CNN-Facial-Attribute-Prediction
```

Alternatively, click **Code → Download ZIP** on GitHub, extract the files, and open the folder in VS Code.

### Step 3: Set Up the Environment

Create and activate a virtual environment to manage project dependencies.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
python -m pip install tensorflow numpy pandas matplotlib scikit-learn pillow opencv-python ipykernel
```

> **Note:** The pretrained `.h5` files use a legacy Keras format and may require compatibility adjustments with newer TensorFlow versions.

### Step 4: Prepare the Dataset

The project uses two training datasets:

| Dataset | Purpose |
|---|---|
| `age_gender.csv` | Original single-task and multi-task experiments |
| `combined_dataset.csv` | Experiments using the expanded dataset |

Download the datasets from the [Google Drive folder](https://drive.google.com/drive/folders/1WfbqyYX9-Aa9nAmyQ36obICfAEnrRHrX?usp=drive_link).

Place the CSV files in the project's working directory, or organize them under a `data/` folder and update the corresponding paths in the notebooks.

### Step 5: Run the Notebooks

1. Open the project folder in VS Code.
2. Select the `.ipynb` notebook you want to explore (see [Repository Structure](#5-repository-structure)).
3. Click **Select Kernel** and choose your `.venv` Python environment.
4. Run the cells sequentially using **▶ Run Cell**.

The notebooks cover data preprocessing, model training, evaluation, and saving trained models.

> **Note:** Some original notebooks contain environment-specific code and may require minor adjustments before running locally.

### Step 6: Open a Jupyter Notebook

Now you can begin exploring the models.

For your first experiment, open:

```text
single_predictor/models.ipynb
```

In VS Code:

1. Locate the file in the Explorer sidebar.
2. Click the notebook to open it.
3. Select **Select Kernel** in the upper-right corner.
4. Choose the Python environment named `.venv`.
5. Use the **▶ Run Cell** button to execute individual cells.

A notebook is made up of cells.

Each cell contains a small part of the program. You can run them one by one to understand what each step does.

### Step 7: Understand the Training Process

The model notebooks follow a general machine learning workflow.

### Part A: Load the Dataset

The notebook reads the CSV data into a Pandas DataFrame.

```python
df = pd.read_csv("age_gender.csv")
```

### Part B: Preprocess the Images

Images are converted into numerical arrays.

Each image contains 48 × 48 = **2,304 grayscale pixel values**.

Pixel values are normalized to help the neural network train more consistently.

For example:

```python
X = X / 255.0
```

This transforms pixel values from the 0–255 range into the 0–1 range.

### Part C: Split the Dataset

The notebooks use `train_test_split` to divide data into:

- **80% training data** — Used to train the neural network.
- **20% held-out data** — Used to evaluate predictions during the experiment.

The same held-out split is also supplied as validation data during training in the existing notebooks.

For a more rigorous future evaluation, a separate validation set and untouched final test set would be preferable.

### Part D: Build the CNN

The code defines the layers of the neural network.

For example, a convolutional layer extracts image features:

```python
Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same"
)
```

The network learns increasingly useful visual patterns from the training images.

### Part E: Train the Model

The `.fit()` function trains the network.

For example:

```python
model.fit(
    X_train,
    y_train,
    epochs=50
)
```

This is a simplified illustration; the actual notebooks contain task-specific training calls and additional arguments.

An **epoch** means one complete pass through the training dataset.

The model repeatedly processes examples, measures prediction errors, and adjusts its parameters.

Training can take time depending on the dataset, model complexity, and available hardware.

### Part F: Evaluate the Model

After training, the notebook evaluates the model using the held-out examples.

The main metrics are:

- Age Mean Absolute Error (MAE)
- Gender Classification Accuracy
- Ethnicity Classification Accuracy

See the **Model Evaluation** section below for details.

### Part G: Save the Model

The notebooks save trained models or their learned weights as `.h5` files.

These files allow the trained parameters to be reused without repeating the entire training process.

---

### Step 8: Try Predictions Using Saved Weights

The `single_predictor/` folder includes three saved weight files:

```text
age_weights.h5
gender_weights.h5
ethnicity_weights.h5
```

The original notebook contains model-building functions and a prediction helper.

To use these weights:

1. Open `single_predictor/models.ipynb`.
2. Run the import and model-definition cells.
3. Load the saved weights into the matching model architectures.
4. Run the image preprocessing and prediction function cells.
5. Provide a valid test image path.

**Important:** The original notebook is organized as a training experiment, not as a standalone inference application. Reusing pretrained weights without retraining may require rearranging the notebook cells and making compatibility adjustments.

The project contains example images in:

```text
testimage/
```

Once the models and `predict_all()` function have been initialized successfully, the prediction helper can be called as follows:

```python
age, gender, ethnicity = predict_all(
    "testimage/testimage1.jpg"
)

print(f"Predicted Age: {age:.1f} years")
print(f"Predicted Gender: {gender}")
print(f"Predicted Ethnicity: {ethnicity}")
```

This example assumes the current working directory is the repository root.

### What Should the Output Look Like?

```text
Predicted Age: <estimated age>
Predicted Gender: <predicted label>
Predicted Ethnicity: <predicted dataset category>
```

The numerical results depend on the input image and trained weights.

These predictions should not be interpreted as verified personal attributes.

---

## 7. Model Evaluation

### Age Prediction — Mean Absolute Error

Age prediction is a regression task.

The primary metric is **Mean Absolute Error (MAE)**:

**MAE:** `MAE = (1/n) × Σ|Actual Age - Predicted Age|`

### Gender and Ethnicity Classification — Accuracy
**Accuracy:** `Accuracy = Correct Predictions / Total Predictions`

For example, 80% accuracy means that 80 out of 100 evaluated predictions match their labels.

### Experimental Comparison

The four model configurations can be summarized using:

| Model | Age MAE ↓ | Gender Accuracy ↑ | Ethnicity Accuracy ↑ |
|---|---|---|---|
| Single-task CNN — Original Data | Pending verification | Pending verification | Pending verification |
| Multi-task CNN — Original Data | Pending verification | Pending verification | Pending verification |
| Single-task CNN — Combined Data | Pending verification | Pending verification | Pending verification |
| Multi-task CNN — Combined Data | Pending verification | Pending verification | Pending verification |

The repository contains evaluation code and saved experiment outputs. 

---

## 8. Limitations and Responsible Use

This project is intended for **educational and experimental purposes**.

Several limitations should be considered:

- Model performance depends on the quality and representativeness of the training data.
- Dataset-defined ethnicity labels do not establish an individual's actual ethnicity.
- Binary gender labels do not represent the full range of gender identities.
- Predictions can be affected by lighting, image quality, pose, and other factors.
- Differences in dataset composition can make model comparisons difficult.
- The original notebook environment and saved models have not been fully validated for modern dependency versions.
- This project should not be used to make consequential decisions about individuals.

---

## 9. Acknowledgments

This repository originated as a collaborative final project for **DS340: Machine Learning & AI at Boston University**.

**Original Repository:** [zihaoguo3/DS340-Final-Project](https://github.com/zihaoguo3/DS340-Final-Project)

The project uses facial image datasets for research and learning. All dataset usage, redistribution, and deployment must comply with the applicable licenses and usage conditions.
