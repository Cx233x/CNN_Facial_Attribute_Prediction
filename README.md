# CNN-Based Facial Attribute Prediction

**Computer Vision · Deep Learning · Single-Task vs. Multi-Task Learning**

An experimental comparison of convolutional neural networks (CNNs) for predicting three dataset-labeled facial attributes—**age, gender, and ethnicity**—from images. The project evaluates two modeling strategies (independent task-specific CNNs and a shared-backbone multi-task CNN) across two dataset configurations.

**Tech stack:** Python · TensorFlow / Keras · NumPy · pandas · scikit-learn · OpenCV · Pillow · Matplotlib · Jupyter

## Project at a Glance

| Prediction task | Model output | Evaluation metric |
| --- | --- | --- |
| Age estimation | Estimated age in years | Mean Absolute Error (MAE) ↓ |
| Gender classification | Binary label defined by the dataset | Accuracy ↑ |
| Ethnicity classification | One of five dataset-defined categories | Accuracy ↑ |

**Research questions**

- How does a **multi-task CNN with shared feature extraction** compare with three independently trained CNNs?
- How do the recorded results differ when the models are trained using an **original** versus a **combined** facial-image dataset?
- What trade-offs arise from sharing representations across regression and classification tasks?

> **Interpretation note:** The gender and ethnicity categories are annotations supplied by the dataset, not verified or definitive descriptions of a person's identity. This is a research/educational project, not a tool for identifying people or making decisions about them.

## Experimental Design

The project uses a **2 × 2 design**: architecture (single-task or multi-task) × training data (original or combined).

| Configuration | Approach | Training data | Notebook |
| --- | --- | --- | --- |
| 1 | Single-task CNNs | Original | [`02_single_task_original.ipynb`](notebooks/02_single_task_original.ipynb) |
| 2 | Multi-task CNN | Original | [`03_multi_task_original.ipynb`](notebooks/03_multi_task_original.ipynb) |
| 3 | Single-task CNNs | Combined | [`04_single_task_combined.ipynb`](notebooks/04_single_task_combined.ipynb) |
| 4 | Multi-task CNN | Combined | [`05_multi_task_combined.ipynb`](notebooks/05_multi_task_combined.ipynb) |

### Model architectures

**Single-task approach:** three separate CNNs are trained for age regression, gender classification, and ethnicity classification. Each task has its own feature extractor and prediction layers.

**Multi-task approach:** one CNN uses shared convolutional feature extraction with three task-specific output heads. The optimization objective combines a regression loss for age and classification losses for the other tasks with task-specific weights.

Both approaches process images as **48 × 48 grayscale** inputs. The notebooks contain the full architecture definitions, training configurations, regularization choices, and evaluation procedures.

## Recorded Results

The following numbers are taken from the **saved evaluation outputs** of the two multi-task notebooks. They have **not been independently rerun or verified** as part of this README update.

| Multi-task CNN | Age MAE (years) ↓ | Gender accuracy ↑ | Ethnicity accuracy ↑ |
| --- | ---: | ---: | ---: |
| Original dataset | **6.91** | **86.23%** | **73.53%** |
| Combined dataset | **8.82** | **79.92%** | **65.78%** |

Sources: [`03_multi_task_original.ipynb`](notebooks/03_multi_task_original.ipynb) and [`05_multi_task_combined.ipynb`](notebooks/05_multi_task_combined.ipynb).

**Observations:** On the evaluation splits recorded in these notebooks, the original-data multi-task model reports lower age error and higher classification accuracy than the combined-data version. This is an **observed difference between experiment runs**, not evidence on its own that adding data reduces model performance: the datasets and evaluation splits are not established here as a common, untouched benchmark.

The single-task notebooks contain their own training/evaluation cells and prediction examples. Their results are not included in this summary table because a consistent, cross-experiment benchmark has not yet been assembled.

## Data Preparation

The experiments use the **UTKFace** dataset and a tabular representation of facial images and associated labels.

- **Original data:** `age_gender.csv` (used by the original-data experiments).
- **Additional image data:** UTKFace `.jpg` files, whose filenames encode age, gender, and ethnicity labels.
- **Combined data:** `combined_dataset.csv`, produced by converting additional images into a compatible 48 × 48 pixel representation and concatenating them with the original records.

The preprocessing workflow is in [`01_data_preprocessing.ipynb`](notebooks/01_data_preprocessing.ipynb), which generates intermediate files such as `utkface_48x48.csv` and `utkface_compact.csv`.

Data sources: [UTKFace project page](https://susanqq.github.io/UTKFace/) · [Project dataset folder referenced in the original documentation](https://drive.google.com/drive/folders/1WfbqyYX9-Aa9nAmyQ36obICfAEnrRHrX?usp=drive_link)

**Data use:** UTKFace is made available for **non-commercial research purposes**; image rights remain with their original owners. Obtain and use the data in accordance with the source terms. Local datasets and raw images are intentionally excluded from Git tracking.

## Repository Layout

```text
CNN_Facial_Attribute_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_single_task_original.ipynb
│   ├── 03_multi_task_original.ipynb
│   ├── 04_single_task_combined.ipynb
│   └── 05_multi_task_combined.ipynb
├── models/
│   ├── single_task/
│   │   ├── original/          # Age, gender, ethnicity .h5 weights
│   │   └── combined/          # Age, gender, ethnicity .h5 weights
│   └── multi_task/
│       ├── original/          # Full multi-task .h5 model
│       └── combined/          # Full multi-task .h5 model
└── assets/
    └── sample_images/        # Example inputs used in prediction cells
```

The notebooks also expect a **local `data/` directory**, which is not committed with the datasets. Some experiment output directories may likewise be created locally rather than appearing in the repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Cx233x/CNN_Facial_Attribute_Prediction.git
cd CNN_Facial_Attribute_Prediction
```

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. To run notebooks in VS Code, install its **Python** and **Jupyter** extensions and select the virtual environment as your notebook kernel (install `ipykernel` if prompted).

### 3. Prepare the data directory

From the repository root:

```bash
mkdir -p data/raw/image
```

Place `age_gender.csv` in `data/` for the original-data experiments. For the image-processing workflow, place the permitted UTKFace `.jpg` files in `data/raw/image/` and run `notebooks/01_data_preprocessing.ipynb` to generate the combined CSV. Alternatively, use an already prepared `data/combined_dataset.csv` if you have access to an appropriately licensed copy.

Expected local layout:

```text
data/
├── age_gender.csv
├── combined_dataset.csv        # Generated or obtained separately
└── raw/
    └── image/                  # UTKFace images, when preprocessing
```

### 4. Run the notebooks

Open the repository in VS Code or Jupyter and run the relevant notebook cells in order:

- `01_data_preprocessing.ipynb` — optional if the combined CSV is already available; required to generate it from raw images.
- `02_single_task_original.ipynb` and `03_multi_task_original.ipynb` — original dataset experiments.
- `04_single_task_combined.ipynb` and `05_multi_task_combined.ipynb` — combined dataset experiments.

The four modeling notebooks represent **alternative experiment configurations**, not sequential training stages. Their path setup assumes the working directory is either the repository root or `notebooks/`.

Model checkpoints are stored under `models/` in the corresponding architecture/dataset subdirectory. The single-task notebooks save **weights**; the multi-task notebooks save **full models**.

### 5. Try the saved single-task models

The single-task notebooks contain `create_*_model()` functions, weight-loading cells, image preprocessing, and a `predict_all()` helper. After initializing these cells in [`02_single_task_original.ipynb`](notebooks/02_single_task_original.ipynb), an example call is:

```python
age, gender, ethnicity = predict_all(
    ASSETS_DIR / "sample_images" / "testimage1.jpg"
)
print(f"Predicted age: {age:.1f} years")
print(f"Predicted gender label: {gender}")
print(f"Predicted ethnicity category: {ethnicity}")
```

This example **depends on the notebook's model definitions and loaded weights**; it is not a standalone prediction script. Predictions are model estimates of dataset labels, not verified personal characteristics.

## Limitations and Responsible Use

- The evaluation figures above come from stored notebook runs and should not be treated as an external benchmark.
- The original experimental workflows reuse held-out data for validation during training; a future controlled comparison should use a separate validation set and an untouched common final test set.
- Dataset composition, labeling practices, and class imbalance may affect reported performance. Differences between the two dataset configurations should not be interpreted causally without additional controls.
- Face images vary in pose, lighting, occlusion, resolution, and other properties that can affect model predictions.
- The dataset's binary gender and five-category ethnicity labels are limited proxies and do not capture people's self-identified attributes.
- No subgroup fairness or external generalization evaluation is claimed here.
- This project is intended solely for education and research; **do not use these predictions for identification, screening, access control, or other consequential decisions about people**.

## Acknowledgments

This work originated as a collaborative final project for **DS340: Machine Learning & AI at Boston University**. The original project repository is [zihaoguo3/DS340-Final-Project](https://github.com/zihaoguo3/DS340-Final-Project).

Dataset credit: [UTKFace](https://susanqq.github.io/UTKFace/). Please consult the original dataset source for its usage terms and attribution information.
