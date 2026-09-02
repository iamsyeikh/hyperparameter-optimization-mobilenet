# hyperparameter-optimization-mobilenet
A deep learning project focused on hyperparameter optimization using Random Search.

This project implements an image classification model using **MobileNet** with **Transfer Learning** and **Random Search** for hyperparameter optimization.

The main objective is to find an effective combination of hyperparameters that improves the validation performance of the MobileNet model.

## Overview

The model uses **MobileNet pretrained on ImageNet** as the feature extraction backbone. The pretrained layers are frozen, while additional classification layers are added on top.

Hyperparameter optimization is performed using **Random Search** to explore different combinations of:

* Dropout rate
* Number of Dense units
* Learning rate

The model is evaluated using:

* Accuracy
* Precision
* Recall
* Validation Accuracy

## Model Architecture

The model architecture consists of the following components:

```text
Input Image
     │
     ▼
MobileNet (ImageNet)
     │
     │ Frozen
     ▼
Global Average Pooling
     │
     ▼
Dropout
     │
     ▼
Dense Layer (ReLU)
     │
     ▼
Output Layer (Softmax)
```

### Base Model

The MobileNet model is initialized with ImageNet pretrained weights:

```python
base_model = MobileNet(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False
```

The pretrained MobileNet layers are frozen to perform transfer learning without updating the weights of the base model during training.

## Classification Head

After the MobileNet feature extractor, the following layers are added:

```python
GlobalAveragePooling2D()
Dropout()
Dense()
Dense()
```

The final Dense layer uses the `softmax` activation function to produce class probabilities.

```python
outputs = Dense(
    NUM_CLASSES,
    activation='softmax'
)(x)
```

## Hyperparameter Optimization

The project uses **Keras Tuner Random Search** to search for suitable hyperparameter combinations.

```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=28,
    executions_per_trial=1,
    directory='tuning_mobilenet',
    project_name='mobilenet_random_search'
)
```

### Search Space

| Hyperparameter | Search Range        |
| -------------- | ------------------- |
| Dropout        | 0.2 – 0.5           |
| Dense Units    | 64 – 256            |
| Learning Rate  | 0.01, 0.001, 0.0001 |

The dropout rate is increased in steps of `0.1`, while the number of Dense units is tested in increments of `64`.

### Random Search Configuration

| Parameter            |               Value |
| -------------------- | ------------------: |
| Objective            | Validation Accuracy |
| Maximum Trials       |                  28 |
| Executions per Trial |                   1 |
| Epochs per Trial     |                   4 |

## Model Compilation

The model uses the Adam optimizer:

```python
Adam(
    learning_rate=hp.Choice(
        'learning_rate',
        [1e-2, 1e-3, 1e-4]
    )
)
```

The loss function is categorical cross-entropy:

```python
loss='categorical_crossentropy'
```

The following metrics are monitored:

```text
Accuracy
Precision
Recall
```

## Training

The tuner is trained using the training and validation generators:

```python
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=4
)
```

Each trial trains a different randomly selected combination of hyperparameters.

## Requirements

Install the required Python packages:

```bash
pip install tensorflow keras-tuner numpy
```

Depending on the environment, `scikit-learn` and other supporting packages may also be required.

## Project Structure

A recommended project structure is:

```text
.
├── dataset/
│   ├── train/
│   └── validation/
│
├── models/
│
├── notebooks/
│
├── tuning_mobilenet/
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Organize the dataset into training and validation directories according to the data generator configuration used by the project.

### 5. Run the training

```bash
python train.py
```

The Random Search process will evaluate the specified number of trials and store the tuning results in:

```text
tuning_mobilenet/
```

## Evaluation

The primary optimization objective is:

```text
Validation Accuracy
```

The final model can additionally be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics provide a more comprehensive evaluation of classification performance.

## Hyperparameter Tuning Workflow

```text
Dataset
   │
   ▼
Train / Validation Split
   │
   ▼
MobileNet with ImageNet Weights
   │
   ▼
Freeze Base Model
   │
   ▼
Add Classification Head
   │
   ▼
Random Search
   │
   ├── Dropout
   ├── Dense Units
   └── Learning Rate
   │
   ▼
Train Multiple Trials
   │
   ▼
Evaluate Validation Accuracy
   │
   ▼
Select Best Hyperparameters
   │
   ▼
Best MobileNet Model
```

## Future Improvements

Potential improvements for this project include:

* Increasing the number of Random Search trials.
* Using more training epochs.
* Adding data augmentation.
* Applying learning-rate scheduling.
* Comparing MobileNet with MobileNetV2, MobileNetV3, or other CNN architectures.
* Performing fine-tuning on the pretrained MobileNet layers.
* Adding F1-score as an evaluation metric.
* Comparing Random Search with Grid Search or Bayesian Optimization.

## License

This project is intended for educational and research purposes.
