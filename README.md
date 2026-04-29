# exp-tracking-DVC
Tracking our version data with DVC

### Water Potability Experiment Tracking with DVC

This project is a machine learning pipeline designed to predict whether water is safe for human consumption. It uses **modern MLOps** practices to track experiments, manage data versions, and monitor model performance over time.

# Project Overview

The main goal is to build a reliable system that takes water quality metrics and determines potability. To make this professional, I used **DVC (Data Version Control) and DVCLive**. This allows us to keep a history of every model we train and see exactly how changes to parameters affect our accuracy and precision.

# Project Layout

```txt
exp_tracking_dvc
|
|_ data/ ........................ Storage for your datasets
|  |_ water_potability.csv ...... The original data used for training the model
|
|_ dvclive/ ..................... Results created by the experiment tracker
|  |_ plots/metrics/ ............ Data files used to draw performance graphs
|  |  |_ accuracy:.tsv .......... Progress of accuracy during training
|  |  |_ f1_score:.tsv .......... History of the F1 balance score
|  |  |_ precision:.tsv ......... History of the precision score
|  |  |_ recall_score:.tsv ...... History of the recall score
|  |_ metrics.json .............. A snapshot of the final model scores
|  |_ params.yaml ............... A log of the specific settings used for the run
|
|_ src/ ......................... Folder for your project code
|  |_ water_model.py ............ The main script that cleans data and trains the model
|
|_ dvc.yaml ..................... The roadmap that tells DVC how to run the project
|_ model.pkl .................... The final "brain" or trained model file
|_ requirements.txt ............. The list of tools needed to run this code
```

## How to Get Started

To run this project on your local machine, follow these steps.
1. Set up the environment

First, create a virtual environment and install the necessary libraries.
```bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Pipeline

Instead of running the Python scripts manually, use DVC. This ensures that every step is tracked and only updated if something has changed.
```bash

dvc repro
```

3. View the Results

After the model finishes training, you can check the performance metrics. I have configured the system to track Accuracy, F1 Score, Precision, and Recall.
```bash

dvc metrics show
```

To see the visual progress of these metrics over time, run:
```bash

dvc plots show
```

# Tracking Experiments

I am using **DVCLive** to monitor the training process. Every time the model trains, it records data into the dvclive/ folder. Currently, the model is achieving the following results:
```txt
Metric	Current Value
Accuracy	0.658
Precision	0.648
F1 Score	0.419
Recall	0.310
```

These values are saved in .tsv files within the plots folder, making it easy to compare different versions of the model as we tune the hyperparameters.
Pipeline Stages

The workflow is broken down into clear stages defined in the dvc.yaml file:

**Training**: The src/water_model.py script reads the data, fills in missing values, and trains a Random Forest Classifier.

**Logging**: DVCLive captures the parameters used and the resulting metrics automatically.

**Artifacts**: The final trained model is saved as model.pkl for later use in predictions.

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for educational or professional purposes. See the **LICENSE** file for more details.