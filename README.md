# CIS6930 - Natural Language Processing 

Repository for the course project of CIS6930 (NLP)

## Project Title: Exploiting Explainability to Design Adversarial Attacks and Evaluate Attack Resilience in Hate-Speech Detection Models

### (S2P2) Team Members:

-   **Sohaib Uddin Syed** - Captain, Linguistics and Data Processing
-   **Pranath Reddy Kumbam** - Experiment Design, Algorithm Implementation and Model Training 
-   **Suhas Harish** - Python Programming and Implementation
-   **Prashanth Thamminedi** - Metrics Calculation, Analysis and Slide Creation

### Overview

In this project, we explore various text classification models for hate speech detection. We use the Kaggle "Hate Speech and Offensive Language Dataset" and UC Berkley "Measuring Hate Speech" dataset from HuggingFace to train and evaluate the performance of the following models:

1.  Random Forest Classifier
2.  Convolutional Neural Network (CNN)
3.  Long Short-Term Memory (LSTM)
4.  Distil BERT

The main objective of our research is to exploit explainability to design adversarial attacks and evaluate the attack resilience of these hate-speech detection models.

### Datasets

-   [Kaggle "Hate Speech and Offensive Language Dataset"](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
-   [UC Berkley "Measuring Hate Speech" dataset](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (from HuggingFace)

### File Structure

```
.
├── Colab NoteBooks
│   ├── HuggingFace
│   │   ├── HuggingFaceCNN.ipynb
│   │   ├── HuggingFaceDistilBERT-TPU.ipynb
│   │   └── HuggingFaceLSTM.ipynb
│   └── Kaggle
│       ├── KaggleCNN.ipynb
│       ├── KaggleDistilBERT-TPU.ipynb
│       ├── KaggleLSTM.ipynb
|       └── KaggleLSTM_Experimentation.ipynb
├── ConfMat_Binary.py
├── ConfMat_Multi.py
├── Data
│   └── Kaggle_Hate_Speech_Dataset.csv
├── HFDataSave.py
├── HuggingFaceRF.py
├── KaggleRF-Baseline.py
├── KaggleRF.py
└── Results
    ├── confusion_matrices
    └── performance_metrics
```

[Link to Model Weights](https://drive.google.com/drive/folders/1qtXdbE8sqyMTq-FZcEdA_1IcED1ISRD3?usp=sharing)

### Usage

To run the scripts, install the required dependencies [to be added soon] and execute the scripts.

#### Colab NoteBooks

This folder contains Google Colab Notebooks for training and evaluating deep learning models (CNN, LSTM, and Distil BERT) on the datasets. To run the notebooks, open them in Google Colab/Drive and execute them.

#### ConfMat_Binary.py and ConfMat_Multi.py

These scripts are used for plotting confusion matrices for binary and multi-class classification respectively. 

#### Data

This folder contains the Kaggle "Hate Speech and Offensive Language Dataset" in CSV format. 

#### HFDataSave.py

This script saves the UC Berkley "Measuring Hate Speech" dataset from HuggingFace for manual data exploration. 

#### HuggingFaceRF.py

This script loads and processes the UC Berkley "Measuring Hate Speech" dataset from HuggingFace and trains a Random Forest model on it. 

#### KaggleRF-Baseline.py and KaggleRF.py

These scripts load and process the Kaggle "Hate Speech and Offensive Language Dataset" and train a Random Forest model. KaggleRF-Baseline.py uses baseline data preprocessing, whereas KaggleRF.py uses custom preprocessing. 

#### Results

This folder contains the generated confusion matrices and performance metrics for all models
