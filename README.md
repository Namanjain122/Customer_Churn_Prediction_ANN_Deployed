# Customer Churn Prediction Using ANN

This project builds and deploys an Artificial Neural Network (ANN) to predict whether a bank customer is likely to churn (leave the bank). It uses a customer dataset from the classic Churn Modelling dataset and exposes a simple interactive web interface with Streamlit.

The app allows a user to enter customer details such as age, balance, credit score, geography, and membership status, then returns a churn probability and a binary prediction.

## Project Overview

Customer churn prediction is an important problem in banking and customer relationship management. Predicting churn early allows businesses to:

- identify at-risk customers
- design targeted retention strategies
- reduce revenue loss
- improve customer satisfaction

This project demonstrates a complete machine learning workflow:

1. loading and preprocessing customer data
2. training an ANN model
3. saving the trained model and preprocessing artifacts
4. creating a Streamlit app for model inference
5. using the model to predict churn for new customer records

## Business Problem

Banks and financial institutions often face the challenge of customers leaving their services. Churn can result in lost revenue and increased acquisition costs. By predicting churn risk, banks can proactively offer incentives, adjust service quality, or contact customers before they leave.

The model predicts whether a customer will exit a bank based on features such as:

- geography
- gender
- age
- tenure
- balance
- number of products
- credit card ownership
- activity status
- estimated salary

## Solution Approach

The project uses a deep learning ANN model trained on a customer churn dataset. The workflow includes:

- reading the churn dataset
- dropping irrelevant columns such as RowNumber, CustomerId, and Surname
- encoding categorical features
- scaling numeric features
- training a neural network classifier
- storing preprocessing objects and model weights
- loading them in a Streamlit app for real-time prediction

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- Streamlit
- pickle
- Jupyter Notebook

## Project Structure

```text
Customer Churn Prediction Project Using ANN/
├── Dataset/
│   └── Churn_Modelling.csv
├── models/
│   ├── model.h5
│   ├── Scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
├── notebooks/
│   ├── Training.ipynb
│   └── prediction.ipynb
├── logs/
├── Streamlit_app.py
├── requirements.txt
├── README.md
├── venv/
└── .ipynb_checkpoints/ (if present)
```

## Key Files

### `Dataset/Churn_Modelling.csv`
This is the source dataset used to train the churn model. It contains customer attributes and a target column named `Exited` that indicates whether the customer churned.

### `notebooks/Training.ipynb`
This notebook contains the model-building workflow, including:

- dataset loading
- preprocessing
- train/test split
- feature encoding
- standardization
- ANN training
- model serialization

### `Streamlit_app.py`
This is the application entry point for the deployed model. It:

- loads the trained TensorFlow model
- loads saved preprocessing artifacts
- collects user input from the UI
- transforms the input using the same preprocessing logic
- predicts churn probability
- displays output to the user

### `models/`
This folder stores the trained model and the preprocessing pipelines used at inference time:

- `model.h5` — trained Keras ANN model
- `Scaler.pkl` — StandardScaler fitted to the feature data
- `label_encoder_gender.pkl` — label encoder for the Gender feature
- `onehot_encoder_geo.pkl` — one-hot encoder for the Geography feature

### `requirements.txt`
Lists the Python dependencies needed to run the project.

## Data Used

The dataset is the well-known Bank Customer Churn dataset, with columns including:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited`

The data is used for supervised learning, where the target variable is `Exited`:

- `1` = customer churned
- `0` = customer did not churn

## Model Details

This project uses a neural network model built in Keras. In the training notebook, categorical variables are transformed before the model is trained, and the numeric features are standardized to keep the scale consistent across inputs.

The inference app recreates the same preprocessing steps in real time so that model input always matches the training setup.

## Setup Instructions

### 1. Clone or download the project

```bash
git clone <repository-url>
cd "Customer Churn Prediction Project Using ANN"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Streamlit user interface:

```bash
streamlit run Streamlit_app.py
```

Then open the local URL shown by Streamlit in the browser (usually `http://localhost:8501`).

## How the App Works

The app lets a user enter customer details through the Streamlit interface:

- Geography
- Gender
- Age
- Balance
- Credit Score
- Estimated Salary
- Tenure
- Number of Products
- Has Credit Card
- Is Active Member

The app then:

1. creates a DataFrame from the inputs
2. encodes the geography field using the saved one-hot encoder
3. encodes the gender field using the saved label encoder
4. drops the original categorical field that is not needed in numeric form
5. standardizes the feature values using the saved scaler
6. sends the processed data to the ANN model
7. calculates the churn probability
8. prints whether the customer is likely to churn or not

## Example Prediction Logic

The app computes a probability between 0 and 1:

- value greater than 0.5 → customer is likely to churn
- value less than or equal to 0.5 → customer is unlikely to churn

## Model Usage Notes

This project is designed for demonstration and learning. It is not a production-grade deployment system, but it shows the core process of:

- training a churn prediction model
- persisting model artifacts
- preparing inference data the same way as training
- exposing the model in a user-facing interface

## Potential Improvements

This project can be enhanced further by adding:

- cross-validation and model comparison
- confusion matrix and evaluation metrics
- feature importance analysis
- hyperparameter tuning
- MLOps tracking
- deployment to cloud services
- monitoring for drift and retraining
- a rest API for integration with other systems

## Dependencies

The project dependencies are listed in `requirements.txt` and include:

```text
tensorflow==2.15.0
pandas
numpy
Scikit-learn
tensorboard
matplotlib
streamlit
```

## License

This project is provided for educational and demonstration purposes. If you are using it in a more formal environment, please check whether you need to add your own licensing terms.

## Summary

This repository demonstrates a practical artificial neural network solution for customer churn prediction in the banking domain. It combines a machine learning workflow with a interactive Streamlit interface to make predictions easy to understand and use.

If you are learning machine learning or want a simple end-to-end churn prediction example, this project is a good starting point.
