# Student Exam Performance Indicator

This is an end-to-end Machine Learning project that predicts a student's performance in Math based on various social and personal factors. The project includes data exploration, a full ML pipeline for training, and a web application for inference.

## Table of Contents
- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)

## Overview

The primary objective of this project is to build a regression model that accurately predicts students' math scores. The model is trained on a dataset containing information about students' gender, ethnicity, parental education level, lunch status, and test preparation course completion, along with their reading and writing scores.

The project culminates in a simple web application built with Flask, where a user can input the required features and receive an instant prediction of the math score.

## Project Workflow

The project follows a standard machine learning project life cycle:

1.  **Data Ingestion**: The initial step involves loading the raw data, splitting it into training and testing sets, and storing them as artifacts.
2.  **Exploratory Data Analysis (EDA)**: The `EDA STUDENT PERFORMANCE.ipynb` notebook contains a detailed analysis of the dataset, exploring relationships between variables and visualizing distributions to gain insights.
3.  **Data Transformation**: A preprocessing pipeline is constructed to handle both numerical and categorical data. This includes:
    *   Imputing missing values.
    *   Scaling numerical features using `StandardScaler`.
    *   Encoding categorical features using `OneHotEncoder`.
    *   The resulting preprocessing object is saved as a pickle file for later use.
4.  **Model Training**: Multiple regression models are trained and evaluated:
    *   Linear Regression
    *   Random Forest Regressor
    *   Decision Tree Regressor
    *   Gradient Boosting Regressor
    *   XGBoost Regressor
    *   CatBoost Regressor
    *   AdaBoost Regressor
    Hyperparameter tuning is performed using `GridSearchCV`. The best model is selected based on its R2 score and saved as `model.pkl`.
5.  **Prediction Pipeline**: A prediction pipeline is established to streamline the process of making predictions on new data. It loads the saved preprocessor and model to transform the input data and return a prediction.
6.  **Web Application**: A Flask application provides a user-friendly interface to interact with the model. Users can input student details via a web form and get the predicted math score.

## Dataset

The dataset used for this project is `stud.csv`, which contains 1000 records of student data.

**Features**:
*   `gender`: (Male/Female)
*   `race_ethnicity`: (Group A, B, C, D, E)
*   `parental_level_of_education`: (bachelor's degree, some college, etc.)
*   `lunch`: (standard or free/reduced)
*   `test_preparation_course`: (completed or none)
*   `reading_score`: (score out of 100)
*   `writing_score`: (score out of 100)

**Target Variable**:
*   `math_score`: (score out of 100)

## Technologies Used

*   **Backend**: Flask
*   **ML Libraries**: Scikit-learn, Pandas, NumPy
*   **Boosting Libraries**: XGBoost, CatBoost
*   **Serialization**: Dill, Pickle
*   **Development & IDE**: Jupyter Notebook, VS Code

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

**1. Clone the repository:**
```bash
git clone <repository-url>
cd <repository-folder>
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required packages:**
The `setup.py` file is configured to install dependencies from `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Run the Flask application:**
```bash
python app.py
```

**5. Access the application:**
Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage

1.  Open the application in your browser.
2.  You will be presented with a form to enter student details.
3.  Fill in all the fields for gender, race/ethnicity, parental education, lunch type, test preparation, writing score, and reading score.
4.  Click on the "Predict your Maths Score" button.
5.  The application will display the predicted math score on the same page.

## Model Training and Evaluation

The model training process is detailed in the `MODEL TRAINING.ipynb` notebook. It involves:

1.  Loading the preprocessed data.
2.  Training a dictionary of regression models.
3.  Using `GridSearchCV` to find the best hyperparameters for each model type.
4.  Evaluating each model based on the R2 score on the test set.
5.  The model with the highest R2 score is selected as the `best_model` and saved to `artifacts/model.pkl`. In this project, **Linear Regression** was found to be the best performing model.
