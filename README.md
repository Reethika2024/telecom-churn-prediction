Telecom Customer Churn Prediction

This project presents an end-to-end machine learning solution for predicting customer churn in the telecom industry. It covers the complete workflow from data analysis and model development to deployment using a Flask web application.

Project Objective

Customer churn significantly impacts revenue in subscription-based businesses.
The objective of this project is to predict whether a telecom customer is likely to churn based on customer demographics, service usage, and billing information. These predictions can help businesses take proactive retention measures.

Dataset

The project uses publicly available telecom customer datasets containing information such as:

Customer demographics

Services subscribed

Contract and payment details

Monthly and total charges

Tenure and churn status

Multiple CSV files are used for analysis and experimentation.

Methodology
Exploratory Data Analysis (EDA)

Studied customer behavior patterns related to churn

Identified correlations between services, tenure, and churn

Handled missing and inconsistent values

Feature Engineering

Encoded categorical variables using one-hot encoding

Grouped tenure into meaningful ranges

Ensured feature consistency between training and inference stages

Handling Class Imbalance

Applied SMOTE-ENN to address skewed churn distribution and improve recall

Model Training

Trained a Random Forest Classifier

Evaluated performance using accuracy, recall, confusion matrix, and classification report

Model Deployment

Serialized the trained model using pickle

Built a Flask web application to perform real-time churn prediction with confidence scores

Tech Stack

Programming Language: Python 3.11

Libraries:

pandas

scikit-learn

imbalanced-learn

Flask

Tools:

Jupyter Notebook

Git and GitHub

Project Structure

telecom-churn-prediction/
├── app.py
├── templates/
│ └── home.html
├── Churn Analysis - EDA.ipynb
├── Churn Analysis - Model Building.ipynb
├── CSV datasets
├── .gitignore
└── README.md

Note: The trained model file (model.sav) is excluded from version control and can be regenerated using the model building notebook.

How to Run the Project
Step 1: Clone the repository

git clone https://github.com/Reethika2024/telecom-churn-prediction.git

cd telecom-churn-prediction

Step 2: Create and activate a virtual environment

python3.11 -m venv venv
source venv/bin/activate

Step 3: Install dependencies

pip install pandas flask scikit-learn imbalanced-learn jupyter

Step 4: Generate the trained model

jupyter notebook

Open Churn Analysis - Model Building.ipynb

Select the kernel: Python (telecom-venv)

Run Kernel → Restart & Run All

This will generate model.sav

Step 5: Run the Flask application

python app.py

Open the browser at:
http://127.0.0.1:5000

Output

The web application predicts:

Whether a customer is likely to churn or not

The confidence score associated with the prediction

Future Enhancements

Add dropdown-based inputs for categorical features

Deploy the application on a cloud platform

Experiment with gradient boosting models

Add automated input validation

Authors

Reethika Peddineni
Vikranth Reddy
Machine Learning & Data Science

Notes

This project demonstrates the complete machine learning lifecycle, including data preprocessing, handling class imbalance, model deployment, and version control best practices.
