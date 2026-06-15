# Income Classification using Apache Spark MLlib
## Overview
This project uses Apache Spark MLlib to build machine learning models capable of predicting whether an individual earns more than $50,000 per year based on demographic and employment information from the U.S. Census Income dataset.

The project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, model evaluation, and interpretation of results using distributed machine learning with Spark.

## Problem Statement
The goal is to classify individuals into one of two income categories:
* <= $50K
* .>$50K
using demographic and employment-related features such as age, education level, occupation, marital status, working hours, and capital gains.

# Technologies Used
- Python
- Apache Spark
- PySpark
- Spark MLlib
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

# The following visualizations were performed:
* Income Class Distribution
* Education Level vs Income
* Age Distribution by Income Class
* Correlation Heatmap
* Model Comparison Charts
* Feature Importance Analysis
* Confusion Matrices
These visualizations helped identify patterns and relationships within the data before model training.

# Data Preprocessing
The following preprocessing steps were applied:
- Loaded the dataset into Spark DataFrames.
- Converted categorical variables into numerical indices using StringIndexer.
- Combined all predictor variables using VectorAssembler.
- Split the dataset into training and testing sets (80/20).
- Prepared the data for distributed machine learning using Spark MLlib.

# Machine Learning Models
## Decision Tree Classifier
A Decision Tree model was trained to learn decision rules from the available features and predict income categories.
## Random Forest Classifier
A Random Forest model consisting of multiple decision trees was trained to improve prediction performance and reduce overfitting.

# Model Evaluation
The models were evaluated using:
- Accuracy Score
- Classification Report
- Precision
- Recall
- F1 Score
- Confusion Matrix

# Feature Importance
Feature importance analysis revealed that the most influential variables included:
- Age
- Education Years
- Capital Gain
- Hours Worked Per Week
- Marital Status
These features contributed most strongly to predicting an individual's income category.

# Key Findings
- Random Forest outperformed the Decision Tree classifier.
- Individuals with higher education levels were more likely to earn above $50K.
- Capital gains and working hours showed strong relationships with income level.
- Distributed machine learning with Apache Spark effectively handled the large dataset.
- Feature importance analysis provided insight into the factors influencing income classification.
