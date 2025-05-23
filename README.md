# **Heart Disease Survival Prediction Using KNN**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy-informational)
![Machine Learning](https://img.shields.io/badge/Topic-Machine%20Learning-yellowgreen)
![kNN](https://img.shields.io/badge/Algorithm-kNN-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

In this project, I build and evaluate a K-Nearest Neighbors (kNN) model to predict patient survival from heart failure, based on real clinical data. The goal is to explore the predictive power of simple yet meaningful features to support early risk assessment in clinical settings.

## Table of Contents
- [00. Project Overview](#00-project-overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
- [01. Data Overview](#01-data-overview)
- [02. Feature Selection](#02-feature-selection)
- [03. kNN Algorithm](#03-knn-algorithm)
- [04. Model Evaluation (MSE)](#04-model-evaluation-mse)
- [05. Optimal K Selection](#05-optimal-k-selection)
- [06. Results Discussion](#06-results-discussion)
- [07. Limitations & Future Work](#07-limitations--future-work)


## **00. Project Overview**
### Context 
Accurately identifying high-risk patients can support timely interventions, closer monitoring, and more personalized care planning, ultimately aiming to improve outcomes and resource use in clinical settings.

In this project, I implement a K-Nearest Neighbors (kNN) classifier to predict whether a patient died during the follow-up period. I use data from 297 patients, selecting only three clinically relevant features:

- Age
- Creatinine Phosphokinase (CPK) — a marker of cardiac muscle damage
- Ejection Fraction (EF) — a measure of heart pumping efficiency

These features are simple yet well-established indicators of cardiovascular risk in medical literature.

### Actions
- Used the first 250 records as the training set and last 47 records as the test set.
- Built a kNN classifier from scratch using NumPy.
- Computed predictions for K = 1 to 10.
- Evaluated each model using Mean Squared Error (MSE) on the test set.
- Visualized the performance of different K values to determine the optimal setting.

### Results
Lowest MSE achieved at K = X (fill in based on your results)

Larger K values generally improved stability but risked underfitting.

The model showed clear patterns linking patient age and low EF with increased risk of death, validating the medical relevance of the features used.


## **01. Data Overview**
- Dataset: `heart_data.csv`
- Records: 297 patients
- Binary target: `DEATH_EVENT` (0 = survived, 1 = died)
- Train/Test Split: 250 / 47

## **02. Feature Selection**

I selected three predictors based on their interpretability and medical relevance:

| Feature                    | Description                     | Reason for Inclusion                                |
|----------------------------|----------------------------------|-----------------------------------------------------|
| `age`                      | Patient age                     | Age is a well-known mortality risk factor           |
| `creatinine_phosphokinase`| Enzyme indicating muscle damage | Higher levels may reflect heart failure             |
| `ejection_fraction`       | Blood pumped out per beat       | Lower values strongly linked to poor outcomes       |


## **03. kNN Algorithm**
- Implemented manually using Euclidean distance in 3D feature space.
- For each test point:
  - Calculate distance to all training points.
  - Select the K closest neighbors.
  - Predict the majority class (0 or 1) among those neighbors.

## **04. Model Evaluation (MSE)**
I used Mean Squared Error (MSE) as a performance metric:

<img src="./images/mse_formula_black_bg.png" width="300"/>

This gives a penalty for incorrect predictions, and helps identify which K leads to the most accurate predictions on the unseen test set

## **05. Optimal K Selection**

![Figure 1](./images/Figure_1.png)

## **06. Results Discussion**

MSE decreased steadily as K increased from 1 to 5, reaching its lowest point at **K = 9**.  
Beyond this, performance plateaued, with slight fluctuations but no further improvement.

- **Overfitting** was evident with very small K values (e.g., K = 1 and 2), which had the highest MSE (~0.40).
- The **lowest MSE** (≈ 0.08) occurred at **K = 9**, suggesting this is the most stable choice for this dataset.
- The model benefited from the predictive strength of **age** and **ejection fraction (EF)** — both clinically meaningful indicators of heart failure risk.

## **07. Limitations & Future Work**
1. Small dataset size
   
The model was trained and tested on a relatively small dataset of 297 patients, with only 47 records in the test set. This limits the generalizability and reliability of the results.

3. Limited feature set
   
Only three features — age, creatinine phosphokinase (CPK), and ejection fraction (EF) — were used. While clinically meaningful, other important predictors such as serum sodium, time, and platelet count were not included.

5. Binary output without probability
   
The model only outputs a hard class prediction (0 or 1). Confidence scores or probabilities would provide more nuanced and useful clinical insight.



