
# Azure Machine Learning Engineer

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
This undertaking serves as the culmination project for the "Machine Learning Engineer for Microsoft Azure" Udacity Nanodegree. The primary objective involves selecting a publicly available external dataset, which will then be employed to train a model using both Automated ML and Hyperdrive methodologies. Subsequently, a comparative analysis of the performance between these two algorithms will be conducted, and the superior model will be deployed. The resulting endpoint will be utilized to extract predictive insights through inquiries.

## Dataset

### Overview

The dataset utilized for this project is [heart_failure_clinical_records_dataset.csv](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). It provides information on various health indicators recorded from patients, comprising nearly 300 rows of data.

### Task

The primary task involves predicting the `DEATH_EVENT`—indicating whether or not a patient deceased during the follow-up period (boolean). The dataset features include:

- `age`: The age of the patient.
- `anaemia`: Decrease of red blood cells or hemoglobin (boolean).
- `creatinine_phosphokinase`: Level of creatinine phosphokinase in the blood (mcg/L).
- `diabetes`: If the patient has diabetes (boolean).
- `ejection_fraction`: Percentage of blood leaving the heart at each contraction (percentage).
- `high_blood_pressure`: If the patient has hypertension (boolean).
- `platelets`: Platelets in the blood (kiloplatelets/mL).
- `serum_creatinine`: Level of serum creatinine in the blood (mg/dL).
- `serum_sodium`: Level of serum sodium in the blood (mEq/L).
- `sex`: Woman or man (binary).
- `smoking`: If the patient smokes or not (boolean).
- `time`: Follow-up period (days).
- `DEATH_EVENT`: If the patient deceased during the follow-up period (boolean).

### Access

The dataset was uploaded to the Azure ML studio from a local file, which is also available in this GitHub repository as [heart_failure_clinical_records_dataset.csv](heart_failure_clinical_records_dataset.csv). Both the `automl.ipynb` and `hyperparameter_tuning.ipynb` notebooks contain code that checks whether the .csv file has been uploaded; if not, the code retrieves the dataset from this repository.


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
