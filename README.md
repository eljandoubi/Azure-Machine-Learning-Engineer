# Capstone Project Udacity Machine Learning Engineer Nanodegree

This repository contains files related to the Capstone Project for Udacity's Machine Learning Nanodegree with Microsoft Azure.

In this project, two experiments were conducted: one using Microsoft Azure Machine Learning [Hyperdrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py), and another using Microsoft Azure [Automated Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train?view=azure-ml-py) (referred to as AutoML) with the [Azure Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py).

The best models from both experiments were compared based on the primary metric (AUC weighted score), and the best performing model was deployed and consumed using a web service.

## Project Workflow
![Project Workflow](pics/capstone-diagram.png)

## Dataset
The project utilized the [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset), aiming to predict employee attrition and understand the contributing factors.

More information about the dataset can be found [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).

### Task
This is a binary classification problem predicting 'Attrition' as either 'true' or 'false'. `Hyperdrive` and `AutoML` were used to train models based on the `AUC Weighted` metric. The best-performing model was deployed and interacted with.

### Access
The data is hosted [here](https://raw.githubusercontent.com/eljandoubi/Azure-Machine-Learning-Engineer/main/attrition-dataset.csv). The Tabular Dataset Factory's `Dataset.Tabular.from_delimited_files()` operation was used to import and save it to the datastore by using `dataset.register()`.

## Automated ML
Automated machine learning selects algorithms and hyperparameters, generating a deployable model. Configuration details are as follows:

| Auto ML Configuration | Value | Explanation |
|:---:|:---:|:---:|
| experiment_timeout_minutes | 30 | Maximum duration in minutes before termination |
| max_concurrent_iterations | 8 | Maximum concurrent iterations |
| primary_metric | AUC_weighted | Metric for model optimization |
| compute_target | cpu_cluster(created) | Compute target for the experiment |
| task | classification | Nature of the machine learning task |
| training_data | dataset(imported) | Training data used in the experiment |
| label_column_name | Attrition | Label column name |
| path | ./automl | Project folder path |
| enable_early_stopping | True | Enable early termination |
| featurization | auto | Automatic featurization |
| debug_log | automl_errors.log | Debug log file |

### Results
The best model, `VotingEnsemble`, achieved an AUC_weighted of **0.840**.

![Best AutoML Model](pics/auto_best_model.png)

**Run Details**
![AutoML Run Completed](pics/auto_run_completed.png)
![AutoML Run Details](pics/auto_run_details.png)
![AutoML Run Table](pics/auto_run_table.png)


### Improve AutoML Results
* Increase experiment timeout duration
* Try a different primary metric
* Engineer new features
* Explore other AutoML configurations

## Hyperparameter Tuning
A Decision Tree model was used for its simplicity and interpretability. HyperDrive configuration details:

| Configuration | Value | Explanation |
|:---:|:---:|:---:|
| hyperparameter_sampling | Value | Explanation |
| policy | early_termination_policy | Early termination policy |
| primary_metric_name | AUC_weighted | Primary metric for evaluation |
| primary_metric_goal | PrimaryMetricGoal.MAXIMIZE | Maximize primary metric |
| max_total_runs | 8 | Maximum number of runs |
| max_concurrent_runs | 4 | Maximum concurrent runs |
| run_config | ScriptRunConfig | configuration to run th script |

Hyperparameters for the Decision Tree:

| Hyperparameter | Value | Explanation |
|:---:|:---:|:---:|
| criterion | choice("gini", "entropy") | Function to measure split quality |
| bootstrap | choice(True, False) | Use of bootstrap samples |
| max_depth | randint(10) | Maximum depth of the tree |

### HyperDrive Results
The best model had Parameter Values as `criterion` = **gini**, `max_depth` = **8**, `bootstrap` = **False**. The AUC_weighted of the Best Run is **0.744**.

![Best HyperDrive Model](pics/hyperdrive_best_model.png)

**Run Details**
![HyperDrive Run Completed](pics/hyperdrive_run_completed.png)
![HyperDrive Run Details](pics/hyperdrive_run_details.png)
![HyperDrive Run Table](pics/hyperdrive_run_table.png)

**Visualization of Runs**
![HyperDrive Output](pics/hyperdrive_visu.png)

### Improve HyperDrive Results
* Choose a different algorithm
* Choose a different classification metric
* Choose a different termination policy
* Specify a different sampling method

## Model Deployment
The AutoML model outperforms the HyperDrive model, so it will be deployed as a web service. The workflow for deploying a model in Azure ML Studio is as follows:

* **Register the model**
* **Prepare an inference configuration**
* **Prepare an entry script**
* **Choose a compute target**
* **Deploy the model to the compute target**
* **Test the resulting web service**

**Healthy Deployed State**
![AutoML Webservice stat](pics/deploy_stat.png)
![AutoML Webservice ins](pics/deploy_ins.png)

## Screen Recording
An overview of this project can be found [here](https://youtu.be/nRRIczcHjpM).  
