from sklearn.tree import RandomForestClassifier
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.metrics import roc_auc_score


def get_data():
    """download, clean and split the data"""
    
    url="https://raw.githubusercontent.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/main/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    data=TabularDatasetFactory.from_delimited_files(path=url)
    
    x_df = data.to_pandas_dataframe().dropna()
    x_df = x_df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis = 1) 
    x_df = pd.get_dummies(x_df, drop_first=True)
    y_df = x_df.pop("Attrition")
    
    return train_test_split(x_df,y_df,test_size=0.2, random_state=42)



def main(args,run):
    """The trainer"""

    run.log("Attribute selection measure:", np.str(args.criterion))
    run.log("Split Strategy:", np.str(args.splitter))
    run.log("Maximum Depth of a Tree:", np.float(args.max_depth))

    model = RandomForestClassifier(criterion=args.criterion,
                                   max_depth=args.max_depth,
                                   bootstrap=args.bootstrap
                                   )
    
    x_train,x_test,y_train,y_test=get_data()
    
    model.fit(x_train, y_train)

    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", np.float(AUC_weighted))
    
    os.makedirs('./outputs', exist_ok=True)    
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', type=str, default="gini", 
                        help="The function to measure the quality of a split.",
                        choices=["gini","entropy"]
                        )
                        
    parser.add_argument('--bootstrap', type=bool, default=True, 
                        help="Whether bootstrap samples are used when building trees.",
                        choices=["best","random"]
                        )
    
    parser.add_argument('--max_depth', type=int, default=None,
                        help="The maximum depth of the tree.")

    amgs = parser.parse_args()
    
    
    main(amgs,Run.get_context())