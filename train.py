from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from sklearn.metrics import roc_auc_score


seed=42


def get_data():
    """download, clean and split the data"""
    
    
    x_df = pd.read_csv("attrition-dataset.csv")
    y_df = x_df.pop("Attrition")
    x_df = x_df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis = 1) 
    x_df = pd.get_dummies(x_df, drop_first=True)
    
    return train_test_split(x_df,y_df,test_size=0.2, random_state=seed)



def main(args,run):
    """The trainer"""

    run.log("Attribute selection measure:", args.criterion)
    run.log("Bootstrap Strategy:", args.bootstrap)
    run.log("Maximum Depth of a Tree:", args.max_depth)

    model = RandomForestClassifier(criterion=args.criterion,
                                   max_depth=args.max_depth,
                                   bootstrap=args.bootstrap,
                                   random_state=seed,
                                   )
    
    x_train,x_test,y_train,y_test=get_data()
    
    model.fit(x_train, y_train)

    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", AUC_weighted)
    
    os.makedirs('./outputs', exist_ok=True)    
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', type=str, default="gini", 
                        help="The function to measure the quality of a split.",
                        choices=["gini","entropy"]
                        )
                        
    parser.add_argument('--bootstrap', type=str, default="True", 
                        help="Whether bootstrap samples are used when building trees.",
                        )
    
    parser.add_argument('--max_depth', type=int, default=None,
                        help="The maximum depth of the tree.")

    amgs = parser.parse_args()
    
    if amgs.max_depth is not None:
        amgs.max_depth+=1

    if amgs.bootstrap=="True":
        amgs.bootstrap=True
        
    else:
        amgs.bootstrap=False
    
    
    main(amgs,Run.get_context())
