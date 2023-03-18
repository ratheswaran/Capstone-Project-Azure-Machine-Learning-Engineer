import os
import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run


def preprocess_data(data_df):
    """
    Store input features and data labels separately 
    
    Parameters
    -------------
    data_df: pandas dataframe of input features and labels
    
    Return
    -------------
    x_df: pandas dataframe of input features
    y_df: pandas dataframe of input features
    """
    y_df = data_df.pop("Survived")
    x_df = data_df
    
    return x_df, y_df

# Create TabularDataset using TabularDatasetFactory
url_path = "https://raw.githubusercontent.com/ratheswaran/directory_name/main/train_numeric.csv"
dataset = TabularDatasetFactory.from_delimited_files(path=url_path)
data_df = dataset.to_pandas_dataframe()

# Split data to features and labels dataframe
x_df, y_df = preprocess_data(data_df)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model, './outputs/model_hyppar_tunning.joblib')

if __name__ == '__main__':
    main()