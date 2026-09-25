import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data_path='tourism_project/data/tourism.csv'):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop 'CustomerID' column as it is not needed for model training
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Define features (X) and target (y)
    X = df.drop('ProdTaken', axis=1) # Features
    y = df['ProdTaken']             # Target variable

    # Handle categorical columns by one-hot encoding for the splitting process if needed
    # For simplicity, we'll assume string columns are categorical and encode them.
    # This should ideally be part of a preprocessing pipeline in train.py but for splitting
    # and saving, it's often easier to have numeric data.
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save the split datasets locally
    X_train.to_csv('Xtrain.csv', index=False)
    X_test.to_csv('Xtest.csv', index=False)
    y_train.to_csv('ytrain.csv', index=False)
    y_test.to_csv('ytest.csv', index=False)

    print("Data preparation complete. Training and testing sets saved.")

if __name__ == '__main__':
    prepare_data()
