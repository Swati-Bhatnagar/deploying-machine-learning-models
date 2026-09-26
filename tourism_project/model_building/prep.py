import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np # Import numpy for numeric type checking

def prepare_data(data_path='/content/tourism.csv'): # Corrected data_path
    # Load the dataset
    df = pd.read_csv(data_path)

    # 1. Handle Duplicate Entries
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped {duplicate_rows} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # 2. Identify and Treat Null Values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Handling null values...")
        for column in df.columns:
            if df[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_val = df[column].median()
                    df[column].fillna(median_val, inplace=True)
                    print(f"  Filled nulls in numerical column '{column}' with median: {median_val}")
                else:
                    mode_val = df[column].mode()[0]
                    df[column].fillna(mode_val, inplace=True)
                    print(f"  Filled nulls in categorical column '{column}' with mode: {mode_val}")
    else:
        print("No null values found.")

    # 3. Rectify 'Gender' Column
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female', 'Famale': 'Female'})
        print("Gender column rectified: 'Fe Male' and 'Famale' standardized to 'Female'.")

    # 4. Rectify 'MaritalStatus' Column
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
        print("MaritalStatus column rectified: 'Unmarried' merged with 'Single'.")

    # Drop 'CustomerID' column as it is not needed for model training
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Dropped 'CustomerID' column.")

    # Drop 'Unnamed: 0' if it exists and contains only index-like values
    if 'Unnamed: 0' in df.columns and df['Unnamed: 0'].nunique() == len(df):
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")

    # Define features (X) and target (y)
    X = df.drop('ProdTaken', axis=1) # Features
    y = df['ProdTaken']             # Target variable

    # Handle categorical columns by one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("Categorical columns one-hot encoded.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save the split datasets locally
    X_train.to_csv('Xtrain.csv', index=False)
    X_test.to_csv('Xtest.csv', index=False)
    y_train.to_csv('ytrain.csv', index=False) # Corrected: y_train to ytrain.csv
    y_test.to_csv('ytest.csv', index=False)   # Corrected: y_test to ytest.csv

    print("Data preparation complete. Training and testing sets saved.")

if __name__ == '__main__':
    prepare_data()
