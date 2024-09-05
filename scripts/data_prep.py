import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import joblib

def prepare_data():
    file_path = 'data/iris.csv'
    df = pd.read_csv(file_path)

    feature_names = [col for col in df.columns if col != 'Species']
    target_name = 'Species'

    X = df[feature_names]
    y = df[target_name]
    
    y = pd.Categorical(y).codes
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    joblib.dump((X_train, y_train), 'data/X_train.pkl')
    joblib.dump((X_test, y_test), 'data/X_test.pkl')
    print("Data preparation complete and data saved to disk.")

if __name__ == "__main__":
    prepare_data()