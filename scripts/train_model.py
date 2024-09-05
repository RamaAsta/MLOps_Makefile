from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    X_train, y_train = joblib.load('data/X_train.pkl')

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    
    return model

if __name__ == "__main__":
    train_model()