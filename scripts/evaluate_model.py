from sklearn.metrics import classification_report
import joblib
from train_model import train_model

def evaluate_model():
    model = train_model()
    X_test, y_test = joblib.load('data/X_test.pkl')
    y_pred = model.predict(X_test)
    

    report = classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
    
    print("Model Evaluation Report:")
    print(report)
    

if __name__ == "__main__":
    evaluate_model()