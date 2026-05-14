from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

def evaluate(model , X_test , y_test):

    y_pred = model.predict(X_test)

    print(f"Accuracy Score Is : {accuracy_score(y_test , y_pred)}")

    print(f"Classification Report is : \n {classification_report(y_test , y_pred)}")

    print(f"Confusion Matrix is : \n {confusion_matrix(y_test , y_pred)}")
    