from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features 
from src.model_training import train_model
from src.model_evaluation import evaluate

#for model saving
import pickle

#Load The Dataset
df = load_and_clean_data("notebooks\\sms.csv")

#define X , y and cv(CountVectorizer)
X , y , Tv = create_features(df)

#train the model and get X_test , y_test and model 
model , X_test , y_test = train_model(X,y)

#model evaluation and get Accuracy score , classification report and confusion matrix
evaluate(model , X_test , y_test)

#save a spam model using pickle
with open("models/spam_model.pkl" , "wb") as f:
    pickle.dump(model , f)

##save a CountVectorizer model using pickle
with open("models/TfidfVectorizer.pkl" , "wb") as f:
    pickle.dump(Tv , f)

print("Models saved succesfully....")