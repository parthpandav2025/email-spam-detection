from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from utils.helper import clean_text

#load the TfidfVectorizer model for text vectorizer :-

with open("models/TfidfVectorizer.pkl" , "rb") as f:
    Tf_model = pickle.load(f)

#load the spam detection model  : -

with open("models/spam_model.pkl" , "rb") as f:
    spam_model = pickle.load(f)


app = FastAPI()

class Data(BaseModel):
    message : str

@app.get("/")
def home():
    return {
        "message" : "Welcome To The FastApi Dashboard"
    }

@app.post("/")
def predict(data : Data):
    
    
    text = data.message

    text = [clean_text(text)]

    text_vectorizer = Tf_model.transform(text)

    prediction = spam_model.predict(text_vectorizer)
    
    return {
        "text" : text ,
        "prediction" : ["Spam" if i == 1 else "Not_Spam" for i in prediction] 
    }