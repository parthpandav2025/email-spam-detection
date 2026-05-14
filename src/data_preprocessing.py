import pandas as pd
from utils.helper import clean_text

def load_and_clean_data(path):

    df = pd.read_csv(path ,usecols=[0,1])

    df["clean_text"] = df["Message"].apply(clean_text)

    #convert label into numeric
    df["label_num"] = df["Label"].map({"ham" : 0 , "spam" : 1})

    return df
