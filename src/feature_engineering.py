from sklearn.feature_extraction.text import  TfidfVectorizer

def create_features(df):

    Tv =  TfidfVectorizer()

    X = Tv.fit_transform(df["clean_text"])

    y = df["label_num"]

    return X , y , Tv

