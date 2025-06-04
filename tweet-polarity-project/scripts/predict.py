
import pandas as pd
import pickle
from utils.text_cleaning import clean_text
from utils.date_utils import convert_to_datetime

# Load model
with open("../models/tweet_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data
df = pd.read_csv("../data/test_tweets.csv")

# Preprocessing
df["text"] = df["text"].fillna("").apply(clean_text)
df["created_at"] = convert_to_datetime(df["created_at"])

# Predict
X_test = df["text"]
predictions = model.predict(X_test)

# Save results
df["predicted_polarity"] = predictions
df.to_csv("../data/predicted_tweets.csv", index=False)
print("Prediction complete. File saved.")
