import pandas as pd
import re
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import train_test_split

# Deep Learning Frameworks
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

# 1. Download Arabic NLP Tools
nltk.download('stopwords')
st = ISRIStemmer()
arabic_stopwords = set(stopwords.words('arabic'))

# 2. Text Cleaning Function (Handles Tashkeel, Normalization, and Stemming)
def clean_text(text):
    text = str(text)
    # Remove Tashkeel and Tatweel
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'ـ+', '', text)
    # Normalize Alef, Yaa, and Hamza
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # Arabic characters only
    # Tokenize and Stem
    words = text.split()
    return " ".join([st.stem(w) for w in words if w not in arabic_stopwords])

def train_deep_learning_model():
    # 3. Load Data
    train_files = {
        'negative': 'train_Arabic_tweets_negative_20190413.tsv',
        'positive': 'train_Arabic_tweets_positive_20190413.tsv'
    }

    dfs = []
    for label, file_name in train_files.items():
        df = pd.read_csv(file_name, sep='\t', header=None, names=['label_raw', 'text'])
        df['label'] = label 
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).dropna(subset=['text'])
    data = data.sample(frac=1).reset_index(drop=True)

    print("Cleaning data...")
    data["text_clean"] = data['text'].apply(clean_text)

    # 4. Tokenization (Turning text into sequences)
    max_words = 20000 
    max_len = 50 
    tokenizer = Tokenizer(num_words=max_words, lower=False)
    tokenizer.fit_on_texts(data['text_clean'])

    X = tokenizer.texts_to_sequences(data['text_clean'])
    X = pad_sequences(X, maxlen=max_len)

    # Convert labels to binary (0 and 1)
    data['label_num'] = data['label'].map({'negative': 0, 'positive': 1})
    y = data['label_num'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Build LSTM Architecture
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 6. Training
    print("Starting training...")
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

    # 7. Save Files for GitHub/Deployment
    model.save("arabic_model.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    print("✅ Process complete. Model and Tokenizer saved!")

if __name__ == "__main__":
    train_deep_learning_model()
