import os
import numpy as np
import torch
import pandas as pd
import gensim.downloader
from tqdm import tqdm
from scipy import sparse
from google.colab import drive
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

drive.mount('/content/drive')

path = '/content/drive/MyDrive/data/'

def get_tfidf(df, max_features):

    file_name = f"tfidf_{max_features}_emb.npz"
    file_path = os.path.join(path, file_name)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return sparse.load_npz(file_path)
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        X_tfidf_sparse = tfidf_vectorizer.fit_transform(df["summary"])
        sparse.save_npz(file_path, X_tfidf_sparse)
        print(f"Saved: {file_path}")
        return X_tfidf_sparse

def get_bow(df, max_features):

    file_name = f"bow_{max_features}_emb.npz"
    file_path = os.path.join(path, file_name)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return sparse.load_npz(file_path)
    else:
        bow_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        X_bow_sparse = bow_vectorizer.fit_transform(df["summary"])
        sparse.save_npz(file_path, X_bow_sparse)
        print(f"Saved: {file_path}")
        return X_bow_sparse

def get_sentence_vector(sentence, model, embedding_layer):
    if not isinstance(sentence, str):
        return np.zeros(model.vector_size, dtype=np.float32)

    word_ids = [
        model.key_to_index[word]
        for word in sentence.split()
        if word in model.key_to_index
    ]

    if not word_ids:
        return np.zeros(model.vector_size, dtype=np.float32)

    word_ids_tensor = torch.LongTensor(word_ids)

    with torch.no_grad():
        word_vectors = embedding_layer(word_ids_tensor)
        sentence_vector = torch.mean(word_vectors, dim=0).numpy()

    return sentence_vector

def get_w2v300(df):
    
    file_name = "w2v_300_emb.npz"
    file_path = os.path.join(path, file_name)


    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        data = np.load(file_path)
        return data["embeddings"]
    else:
        w2v_google = gensim.downloader.load('word2vec-google-news-300')

        weights = torch.FloatTensor(w2v_google.vectors)
        embedding_layer = torch.nn.Embedding.from_pretrained(weights, freeze=True)
        vectors = []

        for text in tqdm(df["summary"].astype(str)):
            vectors.append(
                get_sentence_vector(text, w2v_google, embedding_layer)
            )

        all_w2v_vectors = np.vstack(vectors)

        np.savez_compressed(
            file_path,
            embeddings=all_w2v_vectors
        )

        print(f"Saved: {file_path}")
        return all_w2v_vectors















































        