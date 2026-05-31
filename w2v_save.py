import json
import os
import re
import scipy
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gensim.downloader
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def get_sentence_tensor(sentence, model, max_len=150):
    if not isinstance(sentence, str):
        return torch.zeros(max_len, dtype=torch.long)

    word_ids = [model.key_to_index[word] for word in sentence.split() if word in model.key_to_index]
    word_ids = word_ids[:max_len]

    pad_length = max_len - len(word_ids)
    if pad_length > 0:
        word_ids.extend([0] * pad_length)

    return torch.tensor(word_ids, dtype=torch.long)

def save_w2v(df, output_dir, output_filename, w2v, max_len):
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    vectors = []

    for sentence in tqdm(df["summary"].astype(str)):
        vectors.append(get_sentence_tensor(sentence, w2v, max_len))

    all_w2v_vectors = torch.vstack(vectors)
    # print(all_w2v_vectors.shape)

    np.savez_compressed(
        output_path,
        w2v_idxs=all_w2v_vectors.numpy()
    )

    return all_w2v_vectors

def main(conf_filepath):

    with open(conf_filepath, "r") as conf_file:
        config = json.load(conf_file)
        print(f"Loading configuration from {conf_filepath}")

        df = pd.read_csv(config["data"]["csv_path"])
        df = df.head(config["data"]["sample_size"])

        w2v_model = gensim.downloader.load(config["model"]["pretrained_w2v_name"])
        max_len =config["model"]["max_len"]
        output_dir=config["data"]["output_dir"]
        filename=config["data"]["output_filename"]

        save_w2v(
            df=df, 
            output_dir=output_dir, 
            output_filename =filename,
            w2v=w2v_model,
            max_len= max_len #max len of summary
        )

# from google.colab import drive
# drive.mount('/content/drive')

if __name__ == "__main__":
    main(conf_filepath="w2v_save_json")