
# ----------------------------------- tech ----------------------------------- #
import os
import glob
import pickle
from config import *
from .utils import *

# ---------------------------- Display and friends --------------------------- #
from tqdm import tqdm

# ------------------------- Transformers and freinds ------------------------- #
import torch
import numpy as np

# ------------------------ Classification and friends ------------------------ #
from sklearn.cluster import KMeans

def train():
    all_docs = {}
    text_files = glob.glob(data_path + os.sep +  "*.txt")
    for file_path in tqdm(text_files, "Encoding documents", ascii=True):
        with open(file_path) as f:
            doc = f.read()
        file_name = os.path.basename(file_path).split(".")[0]
        embeddings = forward_doc(doc, file_path, no_grad=True)
        for i,emb in enumerate(embeddings):
            all_docs[file_name+filename_split_key+str(i)] = emb.unsqueeze(0)


    # with open(embeddings_path + os.sep + "all_docs_concepts.pkl", "wb") as f:
    #     pickle.dump(all_docs, f)


    sample_names_list = list(map(lambda x: x[0], all_docs.items()))[:]
    sample_values_list = list(map(lambda x: x[1], all_docs.items()))[:]
    sample = np.array(list(map(lambda x: x.numpy().reshape(-1), sample_values_list))) # array of 1 dim vectors
    sample.shape


    clustering = KMeans(n_clusters = 10).fit(sample)



    clustered_data = {}
    for i,center in enumerate(clustering.cluster_centers_):
        clustered_data[i] = {"center": torch.tensor(center.reshape(1, -1)), "elements": {}}

    for i, cluster in enumerate(clustering.labels_):
        clustered_data[cluster]["elements"][sample_names_list[i]] = all_docs[sample_names_list[i]]


    with open(embeddings_path + os.sep + "clustered_data_concepts.pkl", "wb") as f:
        pickle.dump(clustered_data, f)

if __name__ == "__main__":
    train()