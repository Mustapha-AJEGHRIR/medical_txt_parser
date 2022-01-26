
# ----------------------------------- tech ----------------------------------- #
import os
import glob
import pickle
from config import *
from utils import *

# ---------------------------- Display and friends --------------------------- #
from tqdm import tqdm
from matplotlib import pyplot as plt

# ------------------------- Transformers and freinds ------------------------- #
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
# from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import numpy as np

# ------------------------ Classification and friends ------------------------ #
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE


if __name__ == "__main__":
    
    all_docs = {}
    text_files = glob.glob(data_path + os.sep +  "*.txt")
    for file in tqdm(text_files, "Encoding documents", ascii=True):
        with open(file) as f:
            doc = f.read()
        file_name = os.path.basename(file).split(".")[0]
        embeddings = forward_doc(doc, no_grad=True)
        for i,emb in enumerate(embeddings):
            all_docs[file_name+filename_split_key+str(i)] = emb.unsqueeze(0)


    with open(embeddings_path + os.sep + "all_docs.pkl", "wb") as f:
        pickle.dump(all_docs, f)


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


    with open(embeddings_path + os.sep + "clustered_data.pkl", "wb") as f:
        pickle.dump(clustered_data, f)