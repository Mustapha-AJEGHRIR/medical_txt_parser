# %% [markdown]
# # TODO

# %% [markdown]
#  - Embedding for all the lines of the document
#  <!-- - Embeddings for all concepts -->
#  <!-- - Each concept has a list of neighboring concepts based on similarity (e.g. cosine similarity) -->
#  <!-- - The searched term will be embedded and compared to all concepts -->
#  - The searched term will be embedded and compared to all lines of the corpus (with hashing to accelerate)
#  <!-- - Return patients having the neighboring concepts of the searched term -->
#  - Return patients that have big similarity

# %%
# %pip install -U sentence-transformers -q

# %% [markdown]
# ### Importing

# %%
# ----------------------------------- tech ----------------------------------- #
import os
import glob
import pickle

# ---------------------------- Display and friends --------------------------- #
from tqdm import tqdm
from matplotlib import pyplot as plt

# ------------------------- Transformers and freinds ------------------------- #
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import numpy as np

# ------------------------ Classification and friends ------------------------ #
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

# ----------------------------------- local ---------------------------------- #
from data_preprocessing import Get_and_process_data


# %% [markdown]
# ### Configurations

# %%
lines_per_tokenization = 5
filename_split_key = "__at__"
# Load model from HuggingFace Hub
device = "cuda"
model_checkpoint = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# model_checkpoint = "gsarti/scibert-nli"
# model_checkpoint = "logs/scibert_20_epochs_64_batch_99_train_split"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
data_path = os.join(os.dirname(__file__), "../../data/train/txt")
embeddings_path = data_path + os.sep + "embeddings"
similarity = torch.nn.CosineSimilarity()
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)

# %% [markdown]
# ### utils

# %%
#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts, tokenizer = tokenizer, model= model):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings

def find_cluster(query_emb, clustered_data, similarity=similarity):
    best_cluster = None
    best_score = -1
    for i in clustered_data.keys():
        center = clustered_data[i]["center"]
        score = similarity(query_emb, center)
        if score >= best_score:
            best_cluster = i
            best_score = score
    return best_cluster

def text_splitter(text, lines_per_tokenization=lines_per_tokenization):
    lines = text.split("\n")
    
    texts = []
    for i in range(len(lines)//lines_per_tokenization):
        texts.append("\n".join(lines[i*lines_per_tokenization:(i+1)*lines_per_tokenization]))
        
    return texts

def semantic_search_base(query_emb, doc_emb, docs):
    #Compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    print(doc_score_pairs)
    #Output passages & scores
    for doc, score in doc_score_pairs:
        print("==> ",score) 
        print(doc)
        
def forward(texts, tokenizer= tokenizer, model= model):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


def forward_doc(texts, tokenizer= tokenizer, model= model, no_grad= False):
    texts = text_splitter(texts) 
    
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    if no_grad:
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)
    else :
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # NOTE: This is an easy approach
    # another mean pooling over the lines of the document
    # embeddings = torch.mean(embeddings_lines, 0).unsqueeze(0)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings,  p=2, dim=1)
    
    return embeddings


# %% [markdown]
# ### Saving embeddings

# %%
# what are the elements in the folder ../data/train/txt/
all_docs = {}
text_files = glob.glob(data_path + os.sep +  "*.txt")
for file in tqdm(text_files, "Encoding documents", ascii=True):
    with open(file) as f:
        doc = f.read()
    file_name = os.path.basename(file).split(".")[0]
    embeddings = forward_doc(doc, no_grad=True)
    for i,emb in enumerate(embeddings):
        all_docs[file_name+filename_split_key+str(i)] = emb.unsqueeze(0)

# %%
with open(embeddings_path + os.sep + "all_docs.pkl", "wb") as f:
    pickle.dump(all_docs, f)

# %%
# with open(embeddings_path + os.sep + "all_docs.pkl", "rb") as f:
#     all_docs = pickle.load(f)

# %% [markdown]
# ### Classify the embeddings

# %% [markdown]
# We can use hierachical clustering to classify the embeddings for a very search efficient task. But for simplicity, we will only perform K-means clustering.

# %%
sample_names_list = list(map(lambda x: x[0], all_docs.items()))[:]
sample_values_list = list(map(lambda x: x[1], all_docs.items()))[:]
sample = np.array(list(map(lambda x: x.numpy().reshape(-1), sample_values_list))) # array of 1 dim vectors
sample.shape

# %% [markdown]
# #### K-means clustering

# %%
clustering = KMeans(n_clusters = 10).fit(sample)


# %%
clustered_data = {}
for i,center in enumerate(clustering.cluster_centers_):
    clustered_data[i] = {"center": torch.tensor(center.reshape(1, -1)), "elements": {}}

for i, cluster in enumerate(clustering.labels_):
    clustered_data[cluster]["elements"][sample_names_list[i]] = all_docs[sample_names_list[i]]

# %%
with open(embeddings_path + os.sep + "clustered_data.pkl", "wb") as f:
    pickle.dump(clustered_data, f)
