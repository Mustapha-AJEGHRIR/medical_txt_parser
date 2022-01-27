import config
from config import similarity
from .utils import mean_pooling, encode, find_cluster, text_splitter, semantic_search_base, forward, forward_doc
import pickle
import random
import os
import json
from annoy import AnnoyIndex

class Buffer_best_k:
    def __init__(self, k, initia_value=-float("inf")):
        self.k = k
        self.values = [initia_value] * self.k
        self.data = [None] * self.k

    def new_val(self, value, data=None):
        for i in range(self.k):
            if self.values[i] < value:
                self.values[i + 1 :] = self.values[i:-1]
                self.data[i + 1 :] = self.data[i:-1]
                self.values[i] = value
                self.data[i] = data
                return True
        return False

    def get_data(self):
        return self.data

    def get_values(self):
        return self.values

# # ---------------------------------- Kmeans ---------------------------------- #
# with open(config.embeddings_path + os.sep + "clustered_data_concepts.pkl", "rb") as f:
#     clustered_data = pickle.load(f)

# ----------------------------------- Annoy ---------------------------------- #
with open(config.embeddings_path + os.sep + "index_to_name.pkl", "rb") as f:
    sample_names_list = pickle.load(f)
search_index = AnnoyIndex(config.embedding_size, config.annoy_metric)
search_index.load(config.embeddings_path + os.sep + "annoy_index_concepts.ann")


# --------------------------------- Functions -------------------------------- #
def parse_metadata(filename):
    with open(config.metadata_path + os.sep + filename + ".json") as f:
        metadata = json.load(f)
    print("metadata", metadata)
    if metadata["age"] != None:
        metadata["age"] = int(metadata["age"])
    return metadata


def search_query(query, filters={}, top=30):
    # encore query
    query_emb = encode(query)
    
    
    # ---------------------------------- Kmeans ---------------------------------- #
    # # find cluster of docs it belongs in
    # cluster = find_cluster(query_emb, clustered_data)

    # buffer = Buffer_best_k(k=top)
    # for name, doc_emb in clustered_data[cluster]["elements"].items():
    #     score = similarity(query_emb, doc_emb)
    #     # print(name, "\t{:.2f}".format(float(score)))
    #     buffer.new_val(score, name)

    # scores, data_names = buffer.get_values(), buffer.get_data()
    
    # ----------------------------------- Annoy ---------------------------------- #
    indeces, scores = search_index.get_nns_by_vector(query_emb.numpy().reshape(-1), top, include_distances=True)
    data_names = [sample_names_list[i] for i in indeces]
    
    
    results = []
    for i, name in enumerate(data_names):
        filename, paragraph = name.split(config.filename_split_key)
        paragraph = int(paragraph)
        with open(config.data_path + os.sep + filename + ".txt") as f:
            text = f.read()
        file_path = config.data_path + os.sep + filename + ".txt"
        results.append(
            {
                "score": float(scores[i]),
                "filename": filename,
                "id": name,
                "preview": text_splitter(text, file_path)[paragraph],
                "metadata": parse_metadata(filename),
            }
        )

    # TODO: need a better filtering
    # filter results
    range_filters = ["age", "birthdate", "admission_date", "discharge_date"]
    multiselect_filters = ["sexe"]
    filtered_results = []
    for result in results:
        valid = True
        for key in range_filters:
            if key in filters and result["metadata"][key] != None:
                if filters[key][0] > result["metadata"][key] or filters[key][1] < result["metadata"][key]:
                    valid = False
                    print("filtered", result["metadata"][key], filters[key])
                    break
        if valid:
            for key in multiselect_filters:
                if key in filters:
                    if result["metadata"][key] not in filters[key]:
                        valid = False
                        print("filtered", result["metadata"][key], filters[key])
                        break
        if valid:
            filtered_results.append(result)

    count_filtered = len(filtered_results)
    filtered_results = filtered_results[:top]
    return filtered_results, count_filtered


if __name__ == "__main__":
    query = "What is the best way to train a neural network?"
    print(search_query(query))
