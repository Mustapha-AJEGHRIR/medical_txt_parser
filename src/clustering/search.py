import config
from config import similarity
from .utils import mean_pooling, encode, find_cluster, text_splitter, semantic_search_base, forward, forward_doc
import pickle
import random
import os

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


with open(config.embeddings_path + os.sep + "clustered_data_concepts.pkl", "rb") as f:
    clustered_data = pickle.load(f)


def parse_metadata(filename):
    return {
        "age": 30,
        "sexe": f"{random.choice(['F', 'M'])}",
        "birthdate": f"1990-01-01",
        "admission_date": f"1990-01-01",
        "discharge_date": f"1990-01-01",
    }


def search_query(query, filters={}, top=10):
    # encore query
    query_emb = encode(query)
    # find cluster of docs it belongs in
    cluster = find_cluster(query_emb, clustered_data)

    buffer = Buffer_best_k(k=top)
    for name, doc_emb in clustered_data[cluster]["elements"].items():
        score = similarity(query_emb, doc_emb)
        # print(name, "\t{:.2f}".format(float(score)))
        buffer.new_val(score, name)

    scores, data_names = buffer.get_values(), buffer.get_data()

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

    # filter results
    range_filters = ["age", "birthdate", "admission_date", "discharge_date"]
    multiselect_filters = ["sexe"]
    filtered_results = []
    for result in results:
        valid = True
        for key in range_filters:
            if key in filters:
                if filters[key][0] > result["metadata"][key] or filters[key][1] < result["metadata"][key]:
                    valid = False
                    break
        if valid:
            for key in multiselect_filters:
                if key in filters:
                    if result["metadata"][key] not in filters[key]:
                        valid = False
                        break
        if valid:
            filtered_results.append(result)

    count_filtered = len(filtered_results)
    filtered_results = filtered_results[:top]
    return filtered_results, count_filtered


if __name__ == "__main__":
    query = "What is the best way to train a neural network?"
    print(search_query(query))
