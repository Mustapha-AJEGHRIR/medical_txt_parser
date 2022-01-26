from config import *
from utils import *
import pickle


class Buffer_best_k:
    def __init__(self, k, initia_value=-float("inf")):
        self.k = k
        self.values = [initia_value] * self.k
        self.data = [None] * self.k
    def new_val(self, value, data=None):
        for i in range(self.k):
            if self.values[i] < value:
                self.values[i+1:] = self.values[i:-1]
                self.data[i+1:] = self.data[i:-1]
                self.values[i] = value
                self.data[i] = data
                return True
        return False
    def get_data(self):
        return self.data
    def get_values(self):
        return self.values               




with open(embeddings_path + os.sep + "clustered_data.pkl", "rb") as f:
    clustered_data = pickle.load(f)
    
    
def search(query, k=neighbors_k):
    query_emb = encode(query)
    cluster = find_cluster(query_emb, clustered_data)
    
    buffer = Buffer_best_k(k=k)
    
    for name, doc_emb in clustered_data[cluster]["elements"].items():
        score = similarity(query_emb, doc_emb)
        # print(name, "\t{:.2f}".format(float(score)))
        buffer.new_val(score, name)
    
    scores, data_names = buffer.get_values(), buffer.get_data()
    
    k_data = []
    for i,name in enumerate(data_names):
        filename, paragraph = name.split(filename_split_key)
        paragraph = int(paragraph)
        with open(data_path + os.sep + filename + ".txt") as f:
            text = f.read()
            
        data = {"score" : scores[i],
                "filename" : filename,
                "paragraph" : paragraph,
                "full-text" : text,
                "paragraph-text" : text_splitter(text)[paragraph]}
        k_data.append(data)
    
    return k_data



if __name__ == "__main__":
    query = "What is the best way to train a neural network?"
    print(search(query))
    