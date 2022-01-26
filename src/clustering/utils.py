from config import *
import torch
import torch.nn.functional as F
import numpy as np


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

def text_splitter(text, file_path):
    con_file_path = os.path.dirname(os.path.dirname(file_path)) + os.sep + "concept" + os.sep + os.path.basename(file_path).split(".")[0] + ".con"
    concepts_lines = list(set(parse_concept(con_file_path)["start_line"]))
    concepts_lines.sort()
    texts = text.split("\n")
    concepts = []
    for line in concepts_lines:
        concepts.append(texts[line-1])
    return concepts

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


def forward_doc(text, file_path, tokenizer= tokenizer, model= model, no_grad= False):
    texts = text_splitter(text, file_path) 
    if len(texts) == 0:
        return []
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
