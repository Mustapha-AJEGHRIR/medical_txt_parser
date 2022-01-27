import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch


neighbors_k = 5
clusters = 10
trees = 100
annoy_metric = 'dot'
lines_per_tokenization = 5
filename_split_key = "__at__"
# Load model from HuggingFace Hub
device = "cuda"
model_checkpoint = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# model_checkpoint = "gsarti/scibert-nli"
# model_checkpoint = "logs/scibert_20_epochs_64_batch_99_train_split"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
data_path = os.path.join(os.path.dirname(__file__), "../data/train/txt")
embeddings_path = data_path + os.sep + "embedding"
train_data_path = os.path.join(os.path.dirname(__file__), "../data/train")
metadata_path = train_data_path + os.sep + "metadata"
similarity = torch.nn.CosineSimilarity()
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
embedding_size = model.embeddings.word_embeddings.weight.data.shape[1]

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/predictions')