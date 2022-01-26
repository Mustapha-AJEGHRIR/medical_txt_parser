from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import os
import torch


neighbors_k = 5
lines_per_tokenization = 5
filename_split_key = "__at__"
# Load model from HuggingFace Hub
device = "cuda"
# model_checkpoint = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
model_checkpoint = "gsarti/scibert-nli"
# model_checkpoint = "logs/scibert_20_epochs_64_batch_99_train_split"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
data_path = os.path.join(os.path.dirname(__file__), "../../data/train/txt")
embeddings_path = data_path + os.sep + "embeddings"
similarity = torch.nn.CosineSimilarity()
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)