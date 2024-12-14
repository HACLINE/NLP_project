from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']
sentences1=['I am stupid', 'You are human']
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    model_output1 = model(**encoded_input1)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
# Compute cosine similarities
cosine_scores = F.cosine_similarity(sentence_embeddings, sentence_embeddings1)
print("Cosine scores:", cosine_scores)