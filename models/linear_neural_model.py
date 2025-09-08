import torch
from src.utils import build_dataset, clean_text
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from gensim.models import word2vec
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.tf_idf_glove_model import compute_metrics, get_glove_embeddings, get_top_n_tf_idf
import os

nltk.download('wordnet')
nltk.download('omw-1.4')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


class LinearNeuralModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNeuralModel, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.lin3 = nn.Linear(output_dim, output_dim)
        self.norm3 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin3(out)
        out = self.norm3(out)
        out = self.sigmoid(out)
        return out


def train_model(model, X, y, threshold=0.5, num_epochs=25, batch_size=16, learning_rate=0.001, train_split=0.8):
    train_size = int(train_split * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    train_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(
        X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            batch_y = batch_y.type_as(outputs)
            batch_loss = criterion(outputs, batch_y)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss/len(train_loader):.4f}')
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X).squeeze()
            predicted = (outputs >= threshold).float()
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    y_true = torch.cat([torch.tensor(y_true)], dim=0).cpu().numpy()
    y_pred = torch.cat([torch.tensor(y_pred)], dim=0).cpu().numpy()
    metrics = compute_metrics(y_true, y_pred)
    print(f'Validation Metrics: {metrics}')

    return model, metrics


def get_code_embedding(code_snippet, tokenizer, model):
    inputs = tokenizer(code_snippet, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()


def get_bert_text_embedding(text, tokenizer, model, top_k=5):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    cls_embedding = last_hidden_state[0, 0].unsqueeze(0)
    token_embeddings = last_hidden_state[0, 1:-1]
    # compute cosine similarity between cls_embedding and each token embedding
    cosine_similarities = cosine_similarity(
        cls_embedding.cpu().numpy(), token_embeddings.cpu().numpy())[0]
    top_k_indices = cosine_similarities.argsort()[-top_k:]
    top_k_embeddings = token_embeddings[top_k_indices]
    concatenated_embedding = torch.cat([top_k_embeddings], dim=0)
    return concatenated_embedding.cpu().numpy()


def main():
    # Load dataset
    folder_path = 'data/code_classification_dataset'
    df = build_dataset(folder_path)
    # used tags for multi-tagging
    valid_tags = ['math', 'graphs', 'strings', 'number theory',
                  'trees', 'geometry', 'games', 'probabilities']
    df["tags"] = df["tags"].apply(
        lambda tags: [tag for tag in tags if tag in valid_tags])
    # filtering out rows with no valid tags
    df = df[df["tags"].apply(lambda x: len(x) > 0)]
    df["binary_tags"] = df["tags"].apply(
        lambda x: [1 if tag in x else 0 for tag in valid_tags])
    df['text_input'] = df['prob_desc_description'].fillna(
        "") + " " + df['prob_desc_input_spec'].fillna("")+" " + df['prob_desc_output_spec'].fillna("")
    # cleaning the text data
    df["cleaned_text"] = df["text_input"].apply(clean_text)
    print("processing the cleaned text using bert embeddings...")
    # applying embedding function to the text data using smallbert
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
    model = AutoModel.from_pretrained("prajjwal1/bert-small")
    df["text_embeddings"] = df["cleaned_text"].apply(
        lambda x: get_bert_text_embedding(x, tokenizer, model, top_k=5))
    # applying embedding function to the code snippets
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    df["code_embeddings"] = df["source_code"].apply(
        lambda x: get_code_embedding(x, tokenizer, model))
    # save the dafaframe into a csv
    df.to_csv("data/processed_dataset_with_embeddings.csv", index=False)
    # flattening the embeddings
    X_text = np.array([emb.flatten()
                      for emb in df["text_embeddings"].tolist()])
    X_code = np.array([emb.flatten()
                      for emb in df["code_embeddings"].tolist()])
    X = np.concatenate([X_text, X_code], axis=1)
    print("Feature matrix shape:", X.shape)
    y = np.array(df["binary_tags"].tolist())
    model = LinearNeuralModel(
        input_dim=X.shape[1], output_dim=y.shape[1]).to(DEVICE)
    trained_model, metrics = train_model(
        model, X, y, threshold=0.5, num_epochs=200, batch_size=32, learning_rate=0.001, train_split=0.8)


if __name__ == "__main__":
    main()
