### a code to predict using the saved models (logistic regression and neural network model)###
# using argument giving in the command line interface
import torch
import joblib
import time
import nltk
from src.utils import clean_text
from models.tf_idf_glove_model import get_glove_embeddings
from models.linear_neural_model import get_bert_text_embedding, get_code_embedding, LinearNeuralModel
from gensim.models import word2vec
from transformers import AutoTokenizer, AutoModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    valid_tags = ['math', 'graphs', 'strings', 'number theory',
                  'trees', 'geometry', 'games', 'probabilities']
    print("Inference script, do you wanna use logistic regression or neural network model, or a soft vote model combining the two?(lr,nn,sv)")
    model_type = input().strip().lower()
    if model_type not in ['lr', 'nn', 'sv']:
        print("Invalid input. Please enter 'lr', 'nn', or 'sv'.")
        return
    print("Please enter the problem description:")
    problem_description = input().strip()
    print("Please enter the problem input specification:")
    problem_input_spec = input().strip()
    print("Please enter the problem output specification:")
    problem_output_spec = input().strip()
    text_input = problem_description + " " + \
        problem_input_spec + " " + problem_output_spec
    cleaned_text = clean_text(text_input)
    if model_type in ['nn', 'sv']:
        print("Please enter the source code snippet:")
        source_code = input().strip()
    # print inference time
    start_time = time.time()
    # load the models
    if model_type in ['lr', 'sv']:
        lemmatizer = nltk.WordNetLemmatizer()
        cleaned_text_lemmatized = ' '.join([lemmatizer.lemmatize(word)
                                            for word in cleaned_text.split()])
        # load the saved models
        logistic_models = joblib.load('models/logistic_models.pkl')
        best_thresholds_lr = np.loadtxt(
            'models/best_thresholds_lr.csv', delimiter=',')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        glove_model = word2vec.Word2Vec.load('models/glove_word2vec.model')
        # get the tf-idf features
        X_tfidf = vectorizer.transform([cleaned_text_lemmatized]).toarray()
        # get the top 5 words
        top_keywords = [word for word, score in zip(vectorizer.get_feature_names_out(),
                                                    X_tfidf[0]) if score > 0]
        top_keywords = sorted(top_keywords, key=lambda w: -
                              X_tfidf[0, vectorizer.vocabulary_[w]])[:5]
        # get the glove embeddings
        X_glove = get_glove_embeddings(top_keywords, glove_model).flatten()
        X_lr = X_glove.reshape(1, -1)
        # inference using logistic regression models
        probs_lr = np.array([model.predict_proba(X_lr)[0][1]
                             for model in logistic_models.values()])
        preds_lr = (probs_lr >= best_thresholds_lr).astype(int)
    if model_type in ['nn', 'sv']:
        nn_model_dict = torch.load(
            'models/linear_neural_model.pth', map_location=device)
        nn_model = LinearNeuralModel(input_dim=3328, output_dim=8).to(device)
        nn_model.load_state_dict(nn_model_dict)
        best_thresholds_nn = np.loadtxt(
            'models/best_thresholds_nn.csv', delimiter=',')
        tokenizer_text = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        model_text = AutoModel.from_pretrained("prajjwal1/bert-small")
        tokenizer_code = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base")
        model_code = AutoModel.from_pretrained("microsoft/codebert-base")
        X_text = get_bert_text_embedding(
            cleaned_text, tokenizer_text, model_text, top_k=5)
        X_code = get_code_embedding(source_code, tokenizer_code, model_code)
        X_nn = np.concatenate(
            [X_text.flatten(), X_code.flatten()]).reshape(1, -1)
        # inference using neural network model
        nn_model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_nn, dtype=torch.float32).to(device)
            outputs = nn_model(inputs)
            probs_nn = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds_nn = (probs_nn >= best_thresholds_nn).astype(int)
    if model_type == 'lr':
        print("Predicted tags (Logistic Regression):")
        predicted_tags = [valid_tags[i] for i in range(
            len(valid_tags)) if preds_lr[i] == 1]
        print(predicted_tags)
    elif model_type == 'nn':
        print("Predicted tags (Neural Network):")
        predicted_tags = [valid_tags[i] for i in range(
            len(valid_tags)) if preds_nn[i] == 1]
        print(predicted_tags)
    else:  # soft voting
        final_preds = ((probs_lr + probs_nn) >= 1).astype(int)
        print("Predicted tags (Soft Voting):")
        predicted_tags = [valid_tags[i] for i in range(
            len(valid_tags)) if final_preds[i] == 1]
        print(predicted_tags)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
