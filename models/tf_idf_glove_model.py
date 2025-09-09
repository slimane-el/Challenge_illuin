from src.utils import build_dataset, clean_text
import joblib
import os
import optuna
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, hamming_loss, f1_score
from sklearn.model_selection import StratifiedKFold

# import os to modify path if needed because utils is in ../src


nltk.download('wordnet')
nltk.download('omw-1.4')


def get_top_n_tf_idf(corpus, n=5):
    """
    Computes the TF-IDF representation of the given corpus.

    Args:
        corpus (list): A list of documents (strings).

    Returns:
        tuple: A tuple containing the TF-IDF matrix and the top n keywords for each document.
    """
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    top_n_keywords = []
    for i in range(tf_idf_matrix.shape[0]):
        row = tf_idf_matrix.getrow(i)
        sorted_indices = np.argsort(row.toarray()).flatten()[::-1]
        top_n = [feature_names[idx] for idx in sorted_indices[:n]]
        top_n_keywords.append(top_n)
    return vectorizer, top_n_keywords


def get_glove_embeddings(words, model):
    """
    Retrieves GloVe embeddings for the given words.

    Args:
        words (list): A list of words.
        model (gensim.models.Word2Vec): A pre-trained Word2Vec model.

    Returns:
        np.ndarray: An array of GloVe embeddings for the words.
    """
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)


def compute_metrics(true_labels, predictions):
    """
    Computes precision, recall, hamming loss, and F1-score for multi-label classification.

    Args:
        true_labels (np.ndarray): The true binary labels.
        predictions (np.ndarray): The predicted binary labels.

    Returns:
        dict: A dictionary containing precision, recall, and F1-score per class, and hamming loss.
    """
    precision = precision_score(
        true_labels, predictions, average='micro', zero_division=0)
    recall = recall_score(true_labels, predictions,
                          average='micro', zero_division=0)
    hamming = hamming_loss(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'hamming_loss': hamming
    }


def logistic_regression_by_label(X, y, threshold, X_train, X_test, y_train, y_test):
    """
    Trains a logistic regression model for each unique label in the dataset using a threshold approach.

    Args:
        X (np.ndarray): The feature matrix.
        y (pd.Series): The target labels.

    Returns:
        dict: A dictionary mapping each label to its corresponding trained logistic regression model.
    """
    num_classes = y.shape[1]
    models = {}
    all_predictions = []
    all_true_labels = []
    for i in range(num_classes):
        model = LogisticRegression()
        model.fit(X_train, y_train[:, i])
        models[i] = model
        proba = model.predict_proba(X_test)[:, 1]
        predictions = (proba >= threshold[i]).astype(int)
        all_predictions.append(predictions)
        all_true_labels.append(y_test[:, i])
    return models, all_predictions, all_true_labels


def objective(trial, X, y):
    # Suggest one threshold per label (bounded)
    thresholds = np.array([
        trial.suggest_float(f"thr_{i}", 0.3, 0.7)
        for i in range(8)
    ], dtype=float)
    # skf
    skf = StratifiedKFold(n_splits=5)
    fold_metrics = []
    for train_index, test_index in skf.split(X, y.argmax(axis=1)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        models, preds, true_lbls = logistic_regression_by_label(
            X_train, y_train, threshold=thresholds, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        metrics = compute_metrics(true_lbls, preds)
        fold_metrics.append(metrics)
    # Average across folds
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
        'hamming_loss': np.mean([m['hamming_loss'] for m in fold_metrics]),
    }
    return avg_metrics['hamming_loss']


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
    # lemmatizing the text data
    corpus = df["cleaned_text"].tolist()
    sentences = [text.split() for text in corpus]
    lemmatizer = nltk.WordNetLemmatizer()
    sentences = [[lemmatizer.lemmatize(word) for word in sentence]
                 for sentence in sentences]
    model = word2vec.Word2Vec(
        sentences, vector_size=25, window=5, min_count=1, sg=1, epochs=10)
    # TF-IDF and GloVe embeddings
    vectorizer, top_keywords = get_top_n_tf_idf(corpus, n=5)
    df["keywords"] = top_keywords
    df["embeddings"] = df["keywords"].apply(
        lambda words: get_glove_embeddings(words, model))
    # flattening the embeddings
    X = np.array([emb.flatten() for emb in df["embeddings"]])
    print("Feature matrix shape:", X.shape)
    y = np.array(df["binary_tags"].tolist())
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))
    # Cross-validation and model evaluation
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(
        trial, X_train, y_train), n_trials=200)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    best_thresholds = np.array([trial.params[f"thr_{i}"] for i in range(8)])
    print("Best thresholds:", best_thresholds)
    # print final metrics on the test set
    models, preds, true_lbls = logistic_regression_by_label(
        X, y, threshold=best_thresholds, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    final_metrics = compute_metrics(true_lbls, preds)
    print("Final metrics on the test set:", final_metrics)
    # save the word2vec model and vectorizer model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/glove_word2vec.model')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    # train final models on theentire dataset
    logistic_models, preds, true_lbls = logistic_regression_by_label(
        X, y, threshold=best_thresholds, X_train=X, X_test=X, y_train=y, y_test=y)
    # save logistic models
    joblib.dump(logistic_models, 'models/logistic_models.pkl')
    # save best_threshholds also as a csv
    np.savetxt('models/best_thresholds_lr.csv',
               best_thresholds, delimiter=',')


if __name__ == "__main__":
    main()
