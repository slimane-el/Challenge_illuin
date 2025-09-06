from src.utils import build_dataset, clean_text
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, hamming_loss, f1_score
from sklearn.model_selection import StratifiedGroupKFold

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
    return tf_idf_matrix, top_n_keywords


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
        dict: A dictionary containing precision, recall, and F1-score.
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


def logistic_regression_by_label(X, y, threshold=0.5):
    """
    Trains a logistic regression model for each unique label in the dataset using a threshold approach.

    Args:
        X (np.ndarray): The feature matrix.
        y (pd.Series): The target labels.

    Returns:
        dict: A dictionary mapping each label to its corresponding trained logistic regression model.
    """
    num_classes = Y.shape[1]
    models = {}
    predictions = np.zeros((X.shape[0], num_classes))
    true_labels = np.zeros((X.shape[0], num_classes))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for i in range(num_classes):
        model = LogisticRegression()
        model.fit(X_train, y_train.iloc[:, i])
        models[i] = model
        proba = model.predict_proba(X)[:, 1]
        predictions[:, i] = (proba >= threshold).astype(int)
        true_labels[:, i] = y.iloc[:, i]
    return models, predictions, true_labels


def crossvalidation_logistic_regression(X, y, thresholds=[0.3, 0.5, 0.7], num_folds=5):
    """
    Performs cross-validation for logistic regression models with different thresholds.

    Args:
        X (np.ndarray): The feature matrix.
        y (pd.DataFrame): The target labels.
        thresholds (list): A list of thresholds to evaluate.

    Returns:
        dict: A dictionary mapping each threshold to its corresponding evaluation metrics.
    """
    skf = StratifiedGroupKFold(n_splits=num_folds)
    results = {thr: [] for thr in thresholds}
    groups = np.arange(X.shape[0])  # Dummy groups for StratifiedGroupKFold

    for train_index, test_index in skf.split(X, y.values.argmax(axis=1), groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for thr in thresholds:
            models, preds, true_lbls = logistic_regression_by_label(
                X_train, y_train, threshold=thr)
            metrics = compute_metrics(true_lbls[test_index], preds[test_index])
            results[thr].append(metrics)

    avg_results = {}
    for thr in thresholds:
        avg_metrics = {
            'precision': np.mean([res['precision'] for res in results[thr]]),
            'recall': np.mean([res['recall'] for res in results[thr]]),
            'f1_score': np.mean([res['f1_score'] for res in results[thr]]),
            'hamming_loss': np.mean([res['hamming_loss'] for res in results[thr]])
        }
        avg_results[thr] = avg_metrics
    best_threshold = min(
        avg_results, key=lambda k: avg_results[k]['hamming_loss'])
    return avg_results, best_threshold


def get_embeddings_glove(words, model):
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)


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
        sentences, vector_size=100, window=5, min_count=1, workers=4)
    # TF-IDF and GloVe embeddings
    tf_idf_matrix, top_keywords = get_top_n_tf_idf(corpus, n=5)
    df["keywords"] = top_keywords
    df["embeddings"] = df["keywords"].apply(
        lambda words: get_glove_embeddings(words, model))
    # testing shapes
    print((df["embeddings"][0]), model.vector_size)


if __name__ == "__main__":
    main()
