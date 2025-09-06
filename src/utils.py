import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download stopwords if not already done
nltk.download('stopwords')
nltk.download('punkt')


def build_dataset(file_path):
    """
    Reads a JSON file and converts it into a pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the JSON file.
    """
    dataset = []

    for filename in os.listdir(file_path):
        if filename.endswith('.json'):
            file_id = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                data['id'] = file_id
                dataset.append(data)
    return pd.DataFrame(dataset)


def get_unique_tags(dataset):
    """
    Extracts unique tags from the 'tags' column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tags' column.

    Returns:
        list: A list of unique tags.
    """
    unique_tags = set()
    for tags in dataset['tags']:
        for tag in tags:
            unique_tags.add(tag)
    return list(unique_tags)


def distribution_tags(dataset):
    """
    Computes the distribution of tags in the 'tags' column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tags' column.

    Returns:
        dict: A dictionary with tags as keys and their counts as values.
    """
    tag_distribution = {}
    for tags in dataset['tags']:
        for tag in tags:
            if tag in tag_distribution:
                tag_distribution[tag] += 1
            else:
                tag_distribution[tag] = 1
    return tag_distribution


def plot_tag_distribution(tag_distribution):
    """
    Plots the distribution of tags using a bar chart.

    Args:
        tag_distribution (dict): A dictionary with tags as keys and their counts as values.
    """
    tag_distribution = dict(
        sorted(tag_distribution.items(), key=lambda item: item[1], reverse=True))
    tags = list(tag_distribution.keys())
    counts = list(tag_distribution.values())

    plt.figure(figsize=(10, 6))
    plt.bar(tags, counts, color='skyblue')
    plt.xlabel('Tags')
    plt.ylabel('Counts')
    plt.title('Tag Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def co_occurrence_matrix(dataset, valid_tags=None):
    """
    plots the co-occurrence matrix of tags in the 'tags' column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tags' column.

    Returns:
        pd.DataFrame: A DataFrame representing the co-occurrence matrix of tags.
    """
    if valid_tags is None:
        unique_tags = get_unique_tags(dataset)
    else:
        unique_tags = valid_tags
    tag_index = {tag: idx for idx, tag in enumerate(unique_tags)}

    co_occurrence = pd.DataFrame(0, index=unique_tags, columns=unique_tags)
    for tags in dataset['tags']:
        filtered_tags = [tag for tag in tags if tag in tag_index]
        for i in range(len(filtered_tags)):
            for j in range(i, len(filtered_tags)):
                tag1 = filtered_tags[i]
                tag2 = filtered_tags[j]
                co_occurrence.at[tag1, tag2] += 1
                if tag1 != tag2:
                    co_occurrence.at[tag2, tag1] += 1
    co_occurrence_df = co_occurrence.astype(int)
    plt.figure(figsize=(10, 8))
    plt.imshow(co_occurrence_df, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Co-occurrence Count')
    # Rotate x-axis labels for better readability
    plt.xticks(ticks=range(len(unique_tags)), labels=unique_tags, rotation=90)
    plt.yticks(ticks=range(len(unique_tags)), labels=unique_tags)
    plt.title('Tag Co-occurrence Matrix')
    plt.tight_layout()
    plt.show()


def correlation_tags_difficulty(dataset, valid_tags):
    """
    Computes and plots the correlation between tags and difficulty.

    Args:
        dataset (pd.DataFrame): Input DataFrame with 'tags' (list of tags) 
                                and 'difficulty' (numeric).
        valid_tags (list): List of tags to consider.

    Returns:
        pd.Series: Correlation between each tag and difficulty.
    """
    # Create binary indicator columns for each tag
    tag_matrix = pd.DataFrame({
        tag: dataset['tags'].apply(lambda tags: int(tag in tags))
        for tag in valid_tags
    })

    # Compute correlations
    correlation = tag_matrix.corrwith(dataset['difficulty'])

    # Plot
    plt.figure(figsize=(10, 6))
    correlation.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.xlabel('Tags')
    plt.ylabel('Correlation with Difficulty')
    plt.title('Tag-Difficulty Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return correlation


def clean_text(text):
    # Remove anything between $$$
    cleaned_text = re.sub(r'\$\$\$.*?\$\$\$', '', text)
    # Keep only alphanumeric characters and spaces
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove digits

    tokens = word_tokenize(cleaned_text)

    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word.lower()
                      not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text
