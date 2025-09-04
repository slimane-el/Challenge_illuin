import json
import pandas as pd
import os


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


def unique_tags_dataset(dataset):
    """
    Extracts unique tags from the 'tags' column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tags' column.

    Returns:
        list: A list of unique tags.
    """
    unique_tags = set()
    for tags in df['tags']:
        for tag in tags:
            unique_tags.add(tag)
    return list(unique_tags)
