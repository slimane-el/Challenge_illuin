# Code Problem Multi-Label Classification

This project implements a **multi-label classification pipeline** for coding problem descriptions, using a combination of **text embeddings**, **code embeddings**, and **machine learning models**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [License](#license)

---

## Project Overview

Given a dataset of coding problems with multiple tags (e.g., `math`, `graphs`, `trees`), the goal is to **predict all relevant tags** for a new problem description. The pipeline supports:

- Text preprocessing and lemmatization
- Feature extraction from **text** using TF-IDF and GloVe embeddings
- Feature extraction from **code** using **CodeBERT**
- Multi-label classification using:
  - Logistic Regression (one-vs-all)
  - Neural Network (sigmoid output)
  - Voting ensemble (hard or soft voting)
- Threshold optimization per label using **Optuna**

---

## Features

- Multi-label classification
- Top-`n` keywords extraction using TF-IDF
- GloVe word embeddings for text
- CodeBERT embeddings for code snippets
- Threshold optimization for each label
- Support for **hard and soft voting** ensembles
- Cross-validation with stratified splits

---

## Models

1. **Logistic Regression (One-vs-All)**

   - Multi-label prediction using label-specific thresholds

2. **Neural Network**

   - Fully connected layers with sigmoid activation
   - Binary cross-entropy loss for multi-label classification

3. **Voting Ensemble**
   - Hard voting: label predicted if threshold number of models agree
   - Soft voting: average predicted probabilities per label

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-multilabel-classification.git
cd code-multilabel-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to execute code
By executing main.py on the challenge folder, and following the instructions given (given inputs for the model and choosing the model).
