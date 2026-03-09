```markdown
# SMS Spam Classification with BERT

This repository presents an end-to-end NLP classification pipeline for detecting spam messages using a fine-tuned `bert-base-uncased` model.

## Project Overview

The notebook builds a supervised text classification workflow for SMS spam detection. It loads and standardizes an SMS dataset, performs train/validation/test splitting, tokenizes text using a BERT tokenizer, fine-tunes a sequence classification model, and evaluates performance on a held-out test set.

## Objectives

- Build a robust SMS spam classifier using BERT
- Fine-tune a pretrained transformer model for binary text classification
- Evaluate classification quality using multiple performance metrics
- Demonstrate a reproducible PyTorch + Hugging Face NLP workflow

## Workflow

### 1. Data Loading and Standardization
- Load the SMS spam dataset
- Normalize column names and labels
- Convert target labels into binary format

### 2. Data Splitting
- Create an 80/20 train-test split
- Further reserve validation data from the training portion
- Preserve class distribution through stratified splitting

### 3. Text Tokenization
- Use the `bert-base-uncased` tokenizer
- Convert SMS messages into input IDs, token type IDs, and attention masks
- Pad and truncate sequences to a fixed maximum length

### 4. Model Construction
- Load `bert-base-uncased` with a sequence-classification head
- Configure label mappings for ham vs. spam
- Prepare PyTorch datasets and dataloaders

### 5. Fine-Tuning
- Train the model across multiple epochs
- Track validation performance
- Save the best model checkpoint
- Optimize the decision threshold for classification

### 6. Evaluation
Evaluate the model on the untouched test split using:
- test loss
- accuracy
- sensitivity / recall for spam
- specificity / recall for ham
- decision threshold

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- NumPy

## Repository Structure

- `Answer for_problem2_bert_sms_spam_classification.ipynb`  
  Main notebook for fine-tuning BERT on SMS spam detection

## Installation

```bash
pip install torch transformers scikit-learn pandas numpy
