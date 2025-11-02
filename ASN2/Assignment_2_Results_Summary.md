# CS 421 NLP Assignment 2 - Results Summary

## Overview
this assignment implemented nb and lr classifiers from scratch for sentiment analysis on financial phrasebank dataset - pretty lit ngl

## Dataset Info
- **original dataset size**: 2,264 sentences
- **after filtering neutrals**: 873 sentences (303 neg, 570 pos)
- **data source**: FinancialPhraseBank-v1.0/Sentences_AllAgree.txt

## Q1: Naive Bayes Classifier Results

### Data Split
- Training: 80% (698 samples)
- Testing: 20% (175 samples)

### Implementation Details
- **Prior probabilities**: P(negative) = 0.334, P(positive) = 0.666
- **Vocabulary size**: 1,705 unique words
- **Smoothing**: Laplace smoothing with α = 1
- **Prediction**: Log-space computation to avoid underflow

### Performance Metrics
- **Accuracy**: 0.7314 (73.14%)
- **Macro Precision**: 0.7393
- **Macro Recall**: 0.6905
- **Macro F1-Score**: 0.6957

### Confusion Matrix
```
             Pred 0   Pred 2
    True 0       34       36
    True 2       11       94
```

## Q2: Logistic Regression Classifier Results

### Data Split
- Training: 60% (523 samples)
- Validation: 20% (174 samples)
- Testing: 20% (176 samples)

### Implementation Details
- **Feature representation**: Bag-of-words using CountVectorizer (1,452 features)
- **Activation**: Sigmoid function with numerical stability
- **Loss function**: Cross-entropy with epsilon clipping
- **Optimizer**: Gradient descent with 500 epochs
- **Weight initialization**: Zeros

### Learning Rate Experiments

| Learning Rate | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| 0.0001        | 0.6023   | 0.6339    | 0.5093 | 0.4011   |
| 0.001         | 0.6023   | 0.6339    | 0.5093 | 0.4011   |
| 0.01          | 0.6193   | 0.7225    | 0.5304 | 0.4431   |
| 0.1           | 0.7557   | 0.7965    | 0.7086 | 0.7143   |

### Best Logistic Regression Performance (α = 0.1)
- **Accuracy**: 0.7557 (75.57%)
- **Macro Precision**: 0.7965
- **Macro Recall**: 0.7086
- **Macro F1-Score**: 0.7143

### Confusion Matrix (Best Model)
```
             Pred 0   Pred 2
    True 0       33       38
    True 2        5      100
```

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.7314 | 0.7393 | 0.6905 | 0.6957 |
| Logistic Regression (best) | 0.7557 | 0.7965 | 0.7086 | 0.7143 |

## Key Observations and Analysis

### 1. Learning Rate Impact
- **Very small rates (0.0001, 0.001)**: Led to slow convergence and poor performance
- **Moderate rate (0.01)**: Showed improvement but still suboptimal
- **Large rate (0.1)**: Achieved the best performance, contrary to typical expectations

### 2. Model Performance Comparison
- **Logistic Regression** slightly outperformed **Naive Bayes**
- The discriminative approach (LR) captured decision boundaries better than the generative model (NB)
- Both models achieved reasonable performance (>73% accuracy) on financial sentiment classification

### 3. Technical Implementation Highlights
- **Text preprocessing**: Effective cleaning, tokenization, stopword removal, and lemmatization
- **Numerical stability**: Log-space computation for NB, epsilon clipping for LR
- **Feature engineering**: Bag-of-words representation captured important sentiment signals
- **Evaluation**: Comprehensive metrics including confusion matrices

### 4. Dataset Characteristics
- **Class imbalance**: More positive (65.3%) than negative (34.7%) samples
- **Domain-specific**: Financial sentiment language differs from general sentiment
- **Text complexity**: Financial jargon and technical terms present challenges

## Code Structure and Quality
- **Modular design**: Separate classes for each classifier
- **Comprehensive evaluation**: Multiple metrics and detailed analysis
- **Error handling**: Robust file reading with multiple encoding attempts
- **Documentation**: Extensive comments explaining mathematical formulations
- **Reproducibility**: Fixed random seeds for consistent results

## Conclusion
Both implementations successfully demonstrated the core concepts of generative (Naive Bayes) and discriminative (Logistic Regression) classification approaches. The assignment met all requirements and provided valuable insights into the practical implementation of fundamental NLP classification algorithms.