# Assignment 4  
**CS 421: Natural Language Processing**

## 1. Introduction
In this assignment, you will implement Named Entity Recognition (NER) using an LSTM model. You will also build a TF-IDF vectorizer and compute cosine similarity.

A Named Entity is any word/term that can be referred to with a proper name (person, location, organization). NER identifies spans of text representing proper names and tags them. LSTMs use forget, input, and output gates to manage sequential information.

## 2. Instructions
Use Python 3.6+ and these packages:
- pandas
- numpy
- NLTK or spaCy
- scikit-learn
- Keras

## 3. Questions

### Q1. TF-IDF and Cosine Similarity
Use the conll2003 dataset. Steps include:
1. Load dataset, keep tokens + ner_tags, treat each row as a document.
2. Map vocabulary words to unique indices.
3. Build document frequency dictionary.
4. Implement term frequency:
   `tf(t,d) = log10(count(t,d) + 1)`
5. Implement IDF:
   `idf(t) = log10(N / dft)`
6. Implement TF-IDF vector for each document.
7. Build TF-IDF matrix.
8. Compute cosine similarity for two sentence pairs.

### Q2. Compute PPMI
PPMI(x,y) = max(PMI(x,y), 0)  
PMI(x,y) = log2(p(x,y) / (p(x)p(y)))

Write `calculate_ppmi(words)` returning a dict mapping (word_x, word_y) → PPMI value.

### Q3. Named Entity Recognition Using LSTM
1. Split dataset 80/20, use Google News word2vec embeddings.
2. Build Keras model: embedding → 3 LSTM layers → Dense → softmax (9 tags).
3. Train with cross-entropy, Adam, 10 epochs.
4. Compute accuracy, macro precision, recall, F1.

## 4. Rubric
- Q1: 25 pts
- Q2: 5 pts
- Q3: 20 pts
