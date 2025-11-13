"""
CS 421: Natural Language Processing
Assignment 4: Named Entity Recognition, TF-IDF, and PPMI

Author: [Your Name]
Date: November 2025

This assignment implements:
1. TF-IDF vectorizer from scratch with cosine similarity
2. Positive Pointwise Mutual Information (PPMI) calculation
3. Named Entity Recognition using LSTM neural networks
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
import math
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# For Q3: Deep Learning stuff
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import gensim.downloader as api
    kerasAvailable = True
except ImportError:
    kerasAvailable = False
    print("Note: Keras/TensorFlow not available. Q3 won't run without it.")


# ============================================================================
# QUESTION 1: TF-IDF AND COSINE SIMILARITY (25 points)
# ============================================================================

class TfidfVectorizer:
    """
    Custom TF-IDF Vectorizer - built from scratch
    Implements the formulas:
    - TF(t,d) = log10(count(t,d) + 1)
    - IDF(t) = log10(N / df_t)
    - TF-IDF(t,d) = TF(t,d) * IDF(t)
    """

    def __init__(self):
        self.vocabulary = {}  # maps words to their index positions
        self.idfValues = {}  # stores idf scores for each word
        self.numDocs = 0  # total number of documents

    def buildVocabulary(self, documents: List[List[str]]):
        """
        Build vocabulary from documents - create word-to-index mapping

        Args:
            documents: List of documents, where each document is a list of words
        """
        uniqueWords = set()
        for doc in documents:
            uniqueWords.update(doc)

        # Create word to index mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(uniqueWords))}
        print(f"Vocabulary size: {len(self.vocabulary)}")

    def calculateDocFrequency(self, documents: List[List[str]]) -> Dict[str, int]:
        """
        Calculate document frequency for each term - how many docs contain each word

        Args:
            documents: List of documents

        Returns:
            Dictionary mapping word -> number of documents containing the word
        """
        dfDict = defaultdict(int)
        for doc in documents:
            uniqueWords = set(doc)
            for word in uniqueWords:
                dfDict[word] += 1
        return dict(dfDict)

    def getTermFrequency(self, term: str, document: List[str]) -> float:
        """
        Calculate term frequency using formula: tf(t,d) = log10(count(t,d) + 1)

        Args:
            term: The word to calculate frequency for
            document: List of words in the document

        Returns:
            Term frequency value
        """
        count = document.count(term)
        return math.log10(count + 1)

    def getIdf(self, word: str) -> float:
        """
        Get inverse document frequency: idf(t) = log10(N / df_t)

        Args:
            word: The word to calculate IDF for

        Returns:
            IDF value
        """
        if word in self.idfValues:
            return self.idfValues[word]
        return 0.0

    def fit(self, documents: List[List[str]]):
        """
        Fit the vectorizer on documents - learn the vocabulary and IDF values

        Args:
            documents: List of documents (each document is a list of words)
        """
        self.numDocs = len(documents)

        # Step 1: Build vocabulary
        self.buildVocabulary(documents)

        # Step 2: Calculate document frequencies
        dfDict = self.calculateDocFrequency(documents)

        # Step 3: Calculate IDF for each word
        for word in self.vocabulary:
            df = dfDict.get(word, 0)
            if df > 0:
                self.idfValues[word] = math.log10(self.numDocs / df)
            else:
                self.idfValues[word] = 0.0

        print(f"Fitted on {self.numDocs} documents")

    def getTfidfVector(self, document: List[str]) -> np.ndarray:
        """
        Calculate TF-IDF vector for a single document

        Args:
            document: List of words

        Returns:
            NumPy array of TF-IDF values
        """
        vector = np.zeros(len(self.vocabulary))

        for word in document:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = self.getTermFrequency(word, document)
                idf = self.getIdf(word)
                vector[idx] = tf * idf

        return vector

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Transform documents into TF-IDF matrix

        Args:
            documents: List of documents

        Returns:
            TF-IDF matrix (N x V) where N = num documents, V = vocabulary size
        """
        tfidfMatrix = np.zeros((len(documents), len(self.vocabulary)))

        for i, doc in enumerate(documents):
            tfidfMatrix[i] = self.getTfidfVector(doc)

        return tfidfMatrix

    def fitTransform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            documents: List of documents

        Returns:
            TF-IDF matrix
        """
        self.fit(documents)
        return self.transform(documents)


def calculateCosineSimilarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors

    Formula: cos(θ) = (A · B) / (||A|| * ||B||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    dotProduct = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dotProduct / (norm1 * norm2)


def questionOne_TfidfCosineSimilarity():
    """
    Q1: Implement TF-IDF vectorizer and compute cosine similarity
    """
    print("\n" + "="*80)
    print("QUESTION 1: TF-IDF AND COSINE SIMILARITY")
    print("="*80 + "\n")

    # Load CoNLL2003 dataset
    print("Loading CoNLL2003 dataset...")
    dataset = load_dataset("conll2003")

    # Extract tokens from training set
    trainData = dataset['train']

    # Treat each row as a document (list of tokens)
    documents = []
    for i in range(min(1000, len(trainData))):  # Using first 1000 for efficiency
        tokens = trainData[i]['tokens']
        documents.append([token.lower() for token in tokens])

    print(f"Loaded {len(documents)} documents from CoNLL2003")

    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidfMatrix = vectorizer.fitTransform(documents)

    print(f"TF-IDF Matrix shape: {tfidfMatrix.shape}")
    print(f"Matrix dimensions: {tfidfMatrix.shape[0]} documents x {tfidfMatrix.shape[1]} features\n")

    # Test sentences for cosine similarity
    testPairs = [
        ("I love football", "I do not love football"),
        ("I follow cricket", "I follow baseball")
    ]

    print("Computing cosine similarities:\n")

    for sent1, sent2 in testPairs:
        # Tokenize and lowercase
        tokens1 = sent1.lower().split()
        tokens2 = sent2.lower().split()

        # Get TF-IDF vectors
        vec1 = vectorizer.getTfidfVector(tokens1)
        vec2 = vectorizer.getTfidfVector(tokens2)

        # Calculate cosine similarity
        similarity = calculateCosineSimilarity(vec1, vec2)

        print(f"Sentence 1: '{sent1}'")
        print(f"Sentence 2: '{sent2}'")
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Interpretation: {'Similar' if similarity > 0.5 else 'Dissimilar'}\n")

    return vectorizer, tfidfMatrix


# ============================================================================
# QUESTION 2: PPMI (POSITIVE POINTWISE MUTUAL INFORMATION) (5 points)
# ============================================================================

def calculatePpmi(words: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Calculate Positive Pointwise Mutual Information for word pairs

    PPMI(x, y) = max(PMI(x, y), 0)
    PMI(x, y) = log2(p(x, y) / (p(x) * p(y)))

    Where:
    - p(x) = count(x) / total_words
    - p(y) = count(y) / total_words
    - p(x, y) = count(x, y) / total_pairs

    Args:
        words: List of words

    Returns:
        Dictionary mapping (word_x, word_y) tuples to PPMI values
    """
    print("\n" + "="*80)
    print("QUESTION 2: PPMI CALCULATION")
    print("="*80 + "\n")

    # Count individual words
    wordCounts = Counter(words)
    totalWords = len(words)

    # Count word pairs (co-occurrence)
    pairCounts = Counter()
    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        pairCounts[pair] += 1

    totalPairs = sum(pairCounts.values())

    # Calculate PPMI for each pair
    ppmiDict = {}

    for (wordX, wordY), pairCount in pairCounts.items():
        # Calculate probabilities
        probX = wordCounts[wordX] / totalWords
        probY = wordCounts[wordY] / totalWords
        probXY = pairCount / totalPairs

        # Calculate PMI
        if probX > 0 and probY > 0 and probXY > 0:
            pmi = math.log2(probXY / (probX * probY))
            # PPMI = max(PMI, 0)
            ppmi = max(pmi, 0)
            ppmiDict[(wordX, wordY)] = ppmi

    return ppmiDict


def questionTwo_Ppmi():
    """
    Q2: Demonstrate PPMI calculation with examples
    """
    # Example from assignment
    exampleWords = ['a', 'b', 'a', 'c']
    ppmiResult = calculatePpmi(exampleWords)

    print("Example: words = ['a', 'b', 'a', 'c']")
    print("\nPPMI Results:")
    for pair, ppmiValue in sorted(ppmiResult.items()):
        print(f"  {pair}: {ppmiValue:.4f}")

    # Test with a more realistic example
    print("\n" + "-"*80 + "\n")
    textExample = "the cat sat on the mat the dog sat on the log".split()
    ppmiResult2 = calculatePpmi(textExample)

    print(f"Example: {' '.join(textExample)}")
    print("\nPPMI Results:")
    for pair, ppmiValue in sorted(ppmiResult2.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pair}: {ppmiValue:.4f}")

    return ppmiResult, ppmiResult2


# ============================================================================
# QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM (20 points)
# ============================================================================

class NerModel:
    """
    Named Entity Recognition model using LSTM

    Architecture:
    - Embedding layer (using Word2Vec)
    - 3 LSTM layers
    - 1 Dense layer
    - Output layer with softmax (9 classes)
    """

    def __init__(self, vocabSize: int, embeddingDim: int = 300,
                 maxLength: int = 100, numTags: int = 9):
        """
        Initialize NER model

        Args:
            vocabSize: Size of vocabulary
            embeddingDim: Dimension of word embeddings (default: 300 for Word2Vec)
            maxLength: Maximum sequence length
            numTags: Number of NER tags (9 for CoNLL2003)
        """
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.maxLength = maxLength
        self.numTags = numTags
        self.model = None
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}

    def buildModel(self, embeddingMatrix=None):
        """
        Build the LSTM model architecture

        Args:
            embeddingMatrix: Pre-trained embedding matrix (optional)
        """
        model = Sequential()

        # Embedding layer
        if embeddingMatrix is not None:
            model.add(Embedding(
                input_dim=self.vocabSize,
                output_dim=self.embeddingDim,
                weights=[embeddingMatrix],
                input_length=self.maxLength,
                trainable=False,
                mask_zero=True
            ))
        else:
            model.add(Embedding(
                input_dim=self.vocabSize,
                output_dim=self.embeddingDim,
                input_length=self.maxLength,
                mask_zero=True
            ))

        # LSTM Layer 1
        model.add(LSTM(128, return_sequences=True, dropout=0.2))

        # LSTM Layer 2
        model.add(LSTM(64, return_sequences=True, dropout=0.2))

        # LSTM Layer 3
        model.add(LSTM(32, return_sequences=True, dropout=0.2))

        # Dense layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))

        # Output layer with softmax
        model.add(Dense(self.numTags, activation='softmax'))

        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()


def prepareNerData(dataset, maxSamples=5000):
    """
    Prepare CoNLL2003 data for NER training

    Args:
        dataset: CoNLL2003 dataset
        maxSamples: Maximum number of samples to use

    Returns:
        Tuple of (sentences, tags, word2idx, tag2idx)
    """
    sentences = []
    tags = []

    # Extract sentences and tags
    trainData = dataset['train']
    numSamples = min(maxSamples, len(trainData))

    for i in range(numSamples):
        tokens = [token.lower() for token in trainData[i]['tokens']]
        nerTags = trainData[i]['ner_tags']
        sentences.append(tokens)
        tags.append(nerTags)

    # Build vocabularies
    allWords = set(word for sent in sentences for word in sent)
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted(allWords))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1

    # Tag mapping (0-8 for 9 NER tags)
    tag2idx = {i: i for i in range(9)}

    return sentences, tags, word2idx, tag2idx


def createEmbeddingMatrix(word2idx, word2vecModel, embeddingDim=300):
    """
    Create embedding matrix from Word2Vec model

    Args:
        word2idx: Word to index mapping
        word2vecModel: Loaded Word2Vec model
        embeddingDim: Embedding dimension

    Returns:
        Embedding matrix
    """
    vocabSize = len(word2idx)
    embeddingMatrix = np.zeros((vocabSize, embeddingDim))

    found = 0
    for word, idx in word2idx.items():
        if word in word2vecModel:
            embeddingMatrix[idx] = word2vecModel[word]
            found += 1
        else:
            # Random initialization for unknown words
            embeddingMatrix[idx] = np.random.normal(0, 0.1, embeddingDim)

    print(f"Found {found}/{vocabSize} words in Word2Vec model ({100*found/vocabSize:.2f}%)")
    return embeddingMatrix


def questionThree_NerLstm():
    """
    Q3: Implement Named Entity Recognition using LSTM
    """
    print("\n" + "="*80)
    print("QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM")
    print("="*80 + "\n")

    if not kerasAvailable:
        print("ERROR: Keras/TensorFlow not available. Please install required packages.")
        return None

    # Load dataset
    print("Loading CoNLL2003 dataset...")
    dataset = load_dataset("conll2003")

    # Prepare data
    print("Preparing data...")
    sentences, tags, word2idx, tag2idx = prepareNerData(dataset, maxSamples=5000)
    idx2tag = {v: k for k, v in tag2idx.items()}

    print(f"Number of sentences: {len(sentences)}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of NER tags: {len(tag2idx)}")

    # Determine max length
    maxLength = max(len(sent) for sent in sentences)
    maxLength = min(maxLength, 100)  # Cap at 100 for efficiency
    print(f"Maximum sequence length: {maxLength}\n")

    # Convert sentences to sequences
    sequencesX = []
    sequencesY = []

    for sent, tagSeq in zip(sentences, tags):
        # Convert words to indices
        sentIndices = [word2idx.get(word, word2idx['<UNK>']) for word in sent]
        sequencesX.append(sentIndices)
        sequencesY.append(tagSeq)

    # Pad sequences
    xPadded = pad_sequences(sequencesX, maxlen=maxLength, padding='post', value=word2idx['<PAD>'])
    yPadded = pad_sequences(sequencesY, maxlen=maxLength, padding='post', value=0)

    # Convert tags to categorical
    yCategorical = np.array([to_categorical(seq, num_classes=9) for seq in yPadded])

    # Split data 80/20
    xTrain, xTest, yTrain, yTest = train_test_split(
        xPadded, yCategorical, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(xTrain)}")
    print(f"Testing samples: {len(xTest)}\n")

    # Load Word2Vec embeddings
    print("Loading Word2Vec embeddings (this may take a while)...")
    try:
        word2vecModel = api.load("word2vec-google-news-300")
        print("Word2Vec loaded successfully!\n")

        # Create embedding matrix
        embeddingMatrix = createEmbeddingMatrix(word2idx, word2vecModel)
    except Exception as e:
        print(f"Warning: Could not load Word2Vec: {e}")
        print("Using random embeddings instead.\n")
        embeddingMatrix = None

    # Build model
    print("Building LSTM model...")
    nerModel = NerModel(
        vocabSize=len(word2idx),
        embeddingDim=300,
        maxLength=maxLength,
        numTags=9
    )
    nerModel.word2idx = word2idx
    nerModel.idx2tag = idx2tag
    nerModel.buildModel(embeddingMatrix)
    nerModel.summary()

    # Train model
    print("\nTraining model (10 epochs)...")
    history = nerModel.model.fit(
        xTrain, yTrain,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    yPred = nerModel.model.predict(xTest)
    yPredClasses = np.argmax(yPred, axis=-1)
    yTestClasses = np.argmax(yTest, axis=-1)

    # Flatten predictions and true labels (ignoring padding)
    yPredFlat = []
    yTestFlat = []

    for i in range(len(yTestClasses)):
        for j in range(len(yTestClasses[i])):
            if yTestClasses[i][j] != 0 or j < maxLength:  # Not padding
                yPredFlat.append(yPredClasses[i][j])
                yTestFlat.append(yTestClasses[i][j])

    # Calculate metrics
    accuracy = accuracy_score(yTestFlat, yPredFlat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yTestFlat, yPredFlat, average='macro', zero_division=0
    )

    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print("="*80 + "\n")

    return nerModel, history, (accuracy, precision, recall, f1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all assignment questions
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "CS 421: NLP - ASSIGNMENT 4" + " " * 32 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    results = {}

    # Question 1: TF-IDF and Cosine Similarity
    try:
        vectorizer, tfidfMatrix = questionOne_TfidfCosineSimilarity()
        results['q1'] = {'vectorizer': vectorizer, 'matrix': tfidfMatrix}
    except Exception as e:
        print(f"\nError in Q1: {e}")
        import traceback
        traceback.print_exc()

    # Question 2: PPMI
    try:
        ppmi1, ppmi2 = questionTwo_Ppmi()
        results['q2'] = {'ppmi1': ppmi1, 'ppmi2': ppmi2}
    except Exception as e:
        print(f"\nError in Q2: {e}")
        import traceback
        traceback.print_exc()

    # Question 3: NER with LSTM
    try:
        nerModel, history, metrics = questionThree_NerLstm()
        results['q3'] = {
            'model': nerModel,
            'history': history,
            'metrics': metrics
        }
    except Exception as e:
        print(f"\nError in Q3: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 80)
    print("*" + " " * 25 + "ASSIGNMENT COMPLETE!" + " " * 33 + "*")
    print("*" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = main()
