"""
CS 421: Natural Language Processing
Assignment 4: Named Entity Recognition, TF-IDF, and PPMI

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

# For Q3: Deep Learning
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import gensim.downloader as api
    kerasReady = True
except ImportError:
    kerasReady = False
    print("yo Keras/TensorFlow aint available, Q3 wont run")


# ============================================================================
# QUESTION 1: TF-IDF AND COSINE SIMILARITY (25 points)
# ============================================================================

class TfidfMagicBox:
    """
    Custom TF-IDF Vectorizer built from scratch

    Implements the formulas:
    - TF(t,d) = log10(count(t,d) + 1)
    - IDF(t) = log10(N / df_t)
    - TF-IDF(t,d) = TF(t,d) * IDF(t)
    """

    def __init__(self):
        self.wordToSpot = {}      # Q1.2: word-to-index dictionary
        self.docAppearances = {}  # Q1.3: document frequency dictionary
        self.totalDocs = 0        # Total number of documents (N)
        self.vocabStash = set()   # Q1.1: vocabulary set

    def harvestVocab(self, dataChunk: pd.DataFrame):
        """
        Q1.1: Create a vocabulary set of all words appearing in the tokens column

        Args:
            dataChunk: DataFrame with 'tokens' column
        """
        self.vocabStash = set()
        for wordList in dataChunk['tokens']:
            self.vocabStash.update(wordList)
        print(f"vocab size: {len(self.vocabStash)} words total")

    def mapWordsToNumbers(self):
        """
        Q1.2: Create word-to-index dictionary

        Assigns unique integer index to each word, incremented by one for each new word.
        """
        self.wordToSpot = {}
        for spotNum, word in enumerate(sorted(self.vocabStash), start=1):
            self.wordToSpot[word] = spotNum
        print(f"word to index mapping done, {len(self.wordToSpot)} entries")

    def countDocOccurrences(self, dataChunk: pd.DataFrame):
        """
        Q1.3: Create document frequency dictionary

        key = word
        value = number of documents (rows) in which the word occurs

        Args:
            dataChunk: DataFrame with 'tokens' column
        """
        self.docAppearances = defaultdict(int)
        for wordList in dataChunk['tokens']:
            uniqueOnes = set(wordList)
            for word in uniqueOnes:
                self.docAppearances[word] += 1
        self.docAppearances = dict(self.docAppearances)
        print(f"doc frequency dict ready to go")

    def termFrequency(self, term: str, document: List[str]) -> float:
        """
        Q1.4: Calculate term frequency

        tf(t,d) = log10(count(t,d) + 1)

        Args:
            term: The word to calculate frequency for
            document: List of words in the document

        Returns:
            Term frequency value
        """
        howMany = document.count(term)
        return math.log10(howMany + 1)

    def idf(self, word: str) -> float:
        """
        Q1.5: Calculate inverse document frequency

        idf(t) = log10(N / df_t)

        If the term is not in the dictionary, assume df_t = 1 to avoid division by 0.

        Args:
            word: The word to calculate IDF for

        Returns:
            IDF value
        """
        docFreqCount = self.docAppearances.get(word, 1)  # Assume df_t = 1 if not found
        return math.log10(self.totalDocs / docFreqCount)

    def tfidf(self, document: List[str]) -> np.ndarray:
        """
        Q1.6: Compute TF-IDF for one document

        Creates a NumPy array of all zeros with shape (len(vocab), ).
        For each word in the document:
        - Calculate TF using term_frequency
        - Calculate IDF using idf
        - Compute TF * IDF
        - Update the value in the array at the index of the word

        Args:
            document: List of words in the document

        Returns:
            NumPy array of TF-IDF values
        """
        resultVector = np.zeros(len(self.vocabStash))

        for word in document:
            if word in self.wordToSpot:
                arraySpot = self.wordToSpot[word] - 1
                tfScore = self.termFrequency(word, document)
                idfScore = self.idf(word)
                resultVector[arraySpot] = tfScore * idfScore

        return resultVector

    def learnFromData(self, dataChunk: pd.DataFrame):
        """
        Fit the vectorizer on a DataFrame

        Args:
            dataChunk: DataFrame with 'tokens' column
        """
        self.totalDocs = len(dataChunk)

        # Q1.1: Create vocabulary set
        self.harvestVocab(dataChunk)

        # Q1.2: Create word-to-index dictionary
        self.mapWordsToNumbers()

        # Q1.3: Create document frequency dictionary
        self.countDocOccurrences(dataChunk)

        print(f"fitted on {self.totalDocs} docs")

    def constructTfidfMatrix(self, dataChunk: pd.DataFrame) -> np.ndarray:
        """
        Q1.7: Build the TF-IDF matrix for the dataset

        Creates an empty NumPy array and appends TF-IDF vectors for all documents.

        Args:
            dataChunk: DataFrame with 'tokens' column

        Returns:
            TF-IDF matrix (num_docs x vocab_size)
        """
        matrixPile = []

        for wordList in dataChunk['tokens']:
            singleVector = self.tfidf(wordList)
            matrixPile.append(singleVector)

        return np.array(matrixPile)

    def learnAndTransform(self, dataChunk: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            dataChunk: DataFrame with 'tokens' column

        Returns:
            TF-IDF matrix
        """
        self.learnFromData(dataChunk)
        return self.constructTfidfMatrix(dataChunk)


def cosineSimilarity(vecA: np.ndarray, vecB: np.ndarray) -> float:
    """
    Q1.8: Calculate cosine similarity between two vectors

    cos(v, w) = (v · w) / (||v|| * ||w||)
               = sum(v_i * w_i) / (sqrt(sum(v_i^2)) * sqrt(sum(w_i^2)))

    Args:
        vecA: First vector
        vecB: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    dotProd = np.sum(vecA * vecB)
    magA = np.sqrt(np.sum(vecA ** 2))
    magB = np.sqrt(np.sum(vecB ** 2))

    if magA == 0 or magB == 0:
        return 0.0

    return dotProd / (magA * magB)


def runQuestionOne():
    """
    Q1: Implement TF-IDF vectorizer and compute cosine similarity
    """
    print("\n" + "=" * 70)
    print("QUESTION 1: TF IDF AND COSINE SIMILARITY")
    print("=" * 70 + "\n")

    # Q1.1: Load the dataset
    print("loading CoNLL2003 dataset...")
    datasetBundle = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # Load the dataset in a pandas DataFrame
    rawTrainData = datasetBundle['train']

    # Convert to DataFrame
    dataFrame = pd.DataFrame({
        'tokens': rawTrainData['tokens'],
        'ner_tags': rawTrainData['ner_tags']
    })

    # Drop all columns except tokens and ner_tags (already done by selecting only these)
    print(f"dataset loaded into DataFrame, shape: {dataFrame.shape}")
    print(f"columns: {dataFrame.columns.tolist()}")

    # Use a subset for efficiency
    dataFrame = dataFrame.head(1000)
    print(f"using first {len(dataFrame)} rows\n")

    # Initialize and fit TF-IDF vectorizer
    magicBox = TfidfMagicBox()

    # Fit the vectorizer (creates vocabulary, word-to-index, doc freq)
    magicBox.learnFromData(dataFrame)

    # Q1.7: Build TF-IDF matrix
    print("\nbuilding tfidf matrix...")
    bigMatrix = magicBox.constructTfidfMatrix(dataFrame)
    print(f"tfidf matrix shape: {bigMatrix.shape}")
    print(f"thats {bigMatrix.shape[0]} docs x {bigMatrix.shape[1]} features\n")

    # Q1.8: Cosine similarity examples
    print("=" * 70)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 70 + "\n")

    # Example 1
    print("Example 1:")
    sentA = "I love football"
    sentB = "I do not love football"
    wordsA = sentA.lower().split()
    wordsB = sentB.lower().split()

    vecA = magicBox.tfidf(wordsA)
    vecB = magicBox.tfidf(wordsB)
    simScore1 = cosineSimilarity(vecA, vecB)

    print(f"  sentence 1: '{sentA}'")
    print(f"  sentence 2: '{sentB}'")
    print(f"  cosine similarity: {simScore1:.6f}")
    print(f"\n  observation: they share words like 'I', 'love', 'football' but the")
    print(f"  negation 'not' changes everything. tfidf catches word overlap but")
    print(f"  misses the semantic flip from negation.\n")

    # Example 2
    print("Example 2:")
    sentC = "I follow cricket"
    sentD = "I follow baseball"
    wordsC = sentC.lower().split()
    wordsD = sentD.lower().split()

    vecC = magicBox.tfidf(wordsC)
    vecD = magicBox.tfidf(wordsD)
    simScore2 = cosineSimilarity(vecC, vecD)

    print(f"  sentence 1: '{sentC}'")
    print(f"  sentence 2: '{sentD}'")
    print(f"  cosine similarity: {simScore2:.6f}")
    print(f"\n  observation: same structure, two out of three words match ('I', 'follow').")
    print(f"  only diff is cricket vs baseball, both sports. similarity score reflects")
    print(f"  the structural and lexical similarity pretty well.\n")

    return magicBox, bigMatrix, dataFrame


# ============================================================================
# QUESTION 2: PPMI (POSITIVE POINTWISE MUTUAL INFORMATION) (5 points)
# ============================================================================

def calculatePpmi(wordSequence: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Q2: Calculate Positive Pointwise Mutual Information for word pairs

    PPMI(x, y) = max(PMI(x, y), 0)
    PMI(x, y) = log2(p(x, y) / (p(x) * p(y)))

    Where:
    - p(x) = probability of word x occurring in the text
    - p(y) = probability of word y occurring in the text
    - p(x, y) = probability of x and y occurring together (adjacent)

    Args:
        wordSequence: List of strings representing the words in the text

    Returns:
        Dictionary mapping tuples of words to their PPMI values
    """
    # Count individual word occurrences
    wordTally = Counter(wordSequence)
    totalWordCount = len(wordSequence)

    # Count adjacent word pairs (co-occurrences)
    pairTally = Counter()
    for idx in range(len(wordSequence) - 1):
        wordPair = (wordSequence[idx], wordSequence[idx + 1])
        pairTally[wordPair] += 1

    totalPairCount = sum(pairTally.values())

    # Calculate PPMI for each pair
    ppmiResults = {}

    for (wordX, wordY), pairCount in pairTally.items():
        # Calculate probabilities
        probX = wordTally[wordX] / totalWordCount
        probY = wordTally[wordY] / totalWordCount
        probXY = pairCount / totalPairCount

        # Calculate PMI
        if probX > 0 and probY > 0 and probXY > 0:
            pmiValue = math.log2(probXY / (probX * probY))
            # PPMI = max(PMI, 0)
            ppmiValue = max(pmiValue, 0)
            ppmiResults[(wordX, wordY)] = ppmiValue

    return ppmiResults


def runQuestionTwo():
    """
    Q2: Demonstrate PPMI calculation with examples
    """
    print("\n" + "=" * 70)
    print("QUESTION 2: PPMI CALCULATION")
    print("=" * 70 + "\n")

    # Example from assignment specification
    sampleWords = ['a', 'b', 'a', 'c']
    ppmiOutput = calculatePpmi(sampleWords)

    print("example: words = ['a', 'b', 'a', 'c']")
    print("\nppmi results:")
    print("=" * 40)
    for wordPair, ppmiVal in sorted(ppmiOutput.items()):
        print(f"  {wordPair}: {ppmiVal:.6f}")
    print("=" * 40)

    # Demonstrate with additional example
    print("\n" + "=" * 70)
    print("BONUS EXAMPLE FOR FUN:")
    print("=" * 70 + "\n")

    bonusText = "the cat sat on the mat the dog sat on the log".split()
    ppmiOutput2 = calculatePpmi(bonusText)

    print(f"text: '{' '.join(bonusText)}'")
    print("\nppmi results (sorted by value):")
    print("=" * 50)
    for wordPair, ppmiVal in sorted(ppmiOutput2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {wordPair[0]:8s} -> {wordPair[1]:8s} : {ppmiVal:.6f}")
    print("=" * 50)

    return ppmiOutput, ppmiOutput2


# ============================================================================
# QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM (20 points)
# ============================================================================

class NerBrainModel:
    """
    Named Entity Recognition model using LSTM

    Architecture (as specified):
    - Embedding layer (using Word2Vec embeddings)
    - 3 LSTM layers
    - 1 fully-connected (dense) layer
    - 1 final fully-connected layer with softmax activation and 9 outputs
    """

    def __init__(self, vocabCount: int, embeddingSize: int = 300,
                 maxSeqLen: int = 100, tagCount: int = 9):
        """
        Initialize NER model

        Args:
            vocabCount: Size of vocabulary
            embeddingSize: Dimension of word embeddings (300 for Word2Vec)
            maxSeqLen: Maximum sequence length
            tagCount: Number of NER tags (9 for CoNLL2003)
        """
        self.vocabCount = vocabCount
        self.embeddingSize = embeddingSize
        self.maxSeqLen = maxSeqLen
        self.tagCount = tagCount
        self.neuralNet = None

    def assembleModel(self, embeddingWeights=None):
        """
        Q3.2: Build the LSTM model architecture

        Creates a Sequential model with:
        - Embedding layer
        - 3 LSTM layers
        - 1 fully-connected (dense) layer
        - 1 final fully-connected layer with softmax activation and 9 outputs

        Args:
            embeddingWeights: Pre-trained embedding matrix from Word2Vec
        """
        brainModel = Sequential()

        # Embedding layer
        if embeddingWeights is not None:
            brainModel.add(Embedding(
                input_dim=self.vocabCount,
                output_dim=self.embeddingSize,
                weights=[embeddingWeights],
                input_length=self.maxSeqLen,
                trainable=False,  # Use pre-trained embeddings
                mask_zero=True
            ))
        else:
            brainModel.add(Embedding(
                input_dim=self.vocabCount,
                output_dim=self.embeddingSize,
                input_length=self.maxSeqLen,
                mask_zero=True
            ))

        # 3 LSTM layers
        brainModel.add(LSTM(128, return_sequences=True, dropout=0.2))
        brainModel.add(LSTM(64, return_sequences=True, dropout=0.2))
        brainModel.add(LSTM(32, return_sequences=True, dropout=0.2))

        # 1 fully-connected (dense) layer
        brainModel.add(Dense(64, activation='relu'))
        brainModel.add(Dropout(0.3))

        # Final fully-connected layer with softmax activation and 9 outputs
        brainModel.add(Dense(self.tagCount, activation='softmax'))

        # Q3.3: Compile with cross entropy loss and Adam optimizer
        brainModel.compile(
            loss='categorical_crossentropy',  # Cross entropy loss
            optimizer='adam',                  # Adam optimizer (recommended)
            metrics=['accuracy']
        )

        self.neuralNet = brainModel
        return brainModel

    def showArchitecture(self):
        """Print model summary"""
        if self.neuralNet:
            self.neuralNet.summary()


def prepNerDataFrame(datasetObj):
    """
    Prepare CoNLL2003 data for NER training

    Args:
        datasetObj: CoNLL2003 dataset

    Returns:
        DataFrame with 'tokens' and 'ner_tags' columns
    """
    rawTrainChunk = datasetObj['train']

    # Create DataFrame with only tokens and ner_tags
    cleanFrame = pd.DataFrame({
        'tokens': [[word.lower() for word in wordList] for wordList in rawTrainChunk['tokens']],
        'ner_tags': rawTrainChunk['ner_tags']
    })

    return cleanFrame


def buildEmbeddingWeights(wordMapping: Dict, word2vecEngine, vectorSize: int = 300) -> np.ndarray:
    """
    Create embedding matrix from Word2Vec model

    Args:
        wordMapping: Word to index mapping
        word2vecEngine: Loaded Word2Vec model (Google News corpus)
        vectorSize: Embedding dimension (300)

    Returns:
        Embedding matrix
    """
    totalVocab = len(wordMapping)
    weightMatrix = np.zeros((totalVocab, vectorSize))

    foundCount = 0
    for word, spotNum in wordMapping.items():
        if word in word2vecEngine:
            weightMatrix[spotNum] = word2vecEngine[word]
            foundCount += 1
        else:
            # Random initialization for words not in Word2Vec
            weightMatrix[spotNum] = np.random.normal(0, 0.1, vectorSize)

    print(f"found {foundCount}/{totalVocab} words in word2vec ({100*foundCount/totalVocab:.2f}%)")
    return weightMatrix


def runQuestionThree():
    """
    Q3: Implement Named Entity Recognition using LSTM
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM")
    print("=" * 70 + "\n")

    if not kerasReady:
        print("ERROR: keras/tensorflow not available, install the packages first")
        return None

    # Load dataset
    print("loading CoNLL2003 dataset...")
    datasetBundle = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # Prepare data
    print("prepping the data...")
    cleanFrame = prepNerDataFrame(datasetBundle)
    print(f"total samples: {len(cleanFrame)}")

    # Use subset for efficiency
    cleanFrame = cleanFrame.head(5000)
    print(f"using {len(cleanFrame)} samples\n")

    # Build vocabulary
    allUniqueWords = set()
    for wordList in cleanFrame['tokens']:
        allUniqueWords.update(wordList)

    wordToSpot = {'<PAD>': 0, '<UNK>': 1}
    for spotNum, word in enumerate(sorted(allUniqueWords), start=2):
        wordToSpot[word] = spotNum

    spotToWord = {v: k for k, v in wordToSpot.items()}

    print(f"vocab size: {len(wordToSpot)}")

    # Tag mapping (9 NER tags: 0-8)
    tagToSpot = {i: i for i in range(9)}
    spotToTag = {v: k for k, v in tagToSpot.items()}

    print(f"ner tag count: {len(tagToSpot)}")

    # Determine max length
    longestSeq = max(len(wordList) for wordList in cleanFrame['tokens'])
    longestSeq = min(longestSeq, 100)
    print(f"max sequence length: {longestSeq}\n")

    # Convert sentences to sequences
    inputSeqs = []
    outputSeqs = []

    for wordList, tagList in zip(cleanFrame['tokens'], cleanFrame['ner_tags']):
        # Convert words to indices
        numericSeq = [wordToSpot.get(word, wordToSpot['<UNK>']) for word in wordList]
        inputSeqs.append(numericSeq)
        outputSeqs.append(tagList)

    # Pad sequences
    paddedInputs = pad_sequences(inputSeqs, maxlen=longestSeq, padding='post', value=wordToSpot['<PAD>'])
    paddedOutputs = pad_sequences(outputSeqs, maxlen=longestSeq, padding='post', value=0)

    # Convert tags to categorical (one-hot encoding)
    categoricalOutputs = np.array([to_categorical(seq, num_classes=9) for seq in paddedOutputs])

    # Q3.1: Split data 80/20 using sklearn
    print("splitting data (80% train, 20% test)...")
    trainX, testX, trainY, testY = train_test_split(
        paddedInputs, categoricalOutputs, test_size=0.2, random_state=42
    )

    print(f"training samples: {len(trainX)} (80%)")
    print(f"testing samples: {len(testX)} (20%)\n")

    # Q3.1: Load Word2Vec embeddings from Google News corpus
    print("loading word2vec embeddings (google news corpus)...")
    print("this might take a bit on first run...\n")

    try:
        # Load pre-trained Word2Vec model
        word2vecEngine = api.load("word2vec-google-news-300")
        print("word2vec loaded successfully!\n")

        # Create embedding matrix
        embeddingWeights = buildEmbeddingWeights(wordToSpot, word2vecEngine)
    except Exception as err:
        print(f"warning: couldnt load word2vec: {err}")
        print("using random embeddings instead\n")
        embeddingWeights = None

    # Q3.2: Build LSTM model
    print("building lstm model...")
    nerBrain = NerBrainModel(
        vocabCount=len(wordToSpot),
        embeddingSize=300,
        maxSeqLen=longestSeq,
        tagCount=9
    )
    nerBrain.assembleModel(embeddingWeights)
    nerBrain.showArchitecture()

    # Q3.3: Train model with cross entropy loss, Adam optimizer, 10 epochs
    print("\n" + "=" * 70)
    print("TRAINING MODEL (10 epochs)")
    print("=" * 70 + "\n")

    trainingLog = nerBrain.neuralNet.fit(
        trainX, trainY,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Q3.4: Evaluate model on test set
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70 + "\n")

    print("evaluating model on test set...")
    predictedProbs = nerBrain.neuralNet.predict(testX)
    predictedClasses = np.argmax(predictedProbs, axis=-1)
    actualClasses = np.argmax(testY, axis=-1)

    # Flatten predictions and true labels
    flatPredictions = []
    flatActuals = []

    for idx in range(len(actualClasses)):
        for pos in range(len(actualClasses[idx])):
            flatPredictions.append(predictedClasses[idx][pos])
            flatActuals.append(actualClasses[idx][pos])

    # Q3.4: Calculate metrics
    accScore = accuracy_score(flatActuals, flatPredictions)
    precScore, recScore, f1Score, _ = precision_recall_fscore_support(
        flatActuals, flatPredictions, average='macro', zero_division=0
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  accuracy:               {accScore:.6f}")
    print(f"  macro avg precision:    {precScore:.6f}")
    print(f"  macro avg recall:       {recScore:.6f}")
    print(f"  macro avg f1 score:     {f1Score:.6f}")
    print("=" * 70 + "\n")

    return nerBrain, trainingLog, {
        'accuracy': accScore,
        'precision': precScore,
        'recall': recScore,
        'f1': f1Score
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def runEverything():
    """
    Main function to run all assignment questions
    """
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 17 + "CS 421: NLP ASSIGNMENT 4" + " " * 27 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    outputStash = {}

    # Question 1: TF-IDF and Cosine Similarity (25 points)
    print("\n[running question 1: tfidf and cosine similarity]")
    try:
        magicBox, bigMatrix, dataFrame = runQuestionOne()
        outputStash['q1'] = {
            'vectorizer': magicBox,
            'matrix': bigMatrix,
            'dataframe': dataFrame
        }
    except Exception as err:
        print(f"\nerror in Q1: {err}")
        import traceback
        traceback.print_exc()

    # Question 2: PPMI (5 points)
    print("\n[running question 2: ppmi calculation]")
    try:
        ppmi1, ppmi2 = runQuestionTwo()
        outputStash['q2'] = {'ppmiExample1': ppmi1, 'ppmiExample2': ppmi2}
    except Exception as err:
        print(f"\nerror in Q2: {err}")
        import traceback
        traceback.print_exc()

    # Question 3: NER with LSTM (20 points)
    print("\n[running question 3: ner using lstm]")
    try:
        nerBrain, trainingLog, metricsDict = runQuestionThree()
        outputStash['q3'] = {
            'model': nerBrain,
            'history': trainingLog,
            'metrics': metricsDict
        }
    except Exception as err:
        print(f"\nerror in Q3: {err}")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 70)
    print("*" + " " * 22 + "ASSIGNMENT COMPLETE!" + " " * 26 + "*")
    print("*" * 70 + "\n")

    return outputStash


if __name__ == "__main__":
    outputStash = runEverything()
