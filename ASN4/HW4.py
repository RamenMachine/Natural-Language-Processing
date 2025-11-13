"""
CS 421: Natural Language Processing
Assignment 4: Named Entity Recognition, TF-IDF, and PPMI

Author: [Your Name]
Date: November 2025

This joint implements:
1. TF-IDF vectorizer from scratch with cosine similarity (straight fire)
2. Positive Pointwise Mutual Information (PPMI) calculation (word vibes checker)
3. Named Entity Recognition using LSTM neural networks (big brain time)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
import math
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# For Q3: Deep Learning shenanigans
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import gensim.downloader as api
    kerasIsVibing = True
except ImportError:
    kerasIsVibing = False
    print("Yo fam, Keras/TensorFlow ain't installed. Q3 gonna skip rn.")


# ============================================================================
# QUESTION 1: TF-IDF AND COSINE SIMILARITY (25 points)
# ============================================================================

class TfIdfVectorBoss:
    """
    Custom TF-IDF Vectorizer - we buildin this from scratch no cap
    Implements the spicy formulas:
    - TF(t,d) = log10(count(t,d) + 1)  # how often word shows up
    - IDF(t) = log10(N / df_t)  # how rare the word is across docs
    - TF-IDF(t,d) = TF(t,d) * IDF(t)  # the final boss combo
    """

    def __init__(self):
        self.wordDict = {}  # maps words to their index positions (the roster)
        self.idfVibes = {}  # stores idf scores for each word (rarity meter)
        self.totalDocs = 0  # how many docs we workin with

    def buildVocabSwag(self, docsList: List[List[str]]):
        """
        Build up our vocabulary from all the docs - gotta know what words we got

        Args:
            docsList: List of docs, each doc is a list of words (the whole squad)
        """
        uniqueWords = set()
        for singleDoc in docsList:
            uniqueWords.update(singleDoc)

        # Make a dictionary mapping words to numbers (indexin the homies)
        self.wordDict = {word: idx for idx, word in enumerate(sorted(uniqueWords))}
        print(f"Yo! Vocabulary got {len(self.wordDict)} words in it, that's bussin")

    def calculateDocFreq(self, docsList: List[List[str]]) -> Dict[str, int]:
        """
        Count how many docs each word appears in - popularity contest fr

        Args:
            docsList: All the docs we checkin

        Returns:
            Dictionary with word -> how many docs it's in
        """
        freqTracker = defaultdict(int)
        for singleDoc in docsList:
            uniqueWordsInDoc = set(singleDoc)
            for word in uniqueWordsInDoc:
                freqTracker[word] += 1
        return dict(freqTracker)

    def getTermFrequency(self, wordToCheck: str, docToSearch: List[str]) -> float:
        """
        Calculate term frequency - basically how much this word shows up
        Formula: tf(t,d) = log10(count(t,d) + 1)

        Args:
            wordToCheck: The word we countin
            docToSearch: The doc we searchin in

        Returns:
            TF score (higher = word appears more)
        """
        wordCount = docToSearch.count(wordToCheck)
        return math.log10(wordCount + 1)

    def getIdfScore(self, wordToLookup: str) -> float:
        """
        Get the inverse document frequency - tells us how rare/common a word is
        Formula: idf(t) = log10(N / df_t)

        Args:
            wordToLookup: Word we checkin the rarity for

        Returns:
            IDF value (higher = more rare and spicy)
        """
        if wordToLookup in self.idfVibes:
            return self.idfVibes[wordToLookup]
        return 0.0

    def fitTheData(self, docsList: List[List[str]]):
        """
        Train this bad boy on our docs - learn all the word stats

        Args:
            docsList: List of docs to learn from (the training montage)
        """
        self.totalDocs = len(docsList)

        # Step 1: Build our word roster
        self.buildVocabSwag(docsList)

        # Step 2: Count how many docs each word appears in
        freqDict = self.calculateDocFreq(docsList)

        # Step 3: Calculate IDF for each word (find out who's rare)
        for word in self.wordDict:
            docFreq = freqDict.get(word, 0)
            if docFreq > 0:
                self.idfVibes[word] = math.log10(self.totalDocs / docFreq)
            else:
                self.idfVibes[word] = 0.0

        print(f"Fitted on {self.totalDocs} documents - we ready to roll!")

    def makeTfidfVector(self, singleDoc: List[str]) -> np.ndarray:
        """
        Turn a document into a TF-IDF vector - convert words to numbers

        Args:
            singleDoc: Doc as list of words

        Returns:
            Numpy array of TF-IDF scores (the numerical representation)
        """
        vectorSwag = np.zeros(len(self.wordDict))

        for word in singleDoc:
            if word in self.wordDict:
                wordPosition = self.wordDict[word]
                termFreq = self.getTermFrequency(word, singleDoc)
                idfValue = self.getIdfScore(word)
                vectorSwag[wordPosition] = termFreq * idfValue

        return vectorSwag

    def transformDocs(self, docsList: List[List[str]]) -> np.ndarray:
        """
        Transform a whole bunch of docs into TF-IDF matrix

        Args:
            docsList: All the docs we convertin

        Returns:
            Big matrix where each row is a doc (the whole squad in number form)
        """
        bigMatrix = np.zeros((len(docsList), len(self.wordDict)))

        for docIdx, singleDoc in enumerate(docsList):
            bigMatrix[docIdx] = self.makeTfidfVector(singleDoc)

        return bigMatrix

    def fitAndTransform(self, docsList: List[List[str]]) -> np.ndarray:
        """
        Do the fit and transform in one shot - efficiency gang

        Args:
            docsList: Docs to process

        Returns:
            TF-IDF matrix ready to go
        """
        self.fitTheData(docsList)
        return self.transformDocs(docsList)


def calculateCosineSimilarity(firstVec: np.ndarray, secondVec: np.ndarray) -> float:
    """
    Calculate cosine similarity - see how similar two vectors are
    Formula: cos(θ) = (A · B) / (||A|| * ||B||)

    Args:
        firstVec: First vector we comparin
        secondVec: Second vector we comparin

    Returns:
        Similarity score from -1 to 1 (1 = basically twins, 0 = no relation, -1 = opposites)
    """
    dotProductVibes = np.dot(firstVec, secondVec)
    magnitudeFirst = np.linalg.norm(firstVec)
    magnitudeSecond = np.linalg.norm(secondVec)

    if magnitudeFirst == 0 or magnitudeSecond == 0:
        return 0.0  # can't divide by zero, that ain't it chief

    return dotProductVibes / (magnitudeFirst * magnitudeSecond)


def questionOneLettsGo():
    """
    Q1: Build TF-IDF vectorizer and check how similar sentences are
    """
    print("\n" + "="*80)
    print("QUESTION 1: TF-IDF AND COSINE SIMILARITY - LET'S GET IT")
    print("="*80 + "\n")

    # Load that CoNLL2003 dataset (classic NLP dataset fr fr)
    print("Loadin CoNLL2003 dataset... hold up...")
    datasetStash = load_dataset("conll2003")

    # Grab the training data
    trainingDataRaw = datasetStash['train']

    # Each row becomes a document (we treating each sentence as its own vibe)
    docsCollection = []
    for idx in range(min(1000, len(trainingDataRaw))):  # using first 1000 cuz we aint got all day
        tokensFromDoc = trainingDataRaw[idx]['tokens']
        docsCollection.append([token.lower() for token in tokensFromDoc])

    print(f"Loaded {len(docsCollection)} documents from CoNLL2003, we eatin good!")

    # Initialize and train our TF-IDF boss
    vectorizerGoat = TfIdfVectorBoss()
    tfidfMatrixBig = vectorizerGoat.fitAndTransform(docsCollection)

    print(f"TF-IDF Matrix shape: {tfidfMatrixBig.shape}")
    print(f"Matrix vibes: {tfidfMatrixBig.shape[0]} documents x {tfidfMatrixBig.shape[1]} features\n")

    # Test sentences to compare (the moment of truth)
    testSentencePairs = [
        ("I love football", "I do not love football"),
        ("I follow cricket", "I follow baseball")
    ]

    print("Computing cosine similarities - let's see who's similar:\n")

    for firstSentence, secondSentence in testSentencePairs:
        # Break sentences into words and make em lowercase
        tokensFirst = firstSentence.lower().split()
        tokensSecond = secondSentence.lower().split()

        # Convert to TF-IDF vectors
        vecFirst = vectorizerGoat.makeTfidfVector(tokensFirst)
        vecSecond = vectorizerGoat.makeTfidfVector(tokensSecond)

        # Calculate how similar they are
        similarityScore = calculateCosineSimilarity(vecFirst, vecSecond)

        print(f"Sentence 1: '{firstSentence}'")
        print(f"Sentence 2: '{secondSentence}'")
        print(f"Cosine Similarity: {similarityScore:.4f}")
        print(f"Vibe check: {'Similar vibes' if similarityScore > 0.5 else 'Different energy'}\n")

    return vectorizerGoat, tfidfMatrixBig


# ============================================================================
# QUESTION 2: PPMI (POSITIVE POINTWISE MUTUAL INFORMATION) (5 points)
# ============================================================================

def calculatePpmiScores(wordsList: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Calculate PPMI - basically finds which words like to hang out together

    PPMI(x, y) = max(PMI(x, y), 0)  # we only keep positive vibes
    PMI(x, y) = log2(p(x, y) / (p(x) * p(y)))  # how much more they appear together than random

    Where:
    - p(x) = how often word x shows up / total words
    - p(y) = how often word y shows up / total words
    - p(x, y) = how often x and y are next to each other / total pairs

    Args:
        wordsList: List of words to analyze (the whole squad)

    Returns:
        Dictionary with (word1, word2) -> PPMI score
    """
    print("\n" + "="*80)
    print("QUESTION 2: PPMI CALCULATION - WORD ASSOCIATION VIBES")
    print("="*80 + "\n")

    # Count how many times each word appears (popularity contest)
    wordCountTracker = Counter(wordsList)
    totalWordCount = len(wordsList)

    # Count word pairs that appear next to each other (who hangs with who)
    pairCountTracker = Counter()
    for idx in range(len(wordsList) - 1):
        wordPair = (wordsList[idx], wordsList[idx + 1])
        pairCountTracker[wordPair] += 1

    totalPairsCount = sum(pairCountTracker.values())

    # Calculate PPMI for each pair (find the real homies)
    ppmiResultDict = {}

    for (firstWord, secondWord), pairAppearances in pairCountTracker.items():
        # Calculate probabilities (math time)
        probFirst = wordCountTracker[firstWord] / totalWordCount
        probSecond = wordCountTracker[secondWord] / totalWordCount
        probPair = pairAppearances / totalPairsCount

        # Calculate PMI then PPMI
        if probFirst > 0 and probSecond > 0 and probPair > 0:
            pmiScore = math.log2(probPair / (probFirst * probSecond))
            # PPMI = only keep positive scores (no negativity here)
            ppmiScore = max(pmiScore, 0)
            ppmiResultDict[(firstWord, secondWord)] = ppmiScore

    return ppmiResultDict


def questionTwoShowtime():
    """
    Q2: Demonstrate PPMI calculation with some examples
    """
    # Example from the assignment sheet
    exampleWordList = ['a', 'b', 'a', 'c']
    ppmiResults = calculatePpmiScores(exampleWordList)

    print("Example: words = ['a', 'b', 'a', 'c']")
    print("\nPPMI Results (who's vibin together):")
    for wordPair, ppmiVal in sorted(ppmiResults.items()):
        print(f"  {wordPair}: {ppmiVal:.4f}")

    # Try a more realistic example (actual sentence vibes)
    print("\n" + "-"*80 + "\n")
    sentenceExample = "the cat sat on the mat the dog sat on the log".split()
    ppmiResults2 = calculatePpmiScores(sentenceExample)

    print(f"Example: {' '.join(sentenceExample)}")
    print("\nPPMI Results (top 10 word combos):")
    for wordPair, ppmiVal in sorted(ppmiResults2.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {wordPair}: {ppmiVal:.4f}")

    return ppmiResults, ppmiResults2


# ============================================================================
# QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM (20 points)
# ============================================================================

class NerModelBeast:
    """
    Named Entity Recognition model using LSTM - the big brain neural net

    Architecture (the lineup):
    - Embedding layer (turn words into vectors using Word2Vec)
    - 3 LSTM layers (the memory masters)
    - 1 Dense layer (processing power)
    - Output layer with softmax (make predictions for 9 entity types)
    """

    def __init__(self, vocabSizeTotal: int, embeddingDimensions: int = 300,
                 maxSeqLength: int = 100, numTagTypes: int = 9):
        """
        Initialize the NER model - set up the squad

        Args:
            vocabSizeTotal: How many unique words we got
            embeddingDimensions: Size of word vectors (300 is Word2Vec standard)
            maxSeqLength: Max sentence length we handle
            numTagTypes: Number of entity types (9 for CoNLL2003)
        """
        self.vocabSizeTotal = vocabSizeTotal
        self.embeddingDimensions = embeddingDimensions
        self.maxSeqLength = maxSeqLength
        self.numTagTypes = numTagTypes
        self.neuralModel = None
        self.wordToIndexMap = {}
        self.indexToWordMap = {}
        self.tagToIndexMap = {}
        self.indexToTagMap = {}

    def buildTheModel(self, embeddingMatrixPretrained=None):
        """
        Build the LSTM model architecture - construct the beast

        Args:
            embeddingMatrixPretrained: Pre-trained Word2Vec embeddings (optional but fire)
        """
        neuralStack = Sequential()

        # Embedding layer (word -> vector conversion)
        if embeddingMatrixPretrained is not None:
            neuralStack.add(Embedding(
                input_dim=self.vocabSizeTotal,
                output_dim=self.embeddingDimensions,
                weights=[embeddingMatrixPretrained],
                input_length=self.maxSeqLength,
                trainable=False,  # keep the pretrained weights frozen
                mask_zero=True  # ignore padding
            ))
        else:
            neuralStack.add(Embedding(
                input_dim=self.vocabSizeTotal,
                output_dim=self.embeddingDimensions,
                input_length=self.maxSeqLength,
                mask_zero=True
            ))

        # LSTM Layer 1 (first memory unit, biggest one)
        neuralStack.add(LSTM(128, return_sequences=True, dropout=0.2))

        # LSTM Layer 2 (second memory unit, medium sized)
        neuralStack.add(LSTM(64, return_sequences=True, dropout=0.2))

        # LSTM Layer 3 (third memory unit, smallest but still fire)
        neuralStack.add(LSTM(32, return_sequences=True, dropout=0.2))

        # Dense layer (final processing before predictions)
        neuralStack.add(Dense(64, activation='relu'))
        neuralStack.add(Dropout(0.3))  # prevent overfitting, keep it real

        # Output layer (make predictions for each tag type)
        neuralStack.add(Dense(self.numTagTypes, activation='softmax'))

        # Compile the model (set up training parameters)
        neuralStack.compile(
            loss='categorical_crossentropy',  # loss function for multi-class
            optimizer='adam',  # Adam optimizer is goated
            metrics=['accuracy']
        )

        self.neuralModel = neuralStack
        return neuralStack

    def showModelStats(self):
        """Print model architecture - show what we workin with"""
        if self.neuralModel:
            self.neuralModel.summary()


def prepareNerDataset(datasetRaw, maxSamplesToUse=5000):
    """
    Prepare CoNLL2003 data for NER training - get the data ready

    Args:
        datasetRaw: CoNLL2003 dataset from HuggingFace
        maxSamplesToUse: Max number of samples (we aint training on everything, that takes forever)

    Returns:
        Tuple of (sentences, tags, word2idx, tag2idx) - all the data we need
    """
    sentencesList = []
    tagsList = []

    # Extract sentences and their entity tags
    trainingDataRaw = datasetRaw['train']
    numSamples = min(maxSamplesToUse, len(trainingDataRaw))

    for idx in range(numSamples):
        tokensLowercase = [token.lower() for token in trainingDataRaw[idx]['tokens']]
        nerTagSequence = trainingDataRaw[idx]['ner_tags']
        sentencesList.append(tokensLowercase)
        tagsList.append(nerTagSequence)

    # Build vocabulary mapping (create the word roster)
    allWordsUnique = set(word for sent in sentencesList for word in sent)
    wordToIndexDict = {word: idx + 2 for idx, word in enumerate(sorted(allWordsUnique))}
    wordToIndexDict['<PAD>'] = 0  # padding token (for making all sentences same length)
    wordToIndexDict['<UNK>'] = 1  # unknown token (for words we never seen)

    # Tag mapping (0-8 for 9 NER entity types)
    tagToIndexDict = {i: i for i in range(9)}

    return sentencesList, tagsList, wordToIndexDict, tagToIndexDict


def createEmbeddingMatrix(wordToIdxMap, word2vecModelLoaded, embeddingDims=300):
    """
    Create embedding matrix from Word2Vec model - convert our vocab to vectors

    Args:
        wordToIdxMap: Word to index mapping
        word2vecModelLoaded: Loaded Word2Vec model (Google News 300D)
        embeddingDims: Embedding dimensions (300 is standard)

    Returns:
        Embedding matrix (array of word vectors, ready to plug into model)
    """
    totalVocabSize = len(wordToIdxMap)
    embeddingMatrixFull = np.zeros((totalVocabSize, embeddingDims))

    wordsFoundCount = 0
    for word, wordIdx in wordToIdxMap.items():
        if word in word2vecModelLoaded:
            embeddingMatrixFull[wordIdx] = word2vecModelLoaded[word]
            wordsFoundCount += 1
        else:
            # Random init for words not in Word2Vec (gotta improvise)
            embeddingMatrixFull[wordIdx] = np.random.normal(0, 0.1, embeddingDims)

    coveragePercent = 100 * wordsFoundCount / totalVocabSize
    print(f"Found {wordsFoundCount}/{totalVocabSize} words in Word2Vec ({coveragePercent:.2f}% coverage - not bad!)")
    return embeddingMatrixFull


def questionThreeBigBrainTime():
    """
    Q3: Implement Named Entity Recognition using LSTM - the final boss
    """
    print("\n" + "="*80)
    print("QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM - BIG BRAIN TIME")
    print("="*80 + "\n")

    if not kerasIsVibing:
        print("Bruh, Keras/TensorFlow ain't installed. Install the packages first!")
        return None

    # Load the dataset
    print("Loading CoNLL2003 dataset... this the good stuff...")
    datasetMain = load_dataset("conll2003")

    # Prepare the data for training
    print("Preparing data... gettin it ready...")
    sentencesAll, tagsAll, wordToIdxMap, tagToIdxMap = prepareNerDataset(datasetMain, maxSamplesToUse=5000)
    idxToTagMap = {v: k for k, v in tagToIdxMap.items()}

    print(f"Number of sentences: {len(sentencesAll)} - we got mad data!")
    print(f"Vocabulary size: {len(wordToIdxMap)} - that's a lot of words")
    print(f"Number of NER tags: {len(tagToIdxMap)} - 9 entity types to predict")

    # Find the longest sentence
    maxLengthFound = max(len(sent) for sent in sentencesAll)
    maxLengthCapped = min(maxLengthFound, 100)  # cap at 100 for efficiency (aint nobody got time for super long sentences)
    print(f"Maximum sequence length: {maxLengthCapped}\n")

    # Convert sentences to number sequences (neural nets need numbers not words)
    sequencesX = []
    sequencesY = []

    for singleSent, singleTagSeq in zip(sentencesAll, tagsAll):
        # Turn words into their index numbers
        sentenceIndices = [wordToIdxMap.get(word, wordToIdxMap['<UNK>']) for word in singleSent]
        sequencesX.append(sentenceIndices)
        sequencesY.append(singleTagSeq)

    # Pad sequences so they all the same length (neural nets need uniform input)
    xPaddedArrays = pad_sequences(sequencesX, maxlen=maxLengthCapped, padding='post', value=wordToIdxMap['<PAD>'])
    yPaddedArrays = pad_sequences(sequencesY, maxlen=maxLengthCapped, padding='post', value=0)

    # Convert tags to one-hot encoding (categorical format for neural net)
    yCategoricalArrays = np.array([to_categorical(seq, num_classes=9) for seq in yPaddedArrays])

    # Split into train and test (80/20 split is the move)
    xTrainData, xTestData, yTrainData, yTestData = train_test_split(
        xPaddedArrays, yCategoricalArrays, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(xTrainData)} - this the main dataset")
    print(f"Testing samples: {len(xTestData)} - we holdin this back to test\n")

    # Load Word2Vec embeddings (this might take a minute first time)
    print("Loading Word2Vec embeddings (Google News 300D, this might take a bit first time)...")
    try:
        word2vecLoaded = api.load("word2vec-google-news-300")
        print("Word2Vec loaded successfully! We got the good embeddings\n")

        # Create embedding matrix from Word2Vec
        embeddingMatrixReady = createEmbeddingMatrix(wordToIdxMap, word2vecLoaded)
    except Exception as errorMsg:
        print(f"Yo, couldn't load Word2Vec: {errorMsg}")
        print("Using random embeddings instead - not ideal but we make it work\n")
        embeddingMatrixReady = None

    # Build the model
    print("Building LSTM model... constructin the beast...")
    nerBeastModel = NerModelBeast(
        vocabSizeTotal=len(wordToIdxMap),
        embeddingDimensions=300,
        maxSeqLength=maxLengthCapped,
        numTagTypes=9
    )
    nerBeastModel.wordToIndexMap = wordToIdxMap
    nerBeastModel.indexToTagMap = idxToTagMap
    nerBeastModel.buildTheModel(embeddingMatrixReady)
    nerBeastModel.showModelStats()

    # Train the model (this where the magic happens)
    print("\nTraining model (10 epochs)... let's get it...")
    trainingHistory = nerBeastModel.neuralModel.fit(
        xTrainData, yTrainData,
        validation_split=0.1,  # use 10% of training data for validation
        epochs=10,  # train for 10 epochs as required
        batch_size=32,  # process 32 samples at a time
        verbose=1  # show progress
    )

    # Evaluate the model on test data
    print("\nEvaluating model... moment of truth...")
    predictionsFull = nerBeastModel.neuralModel.predict(xTestData)
    predictedClasses = np.argmax(predictionsFull, axis=-1)
    trueClasses = np.argmax(yTestData, axis=-1)

    # Flatten predictions and true labels (remove padding)
    predictionsFlat = []
    truthFlat = []

    for sampleIdx in range(len(trueClasses)):
        for tokenIdx in range(len(trueClasses[sampleIdx])):
            if trueClasses[sampleIdx][tokenIdx] != 0 or tokenIdx < maxLengthCapped:
                predictionsFlat.append(predictedClasses[sampleIdx][tokenIdx])
                truthFlat.append(trueClasses[sampleIdx][tokenIdx])

    # Calculate performance metrics (see how we did)
    accuracyScore = accuracy_score(truthFlat, predictionsFlat)
    precisionScore, recallScore, f1Score, _ = precision_recall_fscore_support(
        truthFlat, predictionsFlat, average='macro', zero_division=0
    )

    print("\n" + "="*80)
    print("RESULTS - LET'S SEE HOW WE DID:")
    print("="*80)
    print(f"Accuracy: {accuracyScore:.4f} - overall correctness rate")
    print(f"Macro Precision: {precisionScore:.4f} - how precise our predictions are")
    print(f"Macro Recall: {recallScore:.4f} - how many entities we caught")
    print(f"Macro F1-Score: {f1Score:.4f} - the balanced score (precision + recall)")
    print("="*80 + "\n")

    return nerBeastModel, trainingHistory, (accuracyScore, precisionScore, recallScore, f1Score)


# ============================================================================
# MAIN EXECUTION - RUN EVERYTHING
# ============================================================================

def runEverything():
    """
    Main function to run all assignment questions - the whole show
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 15 + "CS 421: NLP - ASSIGNMENT 4 - LET'S GET IT" + " " * 21 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    resultsDict = {}

    # Question 1: TF-IDF and Cosine Similarity
    try:
        vectorizerResult, matrixResult = questionOneLettsGo()
        resultsDict['q1'] = {'vectorizer': vectorizerResult, 'matrix': matrixResult}
    except Exception as errorMsg:
        print(f"\nBruh, error in Q1: {errorMsg}")
        import traceback
        traceback.print_exc()

    # Question 2: PPMI
    try:
        ppmiResult1, ppmiResult2 = questionTwoShowtime()
        resultsDict['q2'] = {'ppmi1': ppmiResult1, 'ppmi2': ppmiResult2}
    except Exception as errorMsg:
        print(f"\nYo, error in Q2: {errorMsg}")
        import traceback
        traceback.print_exc()

    # Question 3: NER with LSTM
    try:
        nerModelFinal, historyObj, metricsResults = questionThreeBigBrainTime()
        resultsDict['q3'] = {
            'model': nerModelFinal,
            'history': historyObj,
            'metrics': metricsResults
        }
    except Exception as errorMsg:
        print(f"\nDang, error in Q3: {errorMsg}")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 80)
    print("*" + " " * 20 + "ASSIGNMENT COMPLETE! WE DID IT FAM!" + " " * 24 + "*")
    print("*" * 80 + "\n")

    return resultsDict


if __name__ == "__main__":
    finalResults = runEverything()
