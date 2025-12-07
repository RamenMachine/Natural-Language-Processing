"""
Question 2: LSTM Slot Filling for ATIS Dataset
==============================================

This implements an LSTM-based slot filler that takes in sentences
and tags each word with the appropriate slot label (like location, date, etc).
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# using keras cause it's easier than raw pytorch
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score


class SlotFillingModel:
    """
    LSTM-based slot filling model for the ATIS dataset.

    Takes in tokenized sentences and predicts slot labels for each word.
    """

    def __init__(self):
        """Initialize the model with empty vocabs and params."""
        self.wordToIdx = {}
        self.idxToWord = {}
        self.slotToIdx = {}
        self.idxToSlot = {}
        self.maxSeqLen = 0
        self.model = None

    def loadData(self, trainPath, valPath, testPath):
        """
        Load the ATIS data from CSV files.

        Returns three dataframes for train, val, and test.
        """
        print("Loading ATIS data files...")

        # read the csv files
        # the files have BOS and EOS tokens we need to handle
        trainDf = pd.read_csv(trainPath)
        valDf = pd.read_csv(valPath)
        testDf = pd.read_csv(testPath)

        print(f"Train: {len(trainDf)} samples")
        print(f"Val: {len(valDf)} samples")
        print(f"Test: {len(testDf)} samples")

        return trainDf, valDf, testDf

    def buildVocabs(self, trainDf):
        """
        Build vocabularies from the training data.

        Creates word vocab and slot label vocab.
        """
        print("\nBuilding vocabularies...")

        # collect all words and slots
        allWords = []
        allSlots = []

        for idx, row in trainDf.iterrows():
            # tokens and slots are space-separated
            tokens = row['tokens'].split()
            slots = row['slots'].split()

            allWords.extend(tokens)
            allSlots.extend(slots)

        # get unique words and slots
        uniqueWords = sorted(list(set(allWords)))
        uniqueSlots = sorted(list(set(allSlots)))

        # build word mappings (start at 1, reserve 0 for padding)
        self.wordToIdx = {word: idx + 1 for idx, word in enumerate(uniqueWords)}
        self.idxToWord = {idx + 1: word for idx, word in enumerate(uniqueWords)}

        # add padding and unknown tokens
        self.wordToIdx['<PAD>'] = 0
        self.wordToIdx['<UNK>'] = len(self.wordToIdx)
        self.idxToWord[0] = '<PAD>'
        self.idxToWord[len(self.idxToWord)] = '<UNK>'

        # build slot mappings (start at 0)
        self.slotToIdx = {slot: idx for idx, slot in enumerate(uniqueSlots)}
        self.idxToSlot = {idx: slot for idx, slot in enumerate(uniqueSlots)}

        print(f"Vocab size: {len(self.wordToIdx)} words")
        print(f"Slot labels: {len(self.slotToIdx)} unique slots")

        return self.wordToIdx, self.slotToIdx

    def preprocessData(self, dataframe, isTrain=False):
        """
        Convert text data to sequences of indices.

        Returns padded sequences for both words and slots.
        """
        wordSequences = []
        slotSequences = []

        # process each row
        for idx, row in dataframe.iterrows():
            tokens = row['tokens'].split()
            slots = row['slots'].split()

            # convert words to indices
            wordIdxs = []
            for word in tokens:
                if word in self.wordToIdx:
                    wordIdxs.append(self.wordToIdx[word])
                else:
                    # unknown word - use UNK token
                    wordIdxs.append(self.wordToIdx['<UNK>'])

            # convert slots to indices
            slotIdxs = [self.slotToIdx[slot] for slot in slots]

            wordSequences.append(wordIdxs)
            slotSequences.append(slotIdxs)

        # find max length if this is training data
        if isTrain:
            self.maxSeqLen = max(len(seq) for seq in wordSequences)
            print(f"Max sequence length: {self.maxSeqLen}")

        # pad all sequences to the same length
        paddedWords = pad_sequences(wordSequences, maxlen=self.maxSeqLen, padding='post', value=0)
        paddedSlots = pad_sequences(slotSequences, maxlen=self.maxSeqLen, padding='post', value=0)

        return paddedWords, paddedSlots

    def buildModel(self, vocabSize, numSlots, embeddingDim=100, lstmUnits=128):
        """
        Build the LSTM architecture for slot filling.

        Architecture:
        Embedding -> BiLSTM -> BiLSTM -> Dense -> TimeDistributed -> Softmax
        """
        print("\nBuilding LSTM model...")

        model = Sequential()

        # embedding layer converts word indices to dense vectors
        model.add(Embedding(input_dim=vocabSize, output_dim=embeddingDim, mask_zero=True))

        # bidirectional LSTM layers
        # going both forward and backward helps capture context
        model.add(Bidirectional(LSTM(lstmUnits, return_sequences=True)))
        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(lstmUnits // 2, return_sequences=True)))
        model.add(Dropout(0.3))

        # time distributed dense layer applies same dense layer to each time step
        model.add(TimeDistributed(Dense(128, activation='relu')))

        # output layer - softmax over all slot labels
        model.add(TimeDistributed(Dense(numSlots, activation='softmax')))

        # compile with adam optimizer and categorical crossentropy
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        print(model.summary())
        self.model = model
        return model

    def train(self, trainWords, trainSlots, valWords, valSlots, epochs=15, batchSize=32):
        """
        Train the model on the training data.

        Uses validation data to monitor performance.
        """
        print("\nTraining the model...")

        # train the model
        history = self.model.fit(
            trainWords, trainSlots,
            validation_data=(valWords, valSlots),
            epochs=epochs,
            batch_size=batchSize,
            verbose=1
        )

        return history

    def evaluate(self, testWords, testSlots):
        """
        Evaluate the model on test data.

        Returns precision, recall, and F-score.
        """
        print("\nEvaluating on test set...")

        # get predictions
        predictions = self.model.predict(testWords)
        predictedLabels = np.argmax(predictions, axis=-1)

        # flatten the arrays for sklearn metrics
        # ignore padding positions (where true label is 0)
        trueLabelsFlat = []
        predLabelsFlat = []

        for i in range(len(testSlots)):
            for j in range(len(testSlots[i])):
                # skip padding
                if testSlots[i][j] != 0:
                    trueLabelsFlat.append(testSlots[i][j])
                    predLabelsFlat.append(predictedLabels[i][j])

        # calculate metrics
        precision = precision_score(trueLabelsFlat, predLabelsFlat, average='weighted', zero_division=0)
        recall = recall_score(trueLabelsFlat, predLabelsFlat, average='weighted', zero_division=0)
        f1 = f1_score(trueLabelsFlat, predLabelsFlat, average='weighted', zero_division=0)

        print(f"\nTest Set Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return precision, recall, f1

    def predictSlots(self, sentence):
        """
        Predict slots for a new sentence.

        Takes a string and returns the predicted slot labels.
        """
        # tokenize the sentence
        tokens = sentence.split()

        # convert to indices
        wordIdxs = []
        for word in tokens:
            if word in self.wordToIdx:
                wordIdxs.append(self.wordToIdx[word])
            else:
                wordIdxs.append(self.wordToIdx['<UNK>'])

        # pad to max length
        paddedSeq = pad_sequences([wordIdxs], maxlen=self.maxSeqLen, padding='post', value=0)

        # get prediction
        prediction = self.model.predict(paddedSeq, verbose=0)
        predictedIdxs = np.argmax(prediction[0], axis=-1)

        # convert back to slot labels
        predictedSlots = [self.idxToSlot[idx] for idx in predictedIdxs[:len(tokens)]]

        return list(zip(tokens, predictedSlots))


def main():
    """Run the complete slot filling pipeline."""
    print("="*60)
    print("Question 2: LSTM Slot Filling")
    print("="*60)

    # initialize the model
    slotFiller = SlotFillingModel()

    # load the data
    trainDf, valDf, testDf = slotFiller.loadData(
        'atis.train(1).csv',
        'atis.val(1).csv',
        'atis.test(1).csv'
    )

    # build vocabularies from training data
    slotFiller.buildVocabs(trainDf)

    # preprocess all datasets
    trainWords, trainSlots = slotFiller.preprocessData(trainDf, isTrain=True)
    valWords, valSlots = slotFiller.preprocessData(valDf, isTrain=False)
    testWords, testSlots = slotFiller.preprocessData(testDf, isTrain=False)

    # build the model
    vocabSize = len(slotFiller.wordToIdx)
    numSlots = len(slotFiller.slotToIdx)
    slotFiller.buildModel(vocabSize, numSlots)

    # train the model
    slotFiller.train(trainWords, trainSlots, valWords, valSlots, epochs=10)

    # evaluate on test set
    precision, recall, f1 = slotFiller.evaluate(testWords, testSlots)

    # test on a few examples
    print("\n" + "="*60)
    print("Testing on Example Sentences")
    print("="*60)

    testSentences = [
        "BOS show me flights from boston to denver EOS",
        "BOS what is the cheapest fare from dallas to atlanta EOS",
        "BOS i need a flight on american airlines EOS"
    ]

    for sentence in testSentences:
        print(f"\nInput: {sentence}")
        results = slotFiller.predictSlots(sentence)
        print("Predictions:")
        for word, slot in results:
            print(f"  {word:20s} -> {slot}")

    print("\n" + "="*60)
    print("Slot Filling Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
