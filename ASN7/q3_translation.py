"""
Question 3: Neural Machine Translation (German -> English)
==========================================================

This implements a sequence-to-sequence model with attention
for translating German sentences to English.

Uses the WMT14 dataset from Huggingface.
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# tensorflow for the neural network
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# for BLEU score evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# huggingface datasets for WMT14
from datasets import load_dataset


class NeuralTranslator:
    """
    Encoder-Decoder with Attention for German->English translation.

    Uses LSTM layers and an attention mechanism to translate sentences.
    """

    def __init__(self):
        """Initialize with empty vocabs and parameters."""
        self.germanWordToIdx = {}
        self.germanIdxToWord = {}
        self.englishWordToIdx = {}
        self.englishIdxToWord = {}
        self.maxGermanLen = 0
        self.maxEnglishLen = 0
        self.encoderModel = None
        self.decoderModel = None
        self.fullModel = None

    def loadWmt14Data(self, numSamples=10000):
        """
        Load WMT14 dataset from Huggingface.

        numSamples: how many samples to use (full dataset is huge)
        Returns: dataframes for train, val, test
        """
        print("Loading WMT14 dataset from Huggingface...")
        print("This might take a few minutes on first run...")

        # load the dataset (de-en configuration)
        # we're using a subset cause the full thing is massive
        dataset = load_dataset('wmt14', 'de-en', split='train', streaming=True)

        # grab the samples
        germanSentences = []
        englishSentences = []

        for idx, sample in enumerate(dataset):
            if idx >= numSamples:
                break

            # the translation field has both languages
            germanSentences.append(sample['translation']['de'])
            englishSentences.append(sample['translation']['en'])

            if (idx + 1) % 1000 == 0:
                print(f"Loaded {idx + 1} samples...")

        # create dataframe
        df = pd.DataFrame({
            'german': germanSentences,
            'english': englishSentences
        })

        # split into train/val/test (80/10/10)
        trainSize = int(0.8 * len(df))
        valSize = int(0.1 * len(df))

        trainDf = df[:trainSize]
        valDf = df[trainSize:trainSize + valSize]
        testDf = df[trainSize + valSize:]

        print(f"\nTrain: {len(trainDf)} samples")
        print(f"Val: {len(valDf)} samples")
        print(f"Test: {len(testDf)} samples")

        return trainDf, valDf, testDf

    def buildVocabs(self, trainDf, maxVocabSize=10000):
        """
        Build vocabularies for German and English.

        Keeps only the most common words to limit vocab size.
        """
        print("\nBuilding vocabularies...")

        # tokenize and count words
        germanWords = []
        englishWords = []

        for idx, row in trainDf.iterrows():
            # simple tokenization - just split on spaces
            germanTokens = row['german'].lower().split()
            englishTokens = row['english'].lower().split()

            germanWords.extend(germanTokens)
            englishWords.extend(englishTokens)

        # get most common words
        germanCounter = Counter(germanWords)
        englishCounter = Counter(englishWords)

        # keep top words
        topGerman = [word for word, count in germanCounter.most_common(maxVocabSize - 3)]
        topEnglish = [word for word, count in englishCounter.most_common(maxVocabSize - 3)]

        # build German vocab (reserve 0 for padding, 1 for UNK)
        self.germanWordToIdx = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(topGerman):
            self.germanWordToIdx[word] = idx + 2
        self.germanIdxToWord = {idx: word for word, idx in self.germanWordToIdx.items()}

        # build English vocab (add start and end tokens)
        self.englishWordToIdx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        for idx, word in enumerate(topEnglish):
            self.englishWordToIdx[word] = idx + 4
        self.englishIdxToWord = {idx: word for word, idx in self.englishWordToIdx.items()}

        print(f"German vocab: {len(self.germanWordToIdx)} words")
        print(f"English vocab: {len(self.englishWordToIdx)} words")

    def preprocessData(self, dataframe, isTrain=False):
        """
        Convert sentences to sequences of indices.

        Adds <START> and <END> tokens to English sentences.
        """
        germanSequences = []
        englishSequences = []

        for idx, row in dataframe.iterrows():
            # tokenize
            germanTokens = row['german'].lower().split()
            englishTokens = row['english'].lower().split()

            # convert German to indices
            germanIdxs = []
            for word in germanTokens:
                if word in self.germanWordToIdx:
                    germanIdxs.append(self.germanWordToIdx[word])
                else:
                    germanIdxs.append(self.germanWordToIdx['<UNK>'])

            # convert English to indices (add START and END)
            englishIdxs = [self.englishWordToIdx['<START>']]
            for word in englishTokens:
                if word in self.englishWordToIdx:
                    englishIdxs.append(self.englishWordToIdx[word])
                else:
                    englishIdxs.append(self.englishWordToIdx['<UNK>'])
            englishIdxs.append(self.englishWordToIdx['<END>'])

            germanSequences.append(germanIdxs)
            englishSequences.append(englishIdxs)

        # find max lengths if training
        if isTrain:
            self.maxGermanLen = max(len(seq) for seq in germanSequences)
            self.maxEnglishLen = max(len(seq) for seq in englishSequences)
            print(f"Max German length: {self.maxGermanLen}")
            print(f"Max English length: {self.maxEnglishLen}")

        # pad sequences
        paddedGerman = pad_sequences(germanSequences, maxlen=self.maxGermanLen, padding='post')
        paddedEnglish = pad_sequences(englishSequences, maxlen=self.maxEnglishLen, padding='post')

        return paddedGerman, paddedEnglish

    def buildModel(self, embeddingDim=100, hiddenUnits=128):
        """
        Build encoder-decoder with attention mechanism.

        Architecture:
        Encoder: Embedding -> LSTM
        Decoder: Embedding -> LSTM -> Attention -> Dense -> Softmax
        """
        print("\nBuilding encoder-decoder model with attention...")

        germanVocabSize = len(self.germanWordToIdx)
        englishVocabSize = len(self.englishWordToIdx)

        # ===== ENCODER =====
        encoderInputs = Input(shape=(self.maxGermanLen,))
        encoderEmbedding = Embedding(germanVocabSize, embeddingDim, mask_zero=True)(encoderInputs)

        # encoder LSTM returns sequences and states
        encoderLstm = LSTM(hiddenUnits, return_sequences=True, return_state=True)
        encoderOutputs, stateH, stateC = encoderLstm(encoderEmbedding)
        encoderStates = [stateH, stateC]

        # ===== DECODER =====
        decoderInputs = Input(shape=(self.maxEnglishLen,))
        decoderEmbedding = Embedding(englishVocabSize, embeddingDim, mask_zero=True)(decoderInputs)

        # decoder LSTM uses encoder states as initial state
        decoderLstm = LSTM(hiddenUnits, return_sequences=True, return_state=True)
        decoderOutputs, _, _ = decoderLstm(decoderEmbedding, initial_state=encoderStates)

        # ===== ATTENTION =====
        # attention layer looks at encoder outputs and decoder outputs
        attentionLayer = Attention()
        attentionResult = attentionLayer([decoderOutputs, encoderOutputs])

        # concatenate attention with decoder outputs
        decoderCombined = Concatenate(axis=-1)([decoderOutputs, attentionResult])

        # ===== OUTPUT =====
        # final dense layer predicts next word
        decoderDense = Dense(englishVocabSize, activation='softmax')
        decoderOutputs = decoderDense(decoderCombined)

        # build the full model
        self.fullModel = Model([encoderInputs, decoderInputs], decoderOutputs)

        # compile with adam and sparse categorical crossentropy
        self.fullModel.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.fullModel.summary())

        return self.fullModel

    def train(self, trainGerman, trainEnglish, valGerman, valEnglish, epochs=10, batchSize=64):
        """
        Train the translation model.

        The decoder input is the English sentence, and the target is
        the English sentence shifted by one position.
        """
        print("\nTraining the model...")

        # prepare decoder targets (shift by one position)
        # we want to predict the next word at each step
        decoderTargets = np.zeros_like(trainEnglish)
        decoderTargets[:, :-1] = trainEnglish[:, 1:]

        valDecoderTargets = np.zeros_like(valEnglish)
        valDecoderTargets[:, :-1] = valEnglish[:, 1:]

        # reshape targets for sparse_categorical_crossentropy
        decoderTargets = np.expand_dims(decoderTargets, -1)
        valDecoderTargets = np.expand_dims(valDecoderTargets, -1)

        # train the model
        history = self.fullModel.fit(
            [trainGerman, trainEnglish],
            decoderTargets,
            validation_data=([valGerman, valEnglish], valDecoderTargets),
            batch_size=batchSize,
            epochs=epochs,
            verbose=1
        )

        return history

    def translate(self, germanSentence):
        """
        Translate a German sentence to English.

        Uses greedy decoding - picks most likely word at each step.
        """
        # tokenize and convert to indices
        tokens = germanSentence.lower().split()
        germanIdxs = []
        for word in tokens:
            if word in self.germanWordToIdx:
                germanIdxs.append(self.germanWordToIdx[word])
            else:
                germanIdxs.append(self.germanWordToIdx['<UNK>'])

        # pad to max length
        germanSeq = pad_sequences([germanIdxs], maxlen=self.maxGermanLen, padding='post')

        # start with START token
        decoderInput = np.zeros((1, self.maxEnglishLen))
        decoderInput[0, 0] = self.englishWordToIdx['<START>']

        # generate word by word
        translatedWords = []

        for i in range(1, self.maxEnglishLen):
            # predict next word
            predictions = self.fullModel.predict([germanSeq, decoderInput], verbose=0)

            # get most likely word at current position
            predictedIdx = np.argmax(predictions[0, i - 1, :])
            predictedWord = self.englishIdxToWord.get(predictedIdx, '<UNK>')

            # check if we hit the end token
            if predictedWord == '<END>':
                break

            translatedWords.append(predictedWord)

            # add predicted word to decoder input for next step
            decoderInput[0, i] = predictedIdx

        return ' '.join(translatedWords)

    def evaluateBleu(self, testGerman, testEnglish):
        """
        Calculate average BLEU score on test set.

        BLEU measures how similar the translation is to the reference.
        """
        print("\nCalculating BLEU scores on test set...")

        bleuScores = []
        smoothingFunc = SmoothingFunction().method1

        for i in range(len(testGerman)):
            # get the German sentence
            germanIdxs = testGerman[i]
            germanWords = [self.germanIdxToWord.get(idx, '') for idx in germanIdxs if idx != 0]
            germanSentence = ' '.join(germanWords)

            # get reference English
            englishIdxs = testEnglish[i]
            referenceWords = [self.englishIdxToWord.get(idx, '') for idx in englishIdxs
                            if idx not in [0, 2, 3]]  # skip PAD, START, END

            # translate
            translation = self.translate(germanSentence)
            candidateWords = translation.split()

            # calculate BLEU score
            if candidateWords and referenceWords:
                bleuScore = sentence_bleu([referenceWords], candidateWords,
                                         smoothing_function=smoothingFunc)
                bleuScores.append(bleuScore)

            # show progress
            if (i + 1) % 100 == 0:
                print(f"Evaluated {i + 1}/{len(testGerman)} samples...")

        avgBleu = np.mean(bleuScores)
        print(f"\nAverage BLEU Score: {avgBleu:.4f}")

        return avgBleu


def main():
    """Run the complete translation pipeline."""
    print("="*60)
    print("Question 3: Neural Machine Translation (German -> English)")
    print("="*60)

    # initialize translator
    translator = NeuralTranslator()

    # load WMT14 data (using subset for faster training)
    # for full training, increase numSamples to 100000+
    trainDf, valDf, testDf = translator.loadWmt14Data(numSamples=5000)

    # build vocabularies
    translator.buildVocabs(trainDf)

    # preprocess data
    trainGerman, trainEnglish = translator.preprocessData(trainDf, isTrain=True)
    valGerman, valEnglish = translator.preprocessData(valDf)
    testGerman, testEnglish = translator.preprocessData(testDf)

    # build model
    translator.buildModel()

    # train the model
    translator.train(trainGerman, trainEnglish, valGerman, valEnglish, epochs=8)

    # test on some examples
    print("\n" + "="*60)
    print("Testing Translations")
    print("="*60)

    testSentences = [
        "das ist ein Test",
        "ich liebe maschinelles Lernen",
        "guten Morgen"
    ]

    for sentence in testSentences:
        translation = translator.translate(sentence)
        print(f"\nGerman: {sentence}")
        print(f"English: {translation}")

    # calculate BLEU score
    # using small subset for demo - would take longer on full test set
    smallTestGerman = testGerman[:100]
    smallTestEnglish = testEnglish[:100]
    bleuScore = translator.evaluateBleu(smallTestGerman, smallTestEnglish)

    print("\n" + "="*60)
    print("Translation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
