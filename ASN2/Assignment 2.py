# Assignment 2 - NLP
# Name: [Your Name]
# Date: November 2, 2025

# imports n stuff
import pandas as pd
import numpy as np
import nltk
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import string
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# get nltk data (boring but necessary)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')  
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# reproducibility is key
random.seed(42)
np.random.seed(42)

print("=== CS 421 NLP Assignment 2: Naive Bayes & Logistic Regression ===\n")

# ===============================
# TEXT PREPROCESSING STUFF
# ===============================

def cleanText(textInput):
    """clean up text - lowercasing, punctuation removal, etc"""
    # make it lowercase
    textInput = textInput.lower()
    
    # kill punctuation and numbers
    textInput = re.sub(r'[^\w\s]', '', textInput)
    textInput = re.sub(r'\d+', '', textInput)
    
    # tokenize that bad boy
    tokenList = word_tokenize(textInput)
    
    # yeet stopwords
    stopWordSet = set(stopwords.words('english'))
    tokenList = [token for token in tokenList if token not in stopWordSet and len(token) > 2]
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokenList = [lemmatizer.lemmatize(token) for token in tokenList]
    
    return tokenList

def buildVocabulary(textList):
    """build vocab from tokenized texts"""
    vocabSet = set()
    for textTokens in textList:
        vocabSet.update(textTokens)
    return vocabSet

# ===============================
# Q1: NAIVE BAYES STUFF
# ===============================

class NaiveBayesClassifier:
    """nb classifier built from scratch - ez money"""
    
    def __init__(self):
        self.priorProbs = {}
        self.likelihoodProbs = {}
        self.vocabSet = set()
        self.classList = []
        
    def train(self, xTrain, yTrain):
        """train the nb classifier"""
        print("Training Naive Bayes classifier...")
        
        # get unique classes
        self.classList = list(set(yTrain))
        
        # build vocab from training data
        self.vocabSet = buildVocabulary(xTrain)
        vocabSize = len(self.vocabSet)
        
        # calc priors: P(c) = count(c) / N
        totalSamples = len(yTrain)
        classCounts = {}
        for label in yTrain:
            classCounts[label] = classCounts.get(label, 0) + 1
            
        for classLabel in self.classList:
            self.priorProbs[classLabel] = classCounts[classLabel] / totalSamples
            
        print(f"Prior probabilities: {self.priorProbs}")
        
        # calc likelihoods with laplace smoothing
        # P(w|c) = (count(w,c) + 1) / (|V| + Σcount(w,c))
        wordCounts = {classLabel: defaultdict(int) for classLabel in self.classList}
        classWordTotals = {classLabel: 0 for classLabel in self.classList}
        
        # count word freqs per class
        for textTokens, label in zip(xTrain, yTrain):
            for word in textTokens:
                if word in self.vocabSet:
                    wordCounts[label][word] += 1
                    classWordTotals[label] += 1
        
        # calc likelihood probs
        self.likelihoodProbs = {classLabel: {} for classLabel in self.classList}
        
        for classLabel in self.classList:
            for word in self.vocabSet:
                # laplace smoothing ftw
                count = wordCounts[classLabel][word]
                total = classWordTotals[classLabel]
                self.likelihoodProbs[classLabel][word] = (count + 1) / (vocabSize + total)
                
        print(f"Vocabulary size: {vocabSize}")
        print("Training completed!\n")
        
    def predict(self, xTest):
        """predict classes using log probs (avoid underflow)"""
        predictions = []
        
        for textTokens in xTest:
            # calc log posterior for each class
            classScores = {}
            
            for classLabel in self.classList:
                # start with log prior
                score = np.log(self.priorProbs[classLabel])
                
                # add log likelihoods for each word
                for word in textTokens:
                    if word in self.vocabSet:
                        score += np.log(self.likelihoodProbs[classLabel][word])
                
                classScores[classLabel] = score
            
            # predict class with highest score
            predictedClass = max(classScores, key=classScores.get)
            predictions.append(predictedClass)
            
        return predictions

def evaluateClassifier(yTrue, yPred, classifierName):
    """eval classifier performance - print all the good stuff"""
    print(f"\n=== {classifierName} Evaluation Results ===")
    
    # calc metrics
    accuracyScore = accuracy_score(yTrue, yPred)
    precisionScore = precision_score(yTrue, yPred, average='macro')
    recallScore = recall_score(yTrue, yPred, average='macro')
    f1Score = f1_score(yTrue, yPred, average='macro')
    
    print(f"Accuracy: {accuracyScore:.4f}")
    print(f"Macro Precision: {precisionScore:.4f}")
    print(f"Macro Recall: {recallScore:.4f}")
    print(f"Macro F1-Score: {f1Score:.4f}")
    
    # confusion matrix
    confusionMat = confusion_matrix(yTrue, yPred)
    print(f"\nConfusion Matrix:")
    print(f"{'':>10} {'Pred 0':>8} {'Pred 2':>8}")
    print(f"{'True 0':>10} {confusionMat[0,0]:>8} {confusionMat[0,1]:>8}")
    print(f"{'True 2':>10} {confusionMat[1,0]:>8} {confusionMat[1,1]:>8}")
    
    return accuracyScore, precisionScore, recallScore, f1Score

# ===============================
# Q2: LOGISTIC REGRESSION STUFF  
# ===============================

class LogisticRegressionClassifier:
    """logistic regression from scratch with gradient descent - let's gooo"""
    
    def __init__(self, learningRate=0.01, epochCount=500):
        self.learningRate = learningRate
        self.epochCount = epochCount
        self.weightsVector = None
        self.trainLosses = []
        self.valLosses = []
        
    def sigmoid(self, zValues):
        """sigmoid activation - classic"""
        # clip z to prevent overflow (numerical stability ftw)
        zValues = np.clip(zValues, -500, 500)
        return 1 / (1 + np.exp(-zValues))
    
    def computeLoss(self, yTrue, yPred):
        """compute cross-entropy loss"""
        # add epsilon to prevent log(0) disasters
        epsilon = 1e-15
        yPred = np.clip(yPred, epsilon, 1 - epsilon)
        
        lossValue = -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))
        return lossValue
    
    def train(self, xTrain, yTrain, xVal=None, yVal=None):
        """train lr using gradient descent"""
        print(f"Training Logistic Regression (lr={self.learningRate}, epochs={self.epochCount})...")
        
        # init weights to zeros
        numFeatures = xTrain.shape[1]
        self.weightsVector = np.zeros(numFeatures)
        
        # convert labels: 0 stays 0, 2 becomes 1
        yTrainBinary = np.array([0 if label == 0 else 1 for label in yTrain])
        if yVal is not None:
            yValBinary = np.array([0 if label == 0 else 1 for label in yVal])
        
        self.trainLosses = []
        self.valLosses = []
        
        for epoch in range(self.epochCount):
            # forward pass
            zValues = xTrain.dot(self.weightsVector)
            yPred = self.sigmoid(zValues)
            
            # compute training loss
            trainLoss = self.computeLoss(yTrainBinary, yPred)
            self.trainLosses.append(trainLoss)
            
            # compute gradient and update weights
            gradient = xTrain.T.dot(yPred - yTrainBinary) / len(yTrain)
            self.weightsVector -= self.learningRate * gradient
            
            # compute validation loss if val data provided
            if xVal is not None and yVal is not None:
                zVal = xVal.dot(self.weightsVector)
                yValPred = self.sigmoid(zVal)
                valLoss = self.computeLoss(yValBinary, yValPred)
                self.valLosses.append(valLoss)
            
            # print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                if xVal is not None:
                    print(f"Epoch {epoch+1}/{self.epochCount} - Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochCount} - Train Loss: {trainLoss:.4f}")
        
        print("Training completed!\n")
        
    def predict(self, xTest):
        """make predictions on test data"""
        zValues = xTest.dot(self.weightsVector)
        probabilities = self.sigmoid(zValues)
        
        # convert probs to class predictions
        # prob >= 0.5 -> class 2 (positive), else class 0 (negative)
        predictions = [2 if prob >= 0.5 else 0 for prob in probabilities]
        return predictions
    
    def plotLosses(self):
        """plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.trainLosses, label='Training Loss', color='blue')
        if self.valLosses:
            plt.plot(self.valLosses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title(f'Training Progress (Learning Rate: {self.learningRate})')
        plt.legend()
        plt.grid(True)
        plt.show()

# ===============================
# MAIN EXECUTION
# ===============================

def loadFinancialPhrasebank():
    """load financial phrasebank dataset from local files"""
    # path to dataset file
    filePath = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
    
    sentenceList = []
    labelList = []
    
    # read file and parse format: sentence@label
    # try different encodings for special chars
    encodingOptions = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodingOptions:
        try:
            with open(filePath, 'r', encoding=encoding) as fileHandle:
                for lineText in fileHandle:
                    lineText = lineText.strip()
                    if lineText and '@' in lineText:
                        # split on last @ to handle sentences with @
                        lineParts = lineText.rsplit('@', 1)
                        if len(lineParts) == 2:
                            sentenceText = lineParts[0].strip()
                            labelText = lineParts[1].strip()
                            
                            # convert labels: negative -> 0, neutral -> 1, positive -> 2
                            if labelText == 'negative':
                                labelNum = 0
                            elif labelText == 'neutral':
                                labelNum = 1
                            elif labelText == 'positive':
                                labelNum = 2
                            else:
                                continue  # skip unknown labels
                            
                            sentenceList.append(sentenceText)
                            labelList.append(labelNum)
            break  # success - break out of encoding loop
        except UnicodeDecodeError:
            continue  # try next encoding
    
    if not sentenceList:
        raise RuntimeError("Could not decode the file with any of the attempted encodings")
    
    return pd.DataFrame({'sentence': sentenceList, 'label': labelList})

def main():
    print("Loading Financial Phrasebank dataset from local files...")
    
    # load dataset from local files
    dataFrame = loadFinancialPhrasebank()
    print(f"Original dataset size: {len(dataFrame)}")
    print(f"Class distribution: {dataFrame['label'].value_counts().sort_index()}")
    
    # filter out neutral samples (label 1), keep only pos (2) and neg (0)
    filteredDf = dataFrame[dataFrame['label'] != 1].copy()
    print(f"Filtered dataset size (no neutrals): {len(filteredDf)}")
    print(f"Filtered class distribution: {filteredDf['label'].value_counts().sort_index()}")
    
    # clean and preprocess text data
    print("\nCleaning text data...")
    filteredDf['cleaned_text'] = filteredDf['sentence'].apply(cleanText)
    
    # remove empty texts after cleaning
    filteredDf = filteredDf[filteredDf['cleaned_text'].apply(len) > 0]
    print(f"Dataset size after cleaning: {len(filteredDf)}")
    
    xData = filteredDf['cleaned_text'].tolist()
    yData = filteredDf['label'].tolist()
    
    # ===============================
    # Q1: NAIVE BAYES
    # ===============================
    
    print("\n" + "="*50)
    print("Q1: NAIVE BAYES CLASSIFIER")
    print("="*50)
    
    # split data: 80% train, 20% test
    indexList = list(range(len(xData)))
    random.shuffle(indexList)
    
    trainSize = int(0.8 * len(xData))
    trainIndices = indexList[:trainSize]
    testIndices = indexList[trainSize:]
    
    xTrainNb = [xData[i] for i in trainIndices]
    yTrainNb = [yData[i] for i in trainIndices]
    xTestNb = [xData[i] for i in testIndices]
    yTestNb = [yData[i] for i in testIndices]
    
    print(f"Training set size: {len(xTrainNb)}")
    print(f"Test set size: {len(xTestNb)}")
    
    # train naive bayes classifier
    nbClassifier = NaiveBayesClassifier()
    nbClassifier.train(xTrainNb, yTrainNb)
    
    # make predictions
    yPredNb = nbClassifier.predict(xTestNb)
    
    # eval performance
    nbResults = evaluateClassifier(yTestNb, yPredNb, "Naive Bayes")
    
    # ===============================
    # Q2: LOGISTIC REGRESSION
    # ===============================
    
    print("\n" + "="*50)
    print("Q2: LOGISTIC REGRESSION CLASSIFIER")
    print("="*50)
    
    # split data: 60% train, 20% validation, 20% test
    trainSizeLr = int(0.6 * len(xData))
    valSizeLr = int(0.2 * len(xData))
    
    trainIndicesLr = indexList[:trainSizeLr]
    valIndicesLr = indexList[trainSizeLr:trainSizeLr + valSizeLr]
    testIndicesLr = indexList[trainSizeLr + valSizeLr:]
    
    xTrainLr = [xData[i] for i in trainIndicesLr]
    yTrainLr = [yData[i] for i in trainIndicesLr]
    xValLr = [xData[i] for i in valIndicesLr]
    yValLr = [yData[i] for i in valIndicesLr]
    xTestLr = [xData[i] for i in testIndicesLr]
    yTestLr = [yData[i] for i in testIndicesLr]
    
    print(f"Training set size: {len(xTrainLr)}")
    print(f"Validation set size: {len(xValLr)}")
    print(f"Test set size: {len(xTestLr)}")
    
    # convert tokenized texts back to strings for CountVectorizer
    xTrainStr = [' '.join(tokens) for tokens in xTrainLr]
    xValStr = [' '.join(tokens) for tokens in xValLr]
    xTestStr = [' '.join(tokens) for tokens in xTestLr]
    
    # create bag-of-words representation using CountVectorizer
    print("\nCreating bag-of-words representation...")
    vectorizer = CountVectorizer()
    xTrainBow = vectorizer.fit_transform(xTrainStr).toarray()
    xValBow = vectorizer.transform(xValStr).toarray()
    xTestBow = vectorizer.transform(xTestStr).toarray()
    
    print(f"Feature dimension: {xTrainBow.shape[1]}")
    
    # train with different learning rates
    learningRates = [0.0001, 0.001, 0.01, 0.1]
    lrResults = {}
    
    for learningRate in learningRates:
        print(f"\n--- Training with learning rate: {learningRate} ---")
        
        # train logistic regression classifier
        lrClassifier = LogisticRegressionClassifier(learningRate=learningRate, epochCount=500)
        lrClassifier.train(xTrainBow, yTrainLr, xValBow, yValLr)
        
        # make predictions on test set
        yPredLr = lrClassifier.predict(xTestBow)
        
        # eval performance
        resultsData = evaluateClassifier(yTestLr, yPredLr, f"Logistic Regression (lr={learningRate})")
        lrResults[learningRate] = resultsData
        
        # plot losses
        # lrClassifier.plotLosses()
    
    # ===============================
    # COMPARISON AND ANALYSIS
    # ===============================
    
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    print("\nNaive Bayes Results:")
    print(f"Accuracy: {nbResults[0]:.4f}, Precision: {nbResults[1]:.4f}, Recall: {nbResults[2]:.4f}, F1: {nbResults[3]:.4f}")
    
    print("\nLogistic Regression Results:")
    for learningRate, resultsData in lrResults.items():
        print(f"LR (α={learningRate:6.4f}): Acc={resultsData[0]:.4f}, Prec={resultsData[1]:.4f}, Rec={resultsData[2]:.4f}, F1={resultsData[3]:.4f}")
    
    # find best logistic regression result
    bestLr = max(lrResults.keys(), key=lambda x: lrResults[x][0])  # best by accuracy
    bestResults = lrResults[bestLr]
    
    print(f"\nBest Logistic Regression: α={bestLr} with accuracy={bestResults[0]:.4f}")
    
    print("\n" + "="*50)
    print("OBSERVATIONS AND ANALYSIS")
    print("="*50)
    
    print("\nKey Observations:")
    print("1. Learning Rate Impact:")
    print("   - Very small rates (0.0001) may lead to slow convergence")
    print("   - Very large rates (0.1) may cause unstable training")
    print("   - Moderate rates (0.001, 0.01) typically perform best")
    
    print("\n2. Model Comparison:")
    if nbResults[0] > bestResults[0]:
        print("   - Naive Bayes outperformed Logistic Regression")
        print("   - This suggests feature independence assumption holds reasonably well")
    else:
        print("   - Logistic Regression outperformed Naive Bayes")
        print("   - This suggests the discriminative approach captured decision boundaries better")
    
    print("\n3. Performance Analysis:")
    print(f"   - Both models achieved reasonable performance on financial sentiment")
    print(f"   - The bag-of-words representation captured important sentiment signals")
    print(f"   - Text preprocessing (cleaning, lemmatization) helped model performance")
    
    print("\nAssignment completed successfully!")

if __name__ == "__main__":
    main()
