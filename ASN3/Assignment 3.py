import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import treebank
import random
from sklearn_crfsuite import CRF
import string

nltk.download('punkt', quiet=True)
nltk.download('treebank', quiet=True)
nltk.download('universal_tagset', quiet=True)

random.seed(42)

# ============================================================================
# QUESTION 1: TEXT GENERATION USING N-GRAMS (BIGRAMS)
# ============================================================================


def loadGreatGatsbyText(filePath):
    textLines = []
    inputFilePath = filePath
    encodingType = 'utf-8'
    fileMode = 'r'
    fileHandle = open(inputFilePath, fileMode, encoding=encodingType)

    singleLine = fileHandle.readline()
    while singleLine:
        currentLine = singleLine
        strippedLine = currentLine.strip()
        lineLength = len(strippedLine)
        zeroValue = 0
        isNotEmpty = lineLength > zeroValue

        if isNotEmpty:
            textLines.append(strippedLine)

        singleLine = fileHandle.readline()

    fileHandle.close()

    spaceCharacter = ' '
    fullText = spaceCharacter.join(textLines)
    returnValue = fullText
    return returnValue


def preprocessText(rawText):
    inputText = rawText
    sentenceList = sent_tokenize(inputText)

    processedSentences = []
    emptyList = []
    processedSentences = emptyList

    sentenceCounter = 0
    totalSentences = len(sentenceList)
    maxSentences = totalSentences

    while sentenceCounter < maxSentences:
        currentSentenceIndex = sentenceCounter
        singleSentence = sentenceList[currentSentenceIndex]
        wordTokens = word_tokenize(singleSentence)

        cleanedTokens = []
        tokenCounter = 0
        numTokens = len(wordTokens)

        while tokenCounter < numTokens:
            currentTokenIndex = tokenCounter
            singleToken = wordTokens[currentTokenIndex]

            punctuationSet = string.punctuation
            isPunctuation = singleToken in punctuationSet
            notPunctuation = not isPunctuation

            if notPunctuation:
                cleanedTokens.append(singleToken)

            incrementValue = 1
            tokenCounter = tokenCounter + incrementValue

        cleanedLength = len(cleanedTokens)
        minLength = 0
        hasTokens = cleanedLength > minLength

        if hasTokens:
            sentenceWithMarkers = []
            startToken = '<s>'
            sentenceWithMarkers.append(startToken)

            tokenIndex = 0
            maxCleanedTokens = len(cleanedTokens)

            while tokenIndex < maxCleanedTokens:
                currentIndex = tokenIndex
                cleanToken = cleanedTokens[currentIndex]
                sentenceWithMarkers.append(cleanToken)
                tokenIndex = tokenIndex + 1

            endToken = '</s>'
            sentenceWithMarkers.append(endToken)

            processedSentences.append(sentenceWithMarkers)

        sentenceCounter = sentenceCounter + 1

    finalResult = processedSentences
    return finalResult


def buildBigramDictionary(sentencesList):
    bigramFrequencies = {}

    for sentenceTokens in sentencesList:
        tokenCount = len(sentenceTokens)

        for tokenIndex in range(tokenCount - 1):
            firstWord = sentenceTokens[tokenIndex]
            secondWord = sentenceTokens[tokenIndex + 1]

            bigramTuple = (firstWord, secondWord)

            if bigramTuple in bigramFrequencies:
                currentCount = bigramFrequencies[bigramTuple]
                newCount = currentCount + 1
                bigramFrequencies[bigramTuple] = newCount
            else:
                bigramFrequencies[bigramTuple] = 1

    return bigramFrequencies


def computeUnigramCounts(sentencesList):
    unigramCounts = {}

    for sentenceTokens in sentencesList:
        for singleToken in sentenceTokens:
            if singleToken in unigramCounts:
                oldCount = unigramCounts[singleToken]
                updatedCount = oldCount + 1
                unigramCounts[singleToken] = updatedCount
            else:
                unigramCounts[singleToken] = 1

    return unigramCounts


def calculateConditionalProbabilities(bigramDict, unigramDict):
    conditionalProbs = {}

    for bigramKey in bigramDict.keys():
        previousWord = bigramKey[0]
        currentWord = bigramKey[1]

        bigramCount = bigramDict[bigramKey]
        unigramCount = unigramDict[previousWord]

        probability = bigramCount / unigramCount

        if previousWord not in conditionalProbs:
            conditionalProbs[previousWord] = {}

        conditionalProbs[previousWord][currentWord] = probability

    return conditionalProbs


def generateTextSequence(startWord, probsDict, maxLength=5):
    generatedWords = []
    emptyWordList = []
    generatedWords = emptyWordList

    currentWord = startWord
    previousWord = currentWord
    currentToken = previousWord

    targetWordCount = maxLength
    wordCountGoal = targetWordCount
    maximumWords = wordCountGoal

    topCandidateCount = 10
    maxCandidates = topCandidateCount
    candidateLimit = maxCandidates

    stepCounter = 0
    stepLimit = maximumWords
    maxSteps = stepLimit

    while stepCounter < maxSteps:
        stepIndex = stepCounter
        workingWord = currentToken

        wordExists = workingWord in probsDict
        shouldContinue = wordExists

        if not shouldContinue:
            break

        candidatesDict = probsDict[workingWord]

        wordsList = []
        emptyWords = []
        wordsList = emptyWords

        probsList = []
        emptyProbs = []
        probsList = emptyProbs

        for nextWord in candidatesDict.keys():
            candidateWord = nextWord
            wordProbability = candidatesDict[candidateWord]
            probabilityValue = wordProbability

            wordsList.append(candidateWord)
            probsList.append(probabilityValue)

        probsListCopy = probsList
        sortedIndices = sorted(range(len(probsListCopy)),
                               key=lambda i: probsListCopy[i], reverse=True)

        topWords = []
        emptyTopList = []
        topWords = emptyTopList

        numIndices = len(sortedIndices)
        candidateCount = candidateLimit
        maxCandidatesValue = min(candidateCount, numIndices)
        topN = maxCandidatesValue

        rankCounter = 0
        rankLimit = topN

        while rankCounter < rankLimit:
            rankIndex = rankCounter
            arrayIndex = sortedIndices[rankIndex]
            indexValue = arrayIndex
            selectedWord = wordsList[indexValue]

            topWords.append(selectedWord)

            rankCounter = rankCounter + 1

        chosenWord = random.choice(topWords)
        selectedToken = chosenWord

        generatedWords.append(selectedToken)

        endMarker = '</s>'
        isEndToken = selectedToken == endMarker

        if isEndToken:
            break

        currentToken = selectedToken

        stepCounter = stepCounter + 1

    outputSequence = generatedWords
    finalOutput = outputSequence
    return finalOutput


def calculatePerplexity(wordSequence, probsDict):
    epsilonConstant = 1e-10
    smallValue = epsilonConstant
    epsilon = smallValue

    logProbabilities = []
    emptyLogList = []
    logProbabilities = emptyLogList

    sequenceLength = len(wordSequence)
    numWords = sequenceLength
    totalWords = numWords

    positionCounter = 0
    maxPosition = totalWords - 1
    lastValidIndex = maxPosition

    while positionCounter < lastValidIndex:
        positionIndex = positionCounter
        currentPosition = positionIndex
        nextPosition = currentPosition + 1

        prevWord = wordSequence[currentPosition]
        nextWord = wordSequence[nextPosition]

        firstWord = prevWord
        secondWord = nextWord

        wordInDict = firstWord in probsDict

        if wordInDict:
            innerDict = probsDict[firstWord]
            nextWordInDict = secondWord in innerDict

            if nextWordInDict:
                probability = innerDict[secondWord]
                probValue = probability
            else:
                probValue = epsilon
        else:
            probValue = epsilon

        finalProb = probValue
        logProb = np.log(finalProb)
        logValue = logProb

        logProbabilities.append(logValue)

        positionCounter = positionCounter + 1

    logProbList = logProbabilities
    sumLogProbs = 0
    initialSum = sumLogProbs
    runningSum = initialSum

    for singleLogProb in logProbList:
        currentLogProb = singleLogProb
        runningSum = runningSum + currentLogProb

    totalSum = runningSum
    sumLogProbs = totalSum

    numBigrams = len(logProbabilities)
    bigramCount = numBigrams
    totalBigrams = bigramCount

    numeratorValue = sumLogProbs
    denominatorValue = totalBigrams
    averageLogProb = numeratorValue / denominatorValue

    negativeOne = -1
    negativeMultiplier = negativeOne
    negativeAverage = negativeMultiplier * averageLogProb

    exponentialInput = negativeAverage
    perplexityValue = np.exp(exponentialInput)

    finalPerplexity = perplexityValue
    outputValue = finalPerplexity

    return outputValue

# ============================================================================
# QUESTION 2: POS TAGGER USING HIDDEN MARKOV MODEL (HMM)
# ============================================================================


def loadTreebankData():
    taggedSentences = treebank.tagged_sents()

    allSentences = taggedSentences
    sentenceList = allSentences

    totalSentences = 0
    sentenceCounter = 0

    for singleSentence in sentenceList:
        sentenceCounter = sentenceCounter + 1

    totalSentences = sentenceCounter
    numSentences = totalSentences

    splitRatio = 0.8
    eightyPercent = splitRatio
    trainingRatio = eightyPercent

    floatTotal = float(numSentences)
    floatRatio = float(trainingRatio)
    splitProduct = floatTotal * floatRatio
    intSplit = int(splitProduct)
    splitIndex = intSplit

    trainingSentences = []
    emptyTrainList = []
    trainingSentences = emptyTrainList

    trainCounter = 0
    trainLimit = splitIndex

    while trainCounter < trainLimit:
        sentIndex = trainCounter
        currentSentence = sentenceList[sentIndex]
        trainingSentences.append(currentSentence)
        trainCounter = trainCounter + 1

    testingSentences = []
    emptyTestList = []
    testingSentences = emptyTestList

    testCounter = splitIndex
    testLimit = numSentences

    while testCounter < testLimit:
        sentIndex = testCounter
        currentSentence = sentenceList[sentIndex]
        testingSentences.append(currentSentence)
        testCounter = testCounter + 1

    trainingData = trainingSentences
    testingData = testingSentences

    return trainingData, testingData


def extractVocabularyAndTags(sentencesList):
    vocabularySet = set()
    emptyVocabSet = set()
    vocabularySet = emptyVocabSet

    tagSet = set()
    emptyTagSet = set()
    tagSet = emptyTagSet

    sentenceCounter = 0
    numSentences = 0

    for sentence in sentencesList:
        numSentences = numSentences + 1

    totalSentences = numSentences

    sentenceIndex = 0

    while sentenceIndex < totalSentences:
        currentIndex = sentenceIndex
        singleSentence = sentencesList[currentIndex]

        pairCounter = 0
        numPairs = 0

        for pair in singleSentence:
            numPairs = numPairs + 1

        totalPairs = numPairs

        pairIndex = 0

        while pairIndex < totalPairs:
            currentPairIndex = pairIndex
            wordTagPair = singleSentence[currentPairIndex]

            firstElement = wordTagPair[0]
            secondElement = wordTagPair[1]

            wordToken = firstElement
            tagToken = secondElement

            vocabularySet.add(wordToken)
            tagSet.add(tagToken)

            pairIndex = pairIndex + 1

        sentenceIndex = sentenceIndex + 1

    finalVocab = vocabularySet
    finalTags = tagSet

    return finalVocab, finalTags


def computeTagTransitionProbs(trainSentences):
    tagBigramCounts = {}
    tagUnigramCounts = {}

    for sentenceData in trainSentences:
        sentenceLength = len(sentenceData)

        for pairIndex in range(sentenceLength - 1):
            currentPair = sentenceData[pairIndex]
            nextPair = sentenceData[pairIndex + 1]

            currentTag = currentPair[1]
            nextTag = nextPair[1]

            if currentTag in tagUnigramCounts:
                tagUnigramCounts[currentTag] = tagUnigramCounts[currentTag] + 1
            else:
                tagUnigramCounts[currentTag] = 1

            bigramKey = (currentTag, nextTag)

            if bigramKey in tagBigramCounts:
                tagBigramCounts[bigramKey] = tagBigramCounts[bigramKey] + 1
            else:
                tagBigramCounts[bigramKey] = 1

    transitionProbabilities = {}

    for bigramPair in tagBigramCounts.keys():
        previousTag = bigramPair[0]
        followingTag = bigramPair[1]

        bigramFrequency = tagBigramCounts[bigramPair]
        unigramFrequency = tagUnigramCounts[previousTag]

        transitionProb = bigramFrequency / unigramFrequency

        if previousTag not in transitionProbabilities:
            transitionProbabilities[previousTag] = {}

        transitionProbabilities[previousTag][followingTag] = transitionProb

    return transitionProbabilities


def computeEmissionProbs(trainSentences):
    wordTagCounts = {}
    tagCounts = {}

    for sentenceData in trainSentences:
        for wordTagPair in sentenceData:
            wordValue = wordTagPair[0]
            tagValue = wordTagPair[1]

            if tagValue in tagCounts:
                tagCounts[tagValue] = tagCounts[tagValue] + 1
            else:
                tagCounts[tagValue] = 1

            pairKey = (wordValue, tagValue)

            if pairKey in wordTagCounts:
                wordTagCounts[pairKey] = wordTagCounts[pairKey] + 1
            else:
                wordTagCounts[pairKey] = 1

    emissionProbabilities = {}

    for pairKey in wordTagCounts.keys():
        wordToken = pairKey[0]
        tagToken = pairKey[1]

        pairCount = wordTagCounts[pairKey]
        tagCount = tagCounts[tagToken]

        emissionProb = pairCount / tagCount

        if wordToken not in emissionProbabilities:
            emissionProbabilities[wordToken] = {}

        emissionProbabilities[wordToken][tagToken] = emissionProb

    return emissionProbabilities


def viterbiDecode(wordSequence, tagsList, transitionProbs, emissionProbs):
    epsilon = 1e-10

    sequenceLength = len(wordSequence)
    numTags = len(tagsList)

    viterbiMatrix = np.zeros((numTags, sequenceLength))
    backpointerMatrix = np.zeros((numTags, sequenceLength), dtype=int)

    tagToIndex = {}
    indexToTag = {}

    for tagIdx in range(numTags):
        tagName = tagsList[tagIdx]
        tagToIndex[tagName] = tagIdx
        indexToTag[tagIdx] = tagName

    firstWord = wordSequence[0]

    for tagIdx in range(numTags):
        currentTag = indexToTag[tagIdx]

        if firstWord in emissionProbs:
            if currentTag in emissionProbs[firstWord]:
                emissionValue = emissionProbs[firstWord][currentTag]
            else:
                emissionValue = epsilon
        else:
            emissionValue = epsilon

        initialProb = (1.0 / numTags) * emissionValue

        logProb = np.log(initialProb + epsilon)

        viterbiMatrix[tagIdx][0] = logProb

    for wordIdx in range(1, sequenceLength):
        currentWord = wordSequence[wordIdx]

        for currTagIdx in range(numTags):
            currentTag = indexToTag[currTagIdx]

            if currentWord in emissionProbs:
                if currentTag in emissionProbs[currentWord]:
                    emissionValue = emissionProbs[currentWord][currentTag]
                else:
                    emissionValue = epsilon
            else:
                emissionValue = epsilon

            maxScore = -np.inf
            bestPrevTag = 0

            for prevTagIdx in range(numTags):
                previousTag = indexToTag[prevTagIdx]

                prevScore = viterbiMatrix[prevTagIdx][wordIdx - 1]

                if previousTag in transitionProbs:
                    if currentTag in transitionProbs[previousTag]:
                        transitionValue = transitionProbs[previousTag][currentTag]
                    else:
                        transitionValue = epsilon
                else:
                    transitionValue = epsilon

                logTransition = np.log(transitionValue + epsilon)
                logEmission = np.log(emissionValue + epsilon)

                totalScore = prevScore + logTransition + logEmission

                if totalScore > maxScore:
                    maxScore = totalScore
                    bestPrevTag = prevTagIdx

            viterbiMatrix[currTagIdx][wordIdx] = maxScore
            backpointerMatrix[currTagIdx][wordIdx] = bestPrevTag

    lastColumnScores = viterbiMatrix[:, sequenceLength - 1]

    bestFinalTag = 0
    bestFinalScore = lastColumnScores[0]

    for tagIdx in range(1, numTags):
        scoreValue = lastColumnScores[tagIdx]
        if scoreValue > bestFinalScore:
            bestFinalScore = scoreValue
            bestFinalTag = tagIdx

    decodedPath = []

    for position in range(sequenceLength):
        decodedPath.append(0)

    decodedPath[sequenceLength - 1] = bestFinalTag

    for wordIdx in range(sequenceLength - 1, 0, -1):
        currentTagIdx = decodedPath[wordIdx]
        previousTagIdx = backpointerMatrix[currentTagIdx][wordIdx]
        decodedPath[wordIdx - 1] = previousTagIdx

    tagSequence = []

    for tagIdx in decodedPath:
        tagName = indexToTag[tagIdx]
        tagSequence.append(tagName)

    return tagSequence


def evaluateHMM(testSentences, transitionProbs, emissionProbs, tagsList):
    correctCount = 0
    totalCount = 0

    for sentenceData in testSentences:
        wordSequence = []
        trueTagSequence = []

        for wordTagPair in sentenceData:
            wordValue = wordTagPair[0]
            tagValue = wordTagPair[1]

            wordSequence.append(wordValue)
            trueTagSequence.append(tagValue)

        predictedTags = viterbiDecode(
            wordSequence, tagsList, transitionProbs, emissionProbs)

        for tagIdx in range(len(trueTagSequence)):
            totalCount = totalCount + 1

            trueTag = trueTagSequence[tagIdx]
            predictedTag = predictedTags[tagIdx]

            if trueTag == predictedTag:
                correctCount = correctCount + 1

    accuracyValue = correctCount / totalCount

    return accuracyValue

# ============================================================================
# QUESTION 3: POS TAGGER USING CONDITIONAL RANDOM FIELD (CRF)
# ============================================================================


def extractWordFeatures(word):
    featureDict = {}

    lowercaseWord = word.lower()
    featureDict['word'] = lowercaseWord

    isDigit = word.isdigit()
    featureDict['isNumber'] = isDigit

    containsHyphen = '-' in word
    featureDict['hasHyphen'] = containsHyphen

    isUppercase = word.isupper()
    featureDict['isAllUpper'] = isUppercase

    hasUpper = False
    for character in word:
        if character.isupper():
            hasUpper = True
            break
    featureDict['hasUpperCase'] = hasUpper

    isLowercase = word.islower()
    featureDict['isAllLower'] = isLowercase

    wordLen = len(word)
    featureDict['wordLength'] = wordLen

    bigramsList = []

    if wordLen >= 2:
        for charIdx in range(wordLen - 1):
            firstChar = word[charIdx]
            secondChar = word[charIdx + 1]
            bigram = firstChar + secondChar
            bigramsList.append(bigram)

    featureDict['wordBigrams'] = bigramsList

    return featureDict


def buildFeatureSet(taggedSentences):
    allSentenceFeatures = []
    allSentenceTags = []

    for sentenceData in taggedSentences:
        sentenceFeatures = []
        sentenceTags = []

        for wordTagPair in sentenceData:
            wordValue = wordTagPair[0]
            tagValue = wordTagPair[1]

            wordFeatures = extractWordFeatures(wordValue)

            sentenceFeatures.append(wordFeatures)
            sentenceTags.append(tagValue)

        allSentenceFeatures.append(sentenceFeatures)
        allSentenceTags.append(sentenceTags)

    return allSentenceFeatures, allSentenceTags


def trainAndEvaluateCRF(trainSentences, testSentences):
    xTrainFeatures, yTrainTags = buildFeatureSet(trainSentences)
    xTestFeatures, yTestTags = buildFeatureSet(testSentences)

    crfModel = CRF()

    crfModel.fit(xTrainFeatures, yTrainTags)

    yPredictedTags = crfModel.predict(xTestFeatures)

    correctCount = 0
    totalCount = 0

    for sentIdx in range(len(yTestTags)):
        trueTags = yTestTags[sentIdx]
        predictedTags = yPredictedTags[sentIdx]

        for tagIdx in range(len(trueTags)):
            totalCount = totalCount + 1

            trueTag = trueTags[tagIdx]
            predictedTag = predictedTags[tagIdx]

            if trueTag == predictedTag:
                correctCount = correctCount + 1

    accuracyValue = correctCount / totalCount

    return accuracyValue

# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    print("CS 421 Assignment 3: N-gram Text Generation and POS Tagging\n")

    print("="*70)
    print("QUESTION 1: BIGRAM TEXT GENERATION")
    print("="*70)

    gatsbyFilePath = "GreatGatsby.txt"
    rawText = loadGreatGatsbyText(gatsbyFilePath)

    processedSentences = preprocessText(rawText)

    bigramDict = buildBigramDictionary(processedSentences)

    unigramDict = computeUnigramCounts(processedSentences)

    conditionalProbs = calculateConditionalProbabilities(
        bigramDict, unigramDict)

    startWord = 'He'
    generatedSequence = generateTextSequence(
        startWord, conditionalProbs, maxLength=5)

    print(f"\nGenerated text starting with '{startWord}':")
    print(f"{startWord} {' '.join(generatedSequence)}")

    fullSequence = [startWord] + generatedSequence

    perplexity = calculatePerplexity(fullSequence, conditionalProbs)

    print(f"\nPerplexity of generated sequence: {perplexity:.2f}")

    print("\n" + "="*70)
    print("QUESTION 2: HMM-BASED POS TAGGING")
    print("="*70)

    trainSentences, testSentences = loadTreebankData()

    vocab, tags = extractVocabularyAndTags(trainSentences)

    tagsList = list(tags)

    transitionProbs = computeTagTransitionProbs(trainSentences)

    emissionProbs = computeEmissionProbs(trainSentences)

    hmmAccuracy = evaluateHMM(
        testSentences, transitionProbs, emissionProbs, tagsList)

    print(f"\nHMM POS Tagging Accuracy: {hmmAccuracy:.4f}")

    print("\n" + "="*70)
    print("QUESTION 3: CRF-BASED POS TAGGING")
    print("="*70)

    crfAccuracy = trainAndEvaluateCRF(trainSentences, testSentences)

    print(f"\nCRF POS Tagging Accuracy: {crfAccuracy:.4f}")

    print("\n" + "="*70)
    print("COMPARISON AND ANALYSIS")
    print("="*70)

    print(f"\nHMM Accuracy: {hmmAccuracy:.4f}")
    print(f"CRF Accuracy: {crfAccuracy:.4f}")

    if crfAccuracy > hmmAccuracy:
        difference = crfAccuracy - hmmAccuracy
        print(f"\nCRF outperforms HMM by {difference:.4f}")
        print("CRF captures richer feature representations and context")
    else:
        difference = hmmAccuracy - crfAccuracy
        print(f"\nHMM outperforms CRF by {difference:.4f}")
        print("HMM's probabilistic framework was sufficient for this task")

    print("\nAssignment 3 completed successfully!")
