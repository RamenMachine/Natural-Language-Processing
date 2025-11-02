import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data
foxHealth = pd.read_csv('Health-Tweets/foxnewshealth.txt', sep='|', names=[
                        'id', 'timestamp', 'tweets'], encoding='latin-1', on_bad_lines='skip')
cnnHealth = pd.read_csv('Health-Tweets/cnnhealth.txt', sep='|', names=[
                        'id', 'timestamp', 'tweets'], encoding='latin-1', on_bad_lines='skip')

foxHealth['source'] = 'Fox News'
cnnHealth['source'] = 'CNN'

healthNewsDF = pd.concat([foxHealth, cnnHealth], ignore_index=True)


def extractHashtags(text):
    if pd.isna(text):
        return []
    hashtagPattern = r'#\w+'
    return re.findall(hashtagPattern, str(text))


healthNewsDF['hashtags'] = healthNewsDF['tweets'].apply(extractHashtags)

allHashtags = [hashtag.lower() for hashtagsList in healthNewsDF['hashtags']
               for hashtag in hashtagsList]

hashtagCounter = Counter(allHashtags)


def cleanTweet(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text


healthNewsDF['cleanedTweets'] = healthNewsDF['tweets'].apply(cleanTweet)


def sentenceTokenize(text):
    if pd.isna(text):
        return []
    return nltk.sent_tokenize(str(text))


def wordTokenize(text):
    if pd.isna(text):
        return []
    sentences = sentenceTokenize(text)
    words = []
    for sentence in sentences:
        words.extend(nltk.word_tokenize(sentence))
    return words


healthNewsDF['sentences'] = healthNewsDF['cleanedTweets'].apply(
    sentenceTokenize)
healthNewsDF['words'] = healthNewsDF['cleanedTweets'].apply(wordTokenize)

allWords = [word for wordsList in healthNewsDF['words']
            for word in wordsList if word.strip()]

wordCounter = Counter(allWords)

# Stopword removal
stopWords = set(stopwords.words('english'))
wordsWithoutStopwords = [word for word in allWords if word not in stopWords]
wordCounterNoStopwords = Counter(wordsWithoutStopwords)

# Lemmatization and stemming
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

lemmatizedWords = [lemmatizer.lemmatize(word)
                   for word in wordsWithoutStopwords]
stemmedWords = [stemmer.stem(word) for word in wordsWithoutStopwords]

lemmatizedCounter = Counter(lemmatizedWords)
stemmedCounter = Counter(stemmedWords)

# Build corpus dictionary
corpus = {
    'original': sorted(set(allWords)),
    'withoutStopwords': sorted(set(wordsWithoutStopwords)),
    'lemmatized': sorted(set(lemmatizedWords)),
    'stemmed': sorted(set(stemmedWords))
}


def minEditDist(target, source, insCost=1, delCost=1, subCost=2):
    n = len(target)
    m = len(source)

    distMatrix = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        distMatrix[i][0] = distMatrix[i-1][0] + delCost

    for j in range(1, m + 1):
        distMatrix[0][j] = distMatrix[0][j-1] + insCost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i-1] == source[j-1]:
                distMatrix[i][j] = distMatrix[i-1][j-1]
            else:
                distMatrix[i][j] = min(
                    distMatrix[i-1][j] + delCost,
                    distMatrix[i][j-1] + insCost,
                    distMatrix[i-1][j-1] + subCost
                )

    return distMatrix[n][m]


def spellChecker(target, corpusWords=None, topN=5):
    if corpusWords is None:
        corpusWords = corpus['withoutStopwords']

    target = target.lower()

    distances = []
    for word in corpusWords:
        if word:
            distance = minEditDist(
                target, word, insCost=1, delCost=1, subCost=0)
            distances.append((word, distance))

    distances.sort(key=lambda x: (x[1], x[0]))

    return distances[:topN]
