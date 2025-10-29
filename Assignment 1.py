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

foxHealth = pd.read_csv('Health-Tweets/foxnewshealth.txt', sep='|', names=[
                        'id', 'timestamp', 'tweets'], encoding='latin-1', on_bad_lines='skip')
cnnHealth = pd.read_csv('Health-Tweets/cnnhealth.txt', sep='|', names=[
                        'id', 'timestamp', 'tweets'], encoding='latin-1', on_bad_lines='skip')

foxHealth['source'] = 'Fox News'
cnnHealth['source'] = 'CNN'

healthNewsDF = pd.concat([foxHealth, cnnHealth], ignore_index=True)

print(healthNewsDF.head())
print(f"\nDataset shape: {healthNewsDF.shape}")
print(f"\nColumns: {healthNewsDF.columns.tolist()}")


def extractHashtags(text):
    if pd.isna(text):
        return []
    hashtagPattern = r'#\w+'
    return re.findall(hashtagPattern, str(text))


healthNewsDF['hashtags'] = healthNewsDF['tweets'].apply(extractHashtags)

allHashtags = [hashtag.lower() for hashtagsList in healthNewsDF['hashtags']
               for hashtag in hashtagsList]

hashtagCounter = Counter(allHashtags)

print("\n" + "="*50)
print("HASHTAG ANALYSIS")
print("="*50)
print(f"\nTotal unique hashtags: {len(hashtagCounter)}")
print(f"Total hashtag occurrences: {sum(hashtagCounter.values())}")
print(f"\nTop 10 most common hashtags:")
for hashtag, count in hashtagCounter.most_common(10):
    print(f"  {hashtag}: {count}")

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

print("\n" + "="*50)
print("CLEANED TWEETS SAMPLE")
print("="*50)
print("\nOriginal tweet:")
print(healthNewsDF['tweets'].iloc[0])
print("\nCleaned tweet:")
print(healthNewsDF['cleanedTweets'].iloc[0])

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

healthNewsDF['sentences'] = healthNewsDF['cleanedTweets'].apply(sentenceTokenize)
healthNewsDF['words'] = healthNewsDF['cleanedTweets'].apply(wordTokenize)

allWords = [word for wordsList in healthNewsDF['words'] 
            for word in wordsList if word.strip()]

wordCounter = Counter(allWords)

print("\n" + "="*50)
print("WORD ANALYSIS (CLEANED)")
print("="*50)
print(f"\nTotal unique words: {len(wordCounter)}")
print(f"Total word occurrences: {sum(wordCounter.values())}")
print(f"\nTop 30 most common words:")
for word, count in wordCounter.most_common(30):
    print(f"  '{word}': {count}")

stopWords = set(stopwords.words('english'))

wordsWithoutStopwords = [word for word in allWords if word not in stopWords]

wordCounterNoStopwords = Counter(wordsWithoutStopwords)

print("\n" + "="*50)
print("WORD ANALYSIS (WITHOUT STOPWORDS)")
print("="*50)
print(f"\nTotal unique words: {len(wordCounterNoStopwords)}")
print(f"Total word occurrences: {sum(wordCounterNoStopwords.values())}")
print(f"\nTop 30 most common words:")
for word, count in wordCounterNoStopwords.most_common(30):
    print(f"  '{word}': {count}")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

lemmatizedWords = [lemmatizer.lemmatize(word) for word in wordsWithoutStopwords]
stemmedWords = [stemmer.stem(word) for word in wordsWithoutStopwords]

lemmatizedCounter = Counter(lemmatizedWords)
stemmedCounter = Counter(stemmedWords)

print("\n" + "="*50)
print("WORD ANALYSIS (LEMMATIZED)")
print("="*50)
print(f"\nTotal unique words: {len(lemmatizedCounter)}")
print(f"Total word occurrences: {sum(lemmatizedCounter.values())}")
print(f"\nTop 30 most common words:")
for word, count in lemmatizedCounter.most_common(30):
    print(f"  '{word}': {count}")

print("\n" + "="*50)
print("WORD ANALYSIS (STEMMED)")
print("="*50)
print(f"\nTotal unique words: {len(stemmedCounter)}")
print(f"Total word occurrences: {sum(stemmedCounter.values())}")
print(f"\nTop 30 most common words:")
for word, count in stemmedCounter.most_common(30):
    print(f"  '{word}': {count}")

print("\n" + "="*50)
print("COMPARISON: LEMMATIZATION vs STEMMING")
print("="*50)
print("\nSample words comparison:")
sampleWords = ['studies', 'studying', 'cancers', 'children', 'running', 'better', 'worse', 'having']
print(f"\n{'Original':<15} {'Lemmatized':<15} {'Stemmed':<15}")
print("-" * 45)
for word in sampleWords:
    lemma = lemmatizer.lemmatize(word)
    stem = stemmer.stem(word)
    print(f"{word:<15} {lemma:<15} {stem:<15}")

corpus = {
    'original': sorted(set(allWords)),
    'withoutStopwords': sorted(set(wordsWithoutStopwords)),
    'lemmatized': sorted(set(lemmatizedWords)),
    'stemmed': sorted(set(stemmedWords))
}

print("\n" + "="*50)
print("CORPUS MAINTENANCE")
print("="*50)
print(f"\nCorpus of Original Words: {len(corpus['original'])} unique words")
print(f"Corpus without Stopwords: {len(corpus['withoutStopwords'])} unique words")
print(f"Corpus of Lemmatized Words: {len(corpus['lemmatized'])} unique words")
print(f"Corpus of Stemmed Words: {len(corpus['stemmed'])} unique words")

print("\n" + "="*50)
print("SAMPLE CORPUS ENTRIES")
print("="*50)
print("\nFirst 20 words from each corpus:")
print(f"\nOriginal: {corpus['original'][:20]}")
print(f"\nWithout Stopwords: {corpus['withoutStopwords'][:20]}")
print(f"\nLemmatized: {corpus['lemmatized'][:20]}")
print(f"\nStemmed: {corpus['stemmed'][:20]}")

corpusDF = pd.DataFrame({
    'Original': pd.Series(corpus['original']),
    'WithoutStopwords': pd.Series(corpus['withoutStopwords']),
    'Lemmatized': pd.Series(corpus['lemmatized']),
    'Stemmed': pd.Series(corpus['stemmed'])
})

corpusDF.to_csv('corpus.csv', index=False)
print("\n✓ Corpus saved to 'corpus.csv'")

vocabSummary = pd.DataFrame({
    'Corpus Type': ['Original', 'Without Stopwords', 'Lemmatized', 'Stemmed'],
    'Unique Words': [len(corpus['original']), len(corpus['withoutStopwords']), 
                     len(corpus['lemmatized']), len(corpus['stemmed'])],
    'Total Occurrences': [len(allWords), len(wordsWithoutStopwords), 
                         len(lemmatizedWords), len(stemmedWords)]
})

print("\n" + "="*50)
print("VOCABULARY SUMMARY")
print("="*50)
print(vocabSummary.to_string(index=False))

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
            distance = minEditDist(target, word, insCost=1, delCost=1, subCost=2)
            distances.append((word, distance))
    
    distances.sort(key=lambda x: (x[1], x[0]))
    
    return distances[:topN]

print("\n" + "="*50)
print("MINIMUM EDIT DISTANCE & SPELL CHECKER")
print("="*50)

testWords = [
    ('helth', 'health'),
    ('cancr', 'cancer'),
    ('studi', 'study'),
    ('patien', 'patient')
]

print("\nMinimum Edit Distance Examples:")
print(f"\n{'Target':<15} {'Source':<15} {'Distance':<10}")
print("-" * 40)
for target, source in testWords:
    dist = minEditDist(target, source, insCost=1, delCost=1, subCost=2)
    print(f"{target:<15} {source:<15} {dist:<10}")

print("\n" + "="*50)
print("SPELL CHECKER DEMO")
print("="*50)

misspelledWords = ['helth', 'cancr', 'studi', 'patien', 'brayn', 'docter']

for misspelled in misspelledWords:
    suggestions = spellChecker(misspelled, corpus['withoutStopwords'], topN=5)
    print(f"\nMisspelled: '{misspelled}'")
    print(f"Top 5 suggestions (word, distance):")
    for word, dist in suggestions:
        print(f"  {word:<20} distance: {dist}")

print("\n" + "="*50)
print("CUSTOM SPELL CHECKER TEST")
print("="*50)
userInput = 'diseaz'
print(f"\nChecking spelling for: '{userInput}'")
suggestions = spellChecker(userInput, corpus['withoutStopwords'], topN=5)
print(f"\nTop 5 suggestions:")
for i, (word, dist) in enumerate(suggestions, 1):
    print(f"  {i}. {word:<20} (edit distance: {dist})")
