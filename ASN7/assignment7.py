"""
NLP Assignment 7 - Complete Implementation
==========================================

This project includes three NLP systems:
- Q1: Corpus-Based Chatbot using TF-IDF
- Q2: LSTM Slot Filling for ATIS dataset
- Q3: Neural Machine Translation (German -> English)

All code uses camelCase naming and casual comments every 4-5 lines.
"""

import nltk
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# download the stuff we need
def downloadNltkData():
    """Grab all the NLTK goodies we need for this assignment."""
    packagesToGrab = ['nps_chat', 'punkt', 'stopwords']

    for packageName in packagesToGrab:
        try:
            # try to find it first, don't download if we already have it
            if packageName == 'nps_chat':
                nltk.data.find('corpora/nps_chat')
            elif packageName == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find(f'corpora/{packageName}')
        except LookupError:
            print(f"Downloading {packageName}...")
            nltk.download(packageName, quiet=True)


# ============================================================================
# QUESTION 1: Corpus-Based Chatbot with TF-IDF
# ============================================================================

class CorpusChatbot:
    """
    A retrieval-based chatbot that finds similar sentences using TF-IDF.

    Uses the NPS Chat corpus and responds with the most similar sentence
    based on cosine similarity of TF-IDF vectors.
    """

    def __init__(self):
        """Set up the chatbot by loading and filtering the corpus."""
        # grab the raw chat sentences from nltk
        print("Loading NPS Chat corpus...")
        rawPosts = nltk.corpus.nps_chat.posts()

        # pull out just the text from each post
        allSentences = []
        for chatPost in rawPosts:
            # each post is a list of tuples (word, tag)
            # handle both tuples and strings
            words = []
            for item in chatPost:
                if isinstance(item, tuple):
                    words.append(item[0])
                elif isinstance(item, str):
                    words.append(item)
            sentenceText = ' '.join(words)
            allSentences.append(sentenceText)

        # now filter out questions and short stuff
        print("Filtering sentences (removing questions and short ones)...")
        self.chatResponses = self._filterSentences(allSentences)
        print(f"Got {len(self.chatResponses)} usable responses!")

        # build our TF-IDF vectors for all the responses
        print("Building TF-IDF vectors...")
        self.documentVectors, self.vocabList, self.idfScores = self._buildTfidfVectors(self.chatResponses)
        print("Chatbot is ready to roll!")

    def _filterSentences(self, sentenceList):
        """
        Filter out questions and super short sentences.

        We keep greetings even if they're short, but otherwise need 4+ words.
        """
        # these are question words we want to avoid
        questionStarters = ['what', 'why', 'when', 'where', 'is', 'how', 'do',
                          'does', 'which', 'are', 'could', 'would', 'should',
                          'has', 'have', 'whom', 'whose', "don't"]

        # greetings we want to keep even if short
        greetingWords = ['hello', 'hi', 'greetings', "what's up", 'hey']

        filteredList = []

        for sentence in sentenceList:
            # make it lowercase for checking
            lowerSentence = sentence.lower().strip()

            # skip empty ones
            if not lowerSentence:
                continue

            # check if it starts with a question word
            isQuestion = any(lowerSentence.startswith(qWord) for qWord in questionStarters)
            if isQuestion:
                continue

            # count the words
            wordCount = len(lowerSentence.split())

            # if it's short, only keep if it has a greeting
            if wordCount <= 4:
                hasGreeting = any(greetWord in lowerSentence for greetWord in greetingWords)
                if not hasGreeting:
                    continue

            filteredList.append(sentence)

        return filteredList

    def _buildTfidfVectors(self, documentList):
        """
        Build TF-IDF vectors for all documents from scratch.

        Returns:
            - documentVectors: list of TF-IDF vectors
            - vocabList: list of unique words
            - idfScores: dict mapping words to their IDF scores
        """
        # tokenize all documents and build vocab
        tokenizedDocs = []
        allWords = []

        for doc in documentList:
            # simple tokenization - just split and lowercase
            tokens = doc.lower().split()
            tokenizedDocs.append(tokens)
            allWords.extend(tokens)

        # get unique words for our vocab
        vocabList = sorted(list(set(allWords)))
        wordToIndex = {word: idx for idx, word in enumerate(vocabList)}

        # calculate IDF scores
        # IDF(word) = log(total docs / docs containing word)
        numDocs = len(documentList)
        docFrequency = Counter()

        for tokens in tokenizedDocs:
            uniqueWordsInDoc = set(tokens)
            for word in uniqueWordsInDoc:
                docFrequency[word] += 1

        idfScores = {}
        for word in vocabList:
            # add 1 to avoid division by zero
            idfScores[word] = math.log(numDocs / (docFrequency[word] + 1))

        # now build TF-IDF vectors for each document
        documentVectors = []

        for tokens in tokenizedDocs:
            # count term frequencies in this doc
            termFreq = Counter(tokens)

            # build the TF-IDF vector
            tfidfVector = np.zeros(len(vocabList))

            for word, count in termFreq.items():
                if word in wordToIndex:
                    idx = wordToIndex[word]
                    # TF = count, IDF from our scores
                    tfidfVector[idx] = count * idfScores[word]

            documentVectors.append(tfidfVector)

        return documentVectors, vocabList, idfScores

    def _cosineSimilarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.

        Returns a value between 0 and 1, where 1 is identical.
        """
        # dot product of the two vectors
        dotProduct = np.dot(vec1, vec2)

        # magnitudes
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

        # avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dotProduct / (mag1 * mag2)

    def _convertToTfidf(self, userInput):
        """Convert a user's input text into a TF-IDF vector."""
        # tokenize the input
        tokens = userInput.lower().split()
        termFreq = Counter(tokens)

        # build TF-IDF vector using our existing vocab and IDF scores
        wordToIndex = {word: idx for idx, word in enumerate(self.vocabList)}
        tfidfVector = np.zeros(len(self.vocabList))

        for word, count in termFreq.items():
            if word in wordToIndex:
                idx = wordToIndex[word]
                # use IDF score if we have it, otherwise use a default
                idfScore = self.idfScores.get(word, 1.0)
                tfidfVector[idx] = count * idfScore

        return tfidfVector

    def getBestResponse(self, userInput):
        """
        Find the most similar response to the user's input.

        Returns the sentence with highest cosine similarity.
        """
        # convert user input to TF-IDF
        userVector = self._convertToTfidf(userInput)

        # find the most similar document
        bestSimilarity = -1
        bestResponseIdx = 0

        for idx, docVector in enumerate(self.documentVectors):
            similarity = self._cosineSimilarity(userVector, docVector)

            if similarity > bestSimilarity:
                bestSimilarity = similarity
                bestResponseIdx = idx

        return self.chatResponses[bestResponseIdx]

    def chat(self):
        """
        Run an interactive chat session.

        User can type messages and get responses. Type 'quit' to exit.
        """
        print("\n" + "="*60)
        print("Corpus-Based Chatbot")
        print("="*60)
        print("Type your message and I'll find a similar response!")
        print("(Type 'quit' to exit)\n")

        while True:
            userInput = input("You: ").strip()

            if userInput.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Later! ✌️")
                break

            if not userInput:
                continue

            # get and display the response
            response = self.getBestResponse(userInput)
            print(f"Bot: {response}\n")


def runChatbot():
    """Fire up the chatbot and let it run."""
    # make sure we have the data
    downloadNltkData()

    # create and run the chatbot
    chatbot = CorpusChatbot()
    chatbot.chat()


# ============================================================================
# Main Runner
# ============================================================================

def main():
    """Main entry point - runs all three questions."""
    print("="*60)
    print("NLP Assignment 7")
    print("="*60)

    print("\n>>> Running Question 1: Corpus-Based Chatbot")
    runChatbot()

    print("\n>>> Questions 2 and 3 are in separate files:")
    print("    - q2_slot_filling.py")
    print("    - q3_translation.py")


if __name__ == "__main__":
    main()
