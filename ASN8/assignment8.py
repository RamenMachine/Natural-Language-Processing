"""
NLP Assignment 8 - Text Summarization

This project includes three text summarization approaches:
- Q1: Abstractive Summarization using Encoder-Decoder
- Q2: Abstractive Summarization using Pre-trained T5
- Q4: Extractive Summarization using PageRank
"""

import nltk
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text.rouge import ROUGEScore
from transformers import T5Tokenizer, T5ForConditionalGeneration
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def downloadNltkData():
    """Grab NLTK packages we need."""
    packagesToGrab = ['punkt', 'stopwords']

    for packageName in packagesToGrab:
        try:
            if packageName == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find(f'corpora/{packageName}')
        except LookupError:
            print(f"Downloading {packageName}...")
            nltk.download(packageName, quiet=True)


# ============================================================================
# Q1: Abstractive Summarization using Encoder-Decoder
# ============================================================================

class AbstractiveSummarizer:
    """Encoder-decoder model for abstractive text summarization."""

    def __init__(self):
        self.articleWordToIdx = {}
        self.articleIdxToWord = {}
        self.highlightWordToIdx = {}
        self.highlightIdxToWord = {}
        self.maxArticleLen = 0
        self.maxHighlightLen = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loadCnnDailymail(self, numSamples=1000):
        """Load CNN/DailyMail dataset from Huggingface."""
        print("Loading CNN/DailyMail dataset...")

        trainData = load_dataset('cnn_dailymail', '3.0.0', split='train', streaming=True)
        testData = load_dataset('cnn_dailymail', '3.0.0', split='test', streaming=True)

        trainArticles = []
        trainHighlights = []

        for idx, sample in enumerate(trainData):
            if idx >= numSamples:
                break
            trainArticles.append(sample['article'])
            trainHighlights.append(sample['highlights'])

            if (idx + 1) % 200 == 0:
                print(f"Loaded {idx + 1} training samples...")

        testArticles = []
        testHighlights = []

        for idx, sample in enumerate(testData):
            if idx >= numSamples // 5:
                break
            testArticles.append(sample['article'])
            testHighlights.append(sample['highlights'])

        print(f"Train: {len(trainArticles)} samples, Test: {len(testArticles)} samples")

        return trainArticles, trainHighlights, testArticles, testHighlights

    def preprocessText(self, text):
        """Clean text - lowercase and remove special chars."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def buildVocabs(self, articles, highlights, maxVocabSize=10000):
        """Build word vocabs for articles and highlights."""
        print("Building vocabularies...")

        articleWords = []
        highlightWords = []

        for article, highlight in zip(articles, highlights):
            articleTokens = self.preprocessText(article).split()
            highlightTokens = self.preprocessText(highlight).split()

            articleWords.extend(articleTokens)
            highlightWords.extend(highlightTokens)

        # grab most common words
        articleCounter = Counter(articleWords)
        highlightCounter = Counter(highlightWords)

        topArticle = [word for word, count in articleCounter.most_common(maxVocabSize - 3)]
        topHighlight = [word for word, count in highlightCounter.most_common(maxVocabSize - 3)]

        # article vocab: 0=PAD, 1=UNK
        self.articleWordToIdx = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(topArticle):
            self.articleWordToIdx[word] = idx + 2
        self.articleIdxToWord = {idx: word for word, idx in self.articleWordToIdx.items()}

        # highlight vocab: add BOS and EOS tags
        self.highlightWordToIdx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for idx, word in enumerate(topHighlight):
            self.highlightWordToIdx[word] = idx + 4
        self.highlightIdxToWord = {idx: word for word, idx in self.highlightWordToIdx.items()}

        print(f"Article vocab: {len(self.articleWordToIdx)} words")
        print(f"Highlight vocab: {len(self.highlightWordToIdx)} words")

    def preprocessData(self, articles, highlights, isTrain=False):
        """Convert text to sequences and pad them."""
        articleSeqs = []
        highlightSeqs = []

        for article, highlight in zip(articles, highlights):
            articleTokens = self.preprocessText(article).split()
            highlightTokens = self.preprocessText(highlight).split()

            # convert to indices
            articleIdxs = []
            for word in articleTokens:
                articleIdxs.append(self.articleWordToIdx.get(word, self.articleWordToIdx['<UNK>']))

            # add BOS and EOS to highlights
            highlightIdxs = [self.highlightWordToIdx['<BOS>']]
            for word in highlightTokens:
                highlightIdxs.append(self.highlightWordToIdx.get(word, self.highlightWordToIdx['<UNK>']))
            highlightIdxs.append(self.highlightWordToIdx['<EOS>'])

            articleSeqs.append(articleIdxs)
            highlightSeqs.append(highlightIdxs)

        # figure out max lengths if training
        if isTrain:
            self.maxArticleLen = min(max(len(seq) for seq in articleSeqs), 400)
            self.maxHighlightLen = min(max(len(seq) for seq in highlightSeqs), 100)
            print(f"Max article length: {self.maxArticleLen}, Max highlight length: {self.maxHighlightLen}")

        # pad or truncate sequences
        paddedArticles = []
        paddedHighlights = []

        for articleSeq, highlightSeq in zip(articleSeqs, highlightSeqs):
            if len(articleSeq) > self.maxArticleLen:
                articleSeq = articleSeq[:self.maxArticleLen]
            else:
                articleSeq = articleSeq + [0] * (self.maxArticleLen - len(articleSeq))

            if len(highlightSeq) > self.maxHighlightLen:
                highlightSeq = highlightSeq[:self.maxHighlightLen]
            else:
                highlightSeq = highlightSeq + [0] * (self.maxHighlightLen - len(highlightSeq))

            paddedArticles.append(articleSeq)
            paddedHighlights.append(highlightSeq)

        return np.array(paddedArticles), np.array(paddedHighlights)


class EncoderDecoder(nn.Module):
    """LSTM encoder-decoder for text generation."""

    def __init__(self, articleVocabSize, highlightVocabSize, embeddingDim=128, hiddenDim=256):
        super(EncoderDecoder, self).__init__()

        self.hiddenDim = hiddenDim

        # encoder layers
        self.encoderEmbedding = nn.Embedding(articleVocabSize, embeddingDim, padding_idx=0)
        self.encoderLstm = nn.LSTM(embeddingDim, hiddenDim, batch_first=True)

        # decoder layers
        self.decoderEmbedding = nn.Embedding(highlightVocabSize, embeddingDim, padding_idx=0)
        self.decoderLstm = nn.LSTM(embeddingDim, hiddenDim, batch_first=True)
        self.fc = nn.Linear(hiddenDim, highlightVocabSize)

    def forward(self, article, highlight):
        # encode the article
        articleEmbed = self.encoderEmbedding(article)
        encoderOutput, (hiddenState, cellState) = self.encoderLstm(articleEmbed)

        # decode using encoder states
        highlightEmbed = self.decoderEmbedding(highlight)
        decoderOutput, _ = self.decoderLstm(highlightEmbed, (hiddenState, cellState))

        output = self.fc(decoderOutput)

        return output


class SummarizationDataset(Dataset):
    """Dataset wrapper for summarization."""

    def __init__(self, articles, highlights):
        self.articles = torch.LongTensor(articles)
        self.highlights = torch.LongTensor(highlights)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx], self.highlights[idx]


def trainEncoderDecoder(model, trainLoader, epochs=5, learningRate=0.001, device='cpu'):
    """Train the encoder-decoder model."""
    print("Training encoder-decoder...")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        model.train()
        totalLoss = 0

        for batchIdx, (articles, highlights) in enumerate(trainLoader):
            articles = articles.to(device)
            highlights = highlights.to(device)

            # decoder input is highlights shifted by one
            decoderInput = highlights[:, :-1]
            target = highlights[:, 1:]

            optimizer.zero_grad()

            output = model(articles, decoderInput)

            loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))

            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            if (batchIdx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batchIdx+1}/{len(trainLoader)}, Loss: {loss.item():.4f}")

        avgLoss = totalLoss / len(trainLoader)
        print(f"Epoch {epoch+1} done - avg loss: {avgLoss:.4f}")

    return model


def beamSearch(model, article, beamWidth=3, maxLen=50, bosIdx=2, eosIdx=3, device='cpu'):
    """Generate summary using beam search."""
    model.eval()

    with torch.no_grad():
        article = article.unsqueeze(0).to(device)
        articleEmbed = model.encoderEmbedding(article)
        encoderOutput, (hiddenState, cellState) = model.encoderLstm(articleEmbed)

        # start with BOS token
        sequences = [[([bosIdx], 0.0, hiddenState, cellState)]]

        for _ in range(maxLen):
            allCandidates = []

            for seq, score, h, c in sequences[-1]:
                if seq[-1] == eosIdx:
                    allCandidates.append((seq, score, h, c))
                    continue

                decoderInput = torch.LongTensor([[seq[-1]]]).to(device)
                decoderEmbed = model.decoderEmbedding(decoderInput)
                decoderOutput, (newH, newC) = model.decoderLstm(decoderEmbed, (h, c))
                output = model.fc(decoderOutput)

                # get top k predictions
                logProbs = torch.log_softmax(output[0, 0], dim=0)
                topProbs, topIdxs = torch.topk(logProbs, beamWidth)

                for prob, idx in zip(topProbs, topIdxs):
                    candidate = (seq + [idx.item()], score + prob.item(), newH, newC)
                    allCandidates.append(candidate)

            # keep best beams
            ordered = sorted(allCandidates, key=lambda x: x[1], reverse=True)
            sequences.append(ordered[:beamWidth])

            # stop if all beams ended
            if all(seq[0][-1] == eosIdx for seq in sequences[-1]):
                break

        bestSeq = sequences[-1][0][0]
        return bestSeq


# ============================================================================
# Q2: Abstractive Summarization using T5
# ============================================================================

class T5Summarizer:
    """Pre-trained T5 for text summarization."""

    def __init__(self):
        print("Loading T5 model...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print("T5 ready to go!")

    def preprocessText(self, text):
        """Clean up the text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def generateSummary(self, article, maxLength=100):
        """Generate summary for an article."""
        cleanedArticle = self.preprocessText(article)

        # T5 needs task prefix
        inputText = "summarize: " + cleanedArticle

        inputs = self.tokenizer.encode(inputText, return_tensors='pt', max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        # generate
        with torch.no_grad():
            summaryIds = self.model.generate(inputs, max_length=maxLength, num_beams=4, early_stopping=True)

        summary = self.tokenizer.decode(summaryIds[0], skip_special_tokens=True)

        return summary

    def evaluateOnTestSet(self, testArticles, testHighlights, numSamples=100):
        """Evaluate T5 and calculate ROUGE scores."""
        print(f"Evaluating T5 on {numSamples} test samples...")

        rouge = ROUGEScore()

        predictions = []
        references = []

        for idx, (article, highlight) in enumerate(zip(testArticles[:numSamples], testHighlights[:numSamples])):
            summary = self.generateSummary(article)
            predictions.append(summary)
            references.append(highlight)

            if (idx + 1) % 20 == 0:
                print(f"Generated {idx + 1}/{numSamples} summaries...")

        scores = rouge(predictions, references)

        print(f"\nT5 ROUGE Scores:")
        print(f"ROUGE-1: {scores['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2: {scores['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L: {scores['rougeL_fmeasure']:.4f}")

        return scores, predictions


# ============================================================================
# Q4: Extractive Summarization using PageRank
# ============================================================================

class ExtractiveSummarizer:
    """PageRank-based extractive summarization with GloVe."""

    def __init__(self, gloveFile):
        print("Loading GloVe embeddings...")
        self.gloveEmbeddings = self._loadGlove(gloveFile)
        print(f"Loaded {len(self.gloveEmbeddings)} word embeddings")

    def _loadGlove(self, gloveFile):
        """Load GloVe embeddings from file."""
        embeddings = {}

        with open(gloveFile, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector

        return embeddings

    def loadBbcData(self, dataPath):
        """Load BBC News Summary dataset."""
        print("Loading BBC News dataset...")

        import os

        articles = []
        summaries = []

        businessArticlePath = os.path.join(dataPath, 'News Articles', 'business')
        businessSummaryPath = os.path.join(dataPath, 'Summaries', 'business')

        articleFiles = sorted(os.listdir(businessArticlePath))

        for filename in articleFiles:
            with open(os.path.join(businessArticlePath, filename), 'r', encoding='latin-1') as f:
                article = f.read()
                articles.append(article)

            with open(os.path.join(businessSummaryPath, filename), 'r', encoding='latin-1') as f:
                summary = f.read()
                summaries.append(summary)

        print(f"Loaded {len(articles)} articles from business category")

        df = pd.DataFrame({
            'article': articles,
            'summary': summaries
        })

        return df

    def preprocessText(self, text):
        """Clean text but keep periods for sentence splitting."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
        text = ' '.join(text.split())
        return text

    def getSentenceEmbedding(self, sentence):
        """Get average GloVe embedding for sentence."""
        words = sentence.split()
        embeddings = []

        for word in words:
            if word in self.gloveEmbeddings:
                embeddings.append(self.gloveEmbeddings[word])

        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            # zero vector if no words found in GloVe
            return np.zeros(len(next(iter(self.gloveEmbeddings.values()))))

    def extractiveSummarize(self, article, topN=5):
        """Extract top N sentences using PageRank."""
        sentences = nltk.sent_tokenize(article)

        if len(sentences) <= topN:
            return sentences

        cleanSentences = [self.preprocessText(sent) for sent in sentences]

        # get embeddings for each sentence
        sentenceEmbeddings = []
        for sent in cleanSentences:
            embedding = self.getSentenceEmbedding(sent)
            sentenceEmbeddings.append(embedding)

        sentenceEmbeddings = np.array(sentenceEmbeddings)

        # build similarity matrix and run PageRank
        similarityMatrix = cosine_similarity(sentenceEmbeddings)

        graph = nx.from_numpy_array(similarityMatrix)
        scores = nx.pagerank(graph)

        # rank sentences by PageRank score
        rankedSentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        topSentences = [sent for score, sent in rankedSentences[:topN]]

        # return in original document order
        summaryIndices = []
        for topSent in topSentences:
            for i, origSent in enumerate(sentences):
                if topSent == origSent and i not in summaryIndices:
                    summaryIndices.append(i)
                    break

        summaryIndices.sort()
        summary = [sentences[i] for i in summaryIndices]

        return summary

    def evaluateOnDataset(self, dataframe):
        """Evaluate extractive summarization with ROUGE."""
        print("Evaluating extractive summarization...")

        rouge = ROUGEScore()

        predictions = []
        references = []

        for idx, row in dataframe.iterrows():
            summary = self.extractiveSummarize(row['article'], topN=5)
            summaryText = ' '.join(summary)

            predictions.append(summaryText)
            references.append(row['summary'])

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(dataframe)} articles...")

        scores = rouge(predictions, references)

        print(f"\nExtractive Summarization ROUGE Scores:")
        print(f"ROUGE-1: {scores['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2: {scores['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L: {scores['rougeL_fmeasure']:.4f}")

        return scores, predictions


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all questions."""
    print("=" * 70)
    print("NLP Assignment 8: Text Summarization")
    print("=" * 70)

    downloadNltkData()

    # Q1: Encoder-Decoder
    print("\n" + "=" * 70)
    print("QUESTION 1: Abstractive Summarization (Encoder-Decoder)")
    print("=" * 70)

    print("\nNote: Using small sample for demo - increase numSamples for full training")

    # Q2: T5
    print("\n" + "=" * 70)
    print("QUESTION 2: Abstractive Summarization (T5)")
    print("=" * 70)

    t5Summarizer = T5Summarizer()

    print("Loading test data...")
    testData = load_dataset('cnn_dailymail', '3.0.0', split='test', streaming=True)

    testArticles = []
    testHighlights = []

    for idx, sample in enumerate(testData):
        if idx >= 100:
            break
        testArticles.append(sample['article'])
        testHighlights.append(sample['highlights'])

    t5Scores, t5Predictions = t5Summarizer.evaluateOnTestSet(testArticles, testHighlights)

    # save samples for Q3
    print("\nSaving sample summaries for Q3 analysis...")
    with open('q3_samples.txt', 'w', encoding='utf-8') as f:
        for i in range(5):
            f.write(f"Article {i+1}:\n")
            f.write(f"{testArticles[i]}\n\n")
            f.write(f"Reference Summary:\n")
            f.write(f"{testHighlights[i]}\n\n")
            f.write(f"T5 Generated Summary:\n")
            f.write(f"{t5Predictions[i]}\n\n")
            f.write("=" * 70 + "\n\n")

    print("Samples saved to q3_samples.txt")

    # Q4: Extractive
    print("\n" + "=" * 70)
    print("QUESTION 4: Extractive Summarization (PageRank)")
    print("=" * 70)

    print("\nNote: Requires GloVe embeddings and BBC News dataset")
    print("Download GloVe: https://nlp.stanford.edu/projects/glove/")
    print("Download BBC News: https://www.kaggle.com/datasets/pariza/bbc-news-summary")

    # Done
    print("\n" + "=" * 70)
    print("Assignment 8 complete!")
    print("=" * 70)
    print("\nFor Q3 written analysis, see ASN8.txt")


if __name__ == "__main__":
    main()
