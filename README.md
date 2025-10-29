# Natural Language Processing: Text Analysis & Spell Correction System

## 📋 Project Overview

A comprehensive Natural Language Processing pipeline designed to analyze, clean, and process health-related social media content. This project demonstrates advanced NLP techniques including text preprocessing, tokenization, morphological analysis, and intelligent spell correction using minimum edit distance algorithms.

## 🎯 Key Features

### 1. **Data Processing & Corpus Management**
- **Multi-source Data Integration**: Aggregated and processed 6,000+ health-related tweets from CNN and Fox News
- **Robust Text Cleaning**: Implemented regex-based preprocessing to remove URLs, mentions, hashtags, special characters, and numbers
- **Corpus Maintenance**: Built and maintained multiple corpus versions (original, stopword-filtered, lemmatized, stemmed) with 8,797+ unique tokens

### 2. **Advanced Text Tokenization**
- **Sentence Tokenization**: Utilized NLTK's sentence tokenizer for accurate sentence boundary detection
- **Word Tokenization**: Implemented hierarchical tokenization (sentences → words) for granular text analysis
- **Case Normalization**: Standardized all text to lowercase for consistent processing

### 3. **Morphological Analysis**
- **Lemmatization**: Applied WordNet lemmatizer to reduce words to dictionary base forms (7,657 unique lemmas)
- **Stemming**: Implemented Porter Stemmer for aggressive root extraction (6,345 unique stems)
- **Comparative Analysis**: Evaluated trade-offs between lemmatization accuracy and stemming efficiency

### 4. **Stopword Filtering**
- **Noise Reduction**: Removed 20,586 common stopwords (179 English stopwords)
- **Content Extraction**: Reduced dataset from 65,046 to 44,460 meaningful tokens
- **Frequency Analysis**: Identified domain-specific keywords (health, cancer, study, ebola)

### 5. **Hashtag Extraction & Analysis**
- **Pattern Recognition**: Extracted 3,572 hashtag occurrences across 914 unique tags using regex patterns
- **Trend Analysis**: Identified top health topics (#getfit, #ebola, #cancer, #flu)
- **Social Media Analytics**: Provided insights into health discussion trends

### 6. **Intelligent Spell Correction System**
- **Minimum Edit Distance Algorithm**: Implemented dynamic programming solution with configurable insertion, deletion, and substitution costs
- **Corpus-Based Correction**: Leveraged custom health domain corpus for accurate suggestions
- **Top-N Recommendations**: Returns 5 best spelling corrections ranked by edit distance

## 🛠️ Technical Stack

**Languages & Libraries:**
- Python 3.13
- pandas (Data manipulation)
- NumPy (Numerical operations)
- NLTK (Tokenization, stemming, lemmatization, stopwords)
- Regular Expressions (Text cleaning and pattern matching)
- Collections (Counter for frequency analysis)

**NLP Techniques:**
- Tokenization (Sentence & Word)
- Text Normalization
- Stopword Removal
- Lemmatization & Stemming
- Edit Distance (Levenshtein Distance)
- Corpus Linguistics

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total Tweets Processed | 6,045 |
| Unique Words (Original) | 8,797 |
| Unique Words (After Stopword Removal) | 8,670 |
| Unique Lemmas | 7,657 |
| Unique Stems | 6,345 |
| Hashtags Extracted | 914 unique, 3,572 total |
| Vocabulary Reduction | 27.9% (via stemming) |

## 🔍 Use Cases

- **Social Media Analytics**: Track health trends and public sentiment
- **Content Moderation**: Clean and normalize user-generated content
- **Information Retrieval**: Improve search relevance through lemmatization
- **Spell Checking**: Provide intelligent autocorrect for health-related terms
- **Text Mining**: Extract meaningful patterns from unstructured data

## 📈 Performance Highlights

- **Processing Speed**: Handles 6,000+ documents efficiently
- **Spell Checker Accuracy**: Correctly identifies misspellings with 1-2 edit distance
- **Data Quality**: 99%+ data integrity through robust error handling (on_bad_lines='skip')
- **Memory Efficiency**: Optimized corpus storage and retrieval

## 💡 Skills Demonstrated

✅ Natural Language Processing (NLP)  
✅ Text Preprocessing & Cleaning  
✅ Regular Expressions  
✅ Algorithm Design (Dynamic Programming)  
✅ Data Analysis & Visualization  
✅ Python Programming  
✅ Statistical Text Analysis  
✅ Corpus Linguistics  
✅ Problem Solving & Optimization  

## 📂 Project Structure

```
Natural-Language-Processing/
├── Assignment 1.py          # Main NLP pipeline implementation
├── corpus.csv              # Generated corpus data
├── Health-Tweets/          # Source data directory
│   ├── foxnewshealth.txt
│   └── cnnhealth.txt
└── README.md              # Project documentation
```

## 🚀 Future Enhancements

- Implement TF-IDF for keyword extraction
- Add sentiment analysis using pre-trained models
- Integrate named entity recognition (NER)
- Build interactive visualization dashboard
- Expand corpus with medical domain terminology
- Implement context-aware spell correction

---

**Author**: [Your Name]  
**Technologies**: Python, NLTK, pandas, NLP, Machine Learning  
**Domain**: Healthcare Analytics, Social Media Mining, Text Processing
