# NLP Assignment 8: Text Summarization

This project implements both abstractive and extractive text summarization approaches using various techniques including encoder-decoder models, pre-trained T5, and PageRank.

## 🚀 Projects

### Q1: Abstractive Summarization (Encoder-Decoder)
Custom encoder-decoder architecture using LSTM for generating abstractive summaries.

**Features:**
- Custom encoder-decoder with LSTM
- Beam search for text generation
- Trained on CNN/DailyMail dataset
- ROUGE score evaluation

**Tech Stack:** PyTorch, NLTK, NumPy

### Q2: Abstractive Summarization (T5)
Pre-trained T5 model for text summarization.

**Features:**
- Uses pre-trained T5-small model
- No training required
- State-of-the-art performance
- Easy to use with Huggingface transformers

**Tech Stack:** Transformers, PyTorch, NLTK

### Q4: Extractive Summarization (PageRank)
Graph-based extractive summarization using PageRank algorithm.

**Features:**
- GloVe word embeddings
- Cosine similarity for sentence ranking
- PageRank algorithm via NetworkX
- Extracts top sentences from article

**Tech Stack:** NetworkX, scikit-learn, NLTK, NumPy

## 📊 Dataset Information

**CNN/DailyMail:**
- ~300,000 news articles with summaries
- Used for Q1 and Q2
- Loaded via Huggingface datasets

**BBC News Summary:**
- Business category articles
- Used for Q4
- Available on Kaggle

**GloVe Embeddings:**
- Wikipedia 2014 + Gigaword 5
- Used for Q4
- Download from Stanford NLP

## 🛠️ Installation

```bash
# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download GloVe embeddings for Q4
# From: https://nlp.stanford.edu/projects/glove/
```

## ▶️ Running the Code

All three questions are implemented in a single file:

```bash
python assignment8.py
```

## 📝 Code Style

All code follows these conventions:
- **Variable naming:** camelCase with descriptive names
- **Comments:** Every 4-5 lines, written in casual language
- **Functions:** Well-documented with clear docstrings

## 🎯 Results Summary

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Encoder-Decoder | ~0.25 | ~0.10 | ~0.20 |
| T5-small | ~0.40 | ~0.18 | ~0.35 |
| Extractive (PageRank) | ~0.35 | ~0.15 | ~0.30 |

*Note: Actual scores may vary based on dataset size and training duration*

## 📋 Written Analysis

The written analysis for Q3 is in **ASN8.txt** and includes:
- Evaluation of 5 sample summaries
- Fluency ratings (1-5 scale)
- Coherence ratings (1-5 scale)
- Fact-preserving ratings (1-3 scale)
- Redundancy ratings (1-3 scale)
- Detailed observations and recommendations

## 🚧 Training Notes

- **Q1 Encoder-Decoder:** Training can take several hours on CPU. Use GPU for faster training. Small sample sizes are used for demonstration.
- **Q2 T5:** No training required, just inference. Can run on CPU but GPU recommended.
- **Q4 Extractive:** Fast processing, works well on CPU.

For faster experimentation:
- Use small data subsets (1-5% of data)
- Reduce batch sizes
- Use fewer epochs
- Consider Google Colab for GPU access

## 📁 File Structure

```
ASN8/
├── assignment8.py          # All coding questions (Q1, Q2, Q4)
├── ASN8.txt               # Written analysis for Q3
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── q3_samples.txt        # Sample summaries for analysis (generated)
```

## 🎓 Learning Outcomes

This project demonstrates:
1. **Abstractive Summarization:** Encoder-decoder architecture, beam search
2. **Pre-trained Models:** Using T5 for summarization
3. **Extractive Summarization:** PageRank algorithm, GloVe embeddings
4. **Evaluation Metrics:** ROUGE scores for summarization quality
5. **Deep Learning:** PyTorch implementation

## 🔮 Future Improvements

- Add attention mechanism to encoder-decoder
- Implement BART or PEGASUS models
- Try different beam search parameters
- Experiment with hybrid extractive-abstractive approaches
- Fine-tune T5 on domain-specific data

## 📄 License

Educational project for CS 421 - Natural Language Processing
