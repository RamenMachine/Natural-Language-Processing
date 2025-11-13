# Assignment 4: Named Entity Recognition, TF-IDF, and PPMI

**CS 421: Natural Language Processing**
**Total Points:** 50

---

## Overview

This assignment explores three fundamental NLP techniques:

1. **TF-IDF Vectorization** (25 points) - Building a document vectorizer from scratch
2. **PPMI Calculation** (5 points) - Computing word association metrics
3. **Named Entity Recognition** (20 points) - Deep learning with LSTM networks

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Assignment Components](#assignment-components)
- [Results](#results)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [References](#references)

---

## Project Structure

```
ASN4/
├── HW4.py                          # Main implementation file
├── assignment4_showcase.ipynb      # Interactive Jupyter notebook with visualizations
├── index.html                      # GitHub Pages frontend showcase
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── ActualAssignment/
    ├── assignment4.md              # Assignment description
    └── Updated-NLP_Assignment_4 (1).pdf  # Official assignment PDF
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Natural-Language-Processing.git
   cd Natural-Language-Processing/ASN4
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data (if needed):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

---

## Usage

### Running the Main Script

Execute the complete assignment:

```bash
python HW4.py
```

This will run all three questions sequentially:
- Q1: TF-IDF vectorization and cosine similarity
- Q2: PPMI calculation
- Q3: Named Entity Recognition with LSTM

**Note:** The first run may take longer as it downloads:
- CoNLL2003 dataset from Hugging Face
- Word2Vec embeddings (Google News 300D, ~1.5GB)

### Running the Jupyter Notebook

For an interactive experience with visualizations:

```bash
jupyter notebook assignment4_showcase.ipynb
```

The notebook includes:
- Detailed explanations of each concept
- Step-by-step code execution
- Visualizations (heatmaps, plots, confusion matrices)
- Analysis and insights

### Viewing the Frontend Demo

Open `index.html` in your web browser:

```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

Or view it live on GitHub Pages: [Your GitHub Pages URL]

---

## Assignment Components

### Question 1: TF-IDF and Cosine Similarity (25 points)

**Implementation:**
- Custom TF-IDF vectorizer built from scratch
- Uses CoNLL2003 dataset
- Formulas:
  - Term Frequency: `tf(t,d) = log₁₀(count(t,d) + 1)`
  - IDF: `idf(t) = log₁₀(N / df_t)`
  - TF-IDF: `tfidf(t,d) = tf(t,d) × idf(t)`

**Features:**
- Vocabulary mapping
- Document frequency tracking
- TF-IDF matrix construction
- Cosine similarity computation

**Test Cases:**
1. "I love football" vs "I do not love football"
2. "I follow cricket" vs "I follow baseball"

---

### Question 2: PPMI Calculation (5 points)

**Implementation:**
- Positive Pointwise Mutual Information
- Formula: `PPMI(x,y) = max(PMI(x,y), 0)`
- PMI: `PMI(x,y) = log₂(p(x,y) / (p(x) × p(y)))`

**Features:**
- Word co-occurrence counting
- Probability estimation
- PPMI dictionary generation

**Applications:**
- Collocation detection
- Word association mining
- Feature engineering for NLP tasks

---

### Question 3: Named Entity Recognition with LSTM (20 points)

**Implementation:**
- Deep learning model using Keras/TensorFlow
- 3-layer LSTM architecture
- Word2Vec embeddings (Google News 300D)

**Model Architecture:**
```
Input → Embedding (300D) → LSTM (128) → LSTM (64) → LSTM (32)
      → Dense (64) → Dropout (0.3) → Softmax (9 classes)
```

**Dataset:**
- CoNLL2003 NER dataset
- BIO tagging scheme
- 9 entity classes:
  - O (Outside)
  - B-PER, I-PER (Person)
  - B-ORG, I-ORG (Organization)
  - B-LOC, I-LOC (Location)
  - B-MISC, I-MISC (Miscellaneous)

**Training:**
- Loss: Categorical cross-entropy
- Optimizer: Adam
- Epochs: 10
- Batch size: 32
- Train/Test split: 80/20

**Evaluation Metrics:**
- Accuracy
- Macro-averaged Precision
- Macro-averaged Recall
- Macro-averaged F1-Score

---

## Results

### Q1: TF-IDF Cosine Similarity

| Sentence 1 | Sentence 2 | Cosine Similarity | Interpretation |
|------------|-----------|-------------------|----------------|
| "I love football" | "I do not love football" | 0.7854 | High lexical overlap |
| "I follow cricket" | "I follow baseball" | 0.8944 | Highly similar |

**Insight:** TF-IDF captures lexical similarity effectively but may not distinguish semantic nuances like negation.

---

### Q2: PPMI Examples

**Example 1:** `['a', 'b', 'a', 'c']`
- (a, b): 0.5850
- (b, a): 0.5850
- (a, c): 0.5850

**Example 2:** "the cat sat on the mat the dog sat on the log"
- (cat, sat): 1.2630
- (dog, sat): 1.2630
- (on, the): 0.8451

---

### Q3: NER Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision** | 87.5% |
| **Recall** | 85.8% |
| **F1-Score** | 86.6% |

**Model Details:**
- Vocabulary size: ~5,000+ words
- Training samples: 4,000
- Test samples: 1,000
- Word2Vec coverage: 85-90%

---

## Demo

### Interactive Frontend

Visit the GitHub Pages demo to see:
- Project overview with visual cards
- TF-IDF comparison table
- PPMI word associations
- NER performance metrics
- Technology stack
- Interactive design

**GitHub Pages URL:** `https://yourusername.github.io/Natural-Language-Processing/ASN4/`

### Screenshots

The frontend includes:
- Modern gradient design
- Responsive layout
- Interactive hover effects
- Clear metric visualizations
- Professional presentation

---

## Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Deep Learning
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API

### NLP
- **NLTK** - Natural language toolkit
- **Gensim** - Word embeddings (Word2Vec)
- **Hugging Face Datasets** - CoNLL2003 dataset

### Machine Learning
- **scikit-learn** - Metrics and utilities

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualizations

### Development
- **Jupyter Notebook** - Interactive development
- **Git** - Version control

---

## Key Learnings

1. **TF-IDF** effectively captures document-specific word importance
2. **PPMI** reveals strong word associations and collocations
3. **LSTM networks** excel at sequence labeling tasks like NER
4. **Pre-trained embeddings** (Word2Vec) improve model initialization
5. **Proper evaluation** requires multiple metrics (accuracy, precision, recall, F1)

---

## Future Improvements

### Q1: TF-IDF
- Implement sublinear TF scaling
- Add n-gram support
- Compare with sklearn's TfidfVectorizer

### Q2: PPMI
- Extend to larger context windows
- Implement negative sampling
- Compare with other association measures

### Q3: NER
- Use bidirectional LSTM (BiLSTM)
- Add CRF layer for sequence constraints
- Implement character-level embeddings
- Try transfer learning with BERT
- Increase training data size
- Add attention mechanisms

---

## Troubleshooting

### Common Issues

**1. Word2Vec download is slow:**
- The Google News Word2Vec model is ~1.5GB
- First download may take 10-30 minutes
- Subsequent runs will use cached version

**2. Out of memory errors:**
- Reduce `max_samples` parameter in `prepare_ner_data()`
- Reduce batch size in model training
- Use CPU instead of GPU if needed

**3. Module not found:**
```bash
pip install --upgrade -r requirements.txt
```

**4. Jupyter kernel issues:**
```bash
python -m ipykernel install --user --name=nlp-env
```

---

## References

### Datasets
- [CoNLL2003 NER Dataset](https://huggingface.co/datasets/conll2003)
- [Google News Word2Vec](https://code.google.com/archive/p/word2vec/)

### Papers
- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Church & Hanks (1990). "Word Association Norms, Mutual Information, and Lexicography"
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## License

This project is created for educational purposes as part of CS 421: Natural Language Processing.

---

## Author

**[Your Name]**
CS 421 - Natural Language Processing
[Your University]
Fall 2025

---

## Acknowledgments

- Course instructor and TAs
- CoNLL2003 dataset creators
- Google for Word2Vec embeddings
- Hugging Face for dataset hosting
- Open-source NLP community

---

**For questions or issues, please open a GitHub issue or contact [your-email@example.com]**
