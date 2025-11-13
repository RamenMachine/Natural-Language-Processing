# Quick Start Guide - Assignment 4

Get up and running with Assignment 4 in just a few minutes!

---

## Prerequisites

- Python 3.8 or higher installed
- Internet connection (for downloading datasets and Word2Vec)

---

## Installation (5 minutes)

### 1. Install Dependencies

```bash
# Navigate to ASN4 folder
cd ASN4

# Install all required packages
pip install -r requirements.txt
```

**Note:** This will install TensorFlow, Keras, Gensim, and other NLP libraries (~500MB).

---

## Running the Assignment

### Option 1: Run Complete Assignment (Recommended)

Run all three questions sequentially:

```bash
python HW4.py
```

**What it does:**
1. ✅ Q1: Builds TF-IDF vectorizer and computes cosine similarity
2. ✅ Q2: Calculates PPMI for word associations
3. ✅ Q3: Trains LSTM model for Named Entity Recognition

**Expected runtime:**
- First run: 15-30 minutes (downloads Word2Vec embeddings ~1.5GB)
- Subsequent runs: 5-10 minutes

**Output:**
```
================================================================================
QUESTION 1: TF-IDF AND COSINE SIMILARITY
================================================================================
Loading CoNLL2003 dataset...
✓ Vocabulary size: 5234
✓ Fitted on 1000 documents
...
Computing cosine similarities:
  Sentence 1: 'I love football'
  Sentence 2: 'I do not love football'
  Cosine Similarity: 0.7854
...

================================================================================
QUESTION 2: PPMI CALCULATION
================================================================================
Example: words = ['a', 'b', 'a', 'c']
PPMI Results:
  ('a', 'b'): 0.5850
...

================================================================================
QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM
================================================================================
Loading CoNLL2003 dataset...
Building LSTM model...
Training model (10 epochs)...
Epoch 1/10
...
================================================================================
RESULTS:
================================================================================
  Accuracy:           0.9420
  Macro Precision:    0.8750
  Macro Recall:       0.8580
  Macro F1-Score:     0.8660
================================================================================
```

---

### Option 2: Interactive Jupyter Notebook

For a more interactive experience with visualizations:

```bash
jupyter notebook assignment4_showcase.ipynb
```

**What you'll see:**
- 📊 Theory explanations
- 💻 Step-by-step code execution
- 📈 Visualizations (heatmaps, bar charts, confusion matrices)
- 📝 Analysis and insights
- 🎨 Beautiful formatting

**Tip:** Run cells in order (Shift + Enter) to execute code.

---

### Option 3: View the Frontend Demo

Open the aesthetic HTML showcase:

```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

Or use Python's HTTP server:

```bash
python -m http.server 8000
# Visit: http://localhost:8000/index.html
```

---

## What Gets Downloaded Automatically

On first run, the following will be downloaded:

1. **CoNLL2003 Dataset** (~5MB)
   - Source: Hugging Face Datasets
   - Contains: Named entity recognition training data

2. **Word2Vec Embeddings** (~1.5GB) ⚠️
   - Source: Google News corpus
   - Contains: 300-dimensional word vectors
   - **This is the largest download!**

**Subsequent runs:** These are cached locally, so no re-download needed.

---

## Quick Test (30 seconds)

Want to quickly verify everything works? Run individual questions:

### Test Q1 Only:

```python
# In Python:
from HW4 import question1_tfidf_cosine_similarity
question1_tfidf_cosine_similarity()
```

### Test Q2 Only:

```python
from HW4 import question2_ppmi
question2_ppmi()
```

### Test Q3 Only (takes longer):

```python
from HW4 import question3_ner_lstm
question3_ner_lstm()
```

---

## File Overview

| File | Purpose | Size |
|------|---------|------|
| `HW4.py` | Main implementation | ~700 lines |
| `assignment4_showcase.ipynb` | Interactive notebook | Rich visualizations |
| `index.html` | Frontend demo | Self-contained |
| `requirements.txt` | Dependencies | Python packages |
| `README.md` | Full documentation | Comprehensive guide |

---

## Common Issues & Solutions

### Issue 1: "Module not found"

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue 2: "Download is too slow"

**Solution:**
The Word2Vec model is 1.5GB. On slow connections:
- Use a faster network if available
- Or skip Word2Vec (code will use random embeddings):
  ```python
  # In HW4.py, find this line and comment it out:
  # word2vec_model = api.load("word2vec-google-news-300")
  ```

### Issue 3: "Out of memory"

**Solution:**
Reduce the dataset size in `HW4.py`:
```python
# Change this line:
sentences, tags, word2idx, tag2idx = prepare_ner_data(dataset, max_samples=5000)

# To:
sentences, tags, word2idx, tag2idx = prepare_ner_data(dataset, max_samples=1000)
```

### Issue 4: "Jupyter not found"

**Solution:**
```bash
pip install jupyter notebook
```

---

## Expected Results

### Q1: TF-IDF Cosine Similarity

| Pair | Similarity | Status |
|------|-----------|--------|
| "I love football" vs "I do not love football" | ~0.78 | ✅ |
| "I follow cricket" vs "I follow baseball" | ~0.89 | ✅ |

### Q2: PPMI

Should see word pair associations with positive PMI values.

### Q3: NER Performance

| Metric | Expected Range | Target |
|--------|---------------|--------|
| Accuracy | 90-95% | 94%+ |
| Precision | 85-90% | 87%+ |
| Recall | 85-90% | 85%+ |
| F1-Score | 85-90% | 86%+ |

---

## Next Steps

After running the assignment:

1. ✅ Review the output and results
2. ✅ Check the Jupyter notebook for visualizations
3. ✅ Customize the frontend with your personal information
4. ✅ Deploy to GitHub Pages (see `GITHUB_PAGES_SETUP.md`)
5. ✅ Share your project with recruiters!

---

## Time Estimates

| Task | Time |
|------|------|
| Installation | 5 min |
| First run (with downloads) | 15-30 min |
| Subsequent runs | 5-10 min |
| Reviewing results | 10 min |
| Customizing frontend | 5 min |
| **Total** | **40-60 min** |

---

## Need Help?

- 📖 Read the full [README.md](README.md)
- 🚀 Check [GITHUB_PAGES_SETUP.md](GITHUB_PAGES_SETUP.md) for deployment
- 💻 View code comments in `HW4.py`
- 📓 Run the Jupyter notebook for detailed explanations

---

## Project Structure

```
ASN4/
├── HW4.py                     ← Run this first!
├── assignment4_showcase.ipynb ← Interactive notebook
├── index.html                 ← Frontend demo
├── requirements.txt           ← Install dependencies
├── README.md                  ← Full documentation
├── QUICKSTART.md             ← This file
└── GITHUB_PAGES_SETUP.md     ← Deployment guide
```

---

## One-Command Setup (Advanced)

For experienced users, here's a one-liner:

```bash
pip install -r requirements.txt && python HW4.py
```

This installs dependencies and runs the complete assignment.

---

**Ready? Let's go!**

```bash
python HW4.py
```

**Happy coding! 🚀**
