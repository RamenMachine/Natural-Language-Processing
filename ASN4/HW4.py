"""
CS 421: Natural Language Processing
Assignment 4: Named Entity Recognition, TF-IDF, and PPMI

This assignment implements:
1. TF-IDF vectorizer from scratch with cosine similarity
2. Positive Pointwise Mutual Information (PPMI) calculation
3. Named Entity Recognition using LSTM neural networks
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
import math
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# For Q3: Deep Learning
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import gensim.downloader as api
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Note: Keras/TensorFlow not available. Q3 will not run.")


# ============================================================================
# QUESTION 1: TF-IDF AND COSINE SIMILARITY (25 points)
# ============================================================================

class TfidfVectorizer:
    """
    Custom TF-IDF Vectorizer built from scratch

    Implements the formulas:
    - TF(t,d) = log10(count(t,d) + 1)
    - IDF(t) = log10(N / df_t)
    - TF-IDF(t,d) = TF(t,d) * IDF(t)
    """

    def __init__(self):
        self.word_to_index = {}  # Q1.2: word-to-index dictionary
        self.doc_freq = {}       # Q1.3: document frequency dictionary
        self.num_docs = 0        # Total number of documents (N)
        self.vocabulary = set()  # Q1.1: vocabulary set

    def build_vocabulary(self, df: pd.DataFrame):
        """
        Q1.1: Create a vocabulary set of all words appearing in the tokens column

        Args:
            df: DataFrame with 'tokens' column
        """
        self.vocabulary = set()
        for tokens in df['tokens']:
            self.vocabulary.update(tokens)
        print(f"Vocabulary size: {len(self.vocabulary)} words")

    def create_word_to_index(self):
        """
        Q1.2: Create word-to-index dictionary

        Assigns unique integer index to each word, incremented by one for each new word.
        """
        self.word_to_index = {}
        for idx, word in enumerate(sorted(self.vocabulary), start=1):
            self.word_to_index[word] = idx
        print(f"Word-to-index dictionary created with {len(self.word_to_index)} entries")

    def create_doc_frequency(self, df: pd.DataFrame):
        """
        Q1.3: Create document frequency dictionary

        key = word
        value = number of documents (rows) in which the word occurs

        Args:
            df: DataFrame with 'tokens' column
        """
        self.doc_freq = defaultdict(int)
        for tokens in df['tokens']:
            unique_words = set(tokens)
            for word in unique_words:
                self.doc_freq[word] += 1
        self.doc_freq = dict(self.doc_freq)
        print(f"Document frequency dictionary created")

    def term_frequency(self, term: str, document: List[str]) -> float:
        """
        Q1.4: Calculate term frequency

        tf(t,d) = log10(count(t,d) + 1)

        Args:
            term: The word to calculate frequency for
            document: List of words in the document

        Returns:
            Term frequency value
        """
        count = document.count(term)
        return math.log10(count + 1)

    def idf(self, word: str) -> float:
        """
        Q1.5: Calculate inverse document frequency

        idf(t) = log10(N / df_t)

        If the term is not in the dictionary, assume df_t = 1 to avoid division by 0.

        Args:
            word: The word to calculate IDF for

        Returns:
            IDF value
        """
        df_t = self.doc_freq.get(word, 1)  # Assume df_t = 1 if not found
        return math.log10(self.num_docs / df_t)

    def tfidf(self, document: List[str]) -> np.ndarray:
        """
        Q1.6: Compute TF-IDF for one document

        Creates a NumPy array of all zeros with shape (len(vocab), ).
        For each word in the document:
        - Calculate TF using term_frequency
        - Calculate IDF using idf
        - Compute TF * IDF
        - Update the value in the array at the index of the word

        Args:
            document: List of words in the document

        Returns:
            NumPy array of TF-IDF values
        """
        vector = np.zeros(len(self.vocabulary))

        for word in document:
            if word in self.word_to_index:
                # Get the index (subtract 1 since indices start at 1)
                idx = self.word_to_index[word] - 1
                tf = self.term_frequency(word, document)
                idf_val = self.idf(word)
                vector[idx] = tf * idf_val

        return vector

    def fit(self, df: pd.DataFrame):
        """
        Fit the vectorizer on a DataFrame

        Args:
            df: DataFrame with 'tokens' column
        """
        self.num_docs = len(df)

        # Q1.1: Create vocabulary set
        self.build_vocabulary(df)

        # Q1.2: Create word-to-index dictionary
        self.create_word_to_index()

        # Q1.3: Create document frequency dictionary
        self.create_doc_frequency(df)

        print(f"Fitted on {self.num_docs} documents")

    def build_tfidf_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Q1.7: Build the TF-IDF matrix for the dataset

        Creates an empty NumPy array and appends TF-IDF vectors for all documents.

        Args:
            df: DataFrame with 'tokens' column

        Returns:
            TF-IDF matrix (num_docs x vocab_size)
        """
        tfidf_matrix = []

        for tokens in df['tokens']:
            tfidf_vector = self.tfidf(tokens)
            tfidf_matrix.append(tfidf_vector)

        return np.array(tfidf_matrix)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            df: DataFrame with 'tokens' column

        Returns:
            TF-IDF matrix
        """
        self.fit(df)
        return self.build_tfidf_matrix(df)


def cosine_similarity(v: np.ndarray, w: np.ndarray) -> float:
    """
    Q1.8: Calculate cosine similarity between two vectors

    cos(v, w) = (v · w) / (||v|| * ||w||)
               = sum(v_i * w_i) / (sqrt(sum(v_i^2)) * sqrt(sum(w_i^2)))

    Args:
        v: First vector
        w: Second vector

    Returns:
        Cosine similarity value between -1 and 1
    """
    dot_product = np.sum(v * w)
    norm_v = np.sqrt(np.sum(v ** 2))
    norm_w = np.sqrt(np.sum(w ** 2))

    if norm_v == 0 or norm_w == 0:
        return 0.0

    return dot_product / (norm_v * norm_w)


def question_one():
    """
    Q1: Implement TF-IDF vectorizer and compute cosine similarity
    """
    print("\n" + "=" * 80)
    print("QUESTION 1: TF-IDF AND COSINE SIMILARITY")
    print("=" * 80 + "\n")

    # Q1.1: Load the dataset
    print("Loading CoNLL2003 dataset...")
    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # Load the dataset in a pandas DataFrame
    train_data = dataset['train']

    # Convert to DataFrame
    df = pd.DataFrame({
        'tokens': train_data['tokens'],
        'ner_tags': train_data['ner_tags']
    })

    # Drop all columns except tokens and ner_tags (already done by selecting only these)
    print(f"Dataset loaded into DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Use a subset for efficiency
    df = df.head(1000)
    print(f"Using first {len(df)} rows for processing\n")

    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer (creates vocabulary, word-to-index, doc freq)
    vectorizer.fit(df)

    # Q1.7: Build TF-IDF matrix
    print("\nBuilding TF-IDF matrix...")
    tfidf_matrix = vectorizer.build_tfidf_matrix(df)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"Matrix dimensions: {tfidf_matrix.shape[0]} documents x {tfidf_matrix.shape[1]} features\n")

    # Q1.8: Cosine similarity examples
    print("=" * 80)
    print("COSINE SIMILARITY ANALYSIS")
    print("=" * 80 + "\n")

    # Example 1
    print("Example 1:")
    sent1 = "I love football"
    sent2 = "I do not love football"
    tokens1 = sent1.lower().split()
    tokens2 = sent2.lower().split()

    vec1 = vectorizer.tfidf(tokens1)
    vec2 = vectorizer.tfidf(tokens2)
    similarity1 = cosine_similarity(vec1, vec2)

    print(f"  Sentence 1: '{sent1}'")
    print(f"  Sentence 2: '{sent2}'")
    print(f"  Cosine Similarity: {similarity1:.6f}")
    print(f"\n  Observation: Despite sharing many words ('I', 'love', 'football'),")
    print(f"  the similarity is affected by the negation word 'not'. The TF-IDF")
    print(f"  approach captures lexical overlap but may not fully capture semantic")
    print(f"  opposition caused by negation.\n")

    # Example 2
    print("Example 2:")
    sent3 = "I follow cricket"
    sent4 = "I follow baseball"
    tokens3 = sent3.lower().split()
    tokens4 = sent4.lower().split()

    vec3 = vectorizer.tfidf(tokens3)
    vec4 = vectorizer.tfidf(tokens4)
    similarity2 = cosine_similarity(vec3, vec4)

    print(f"  Sentence 1: '{sent3}'")
    print(f"  Sentence 2: '{sent4}'")
    print(f"  Cosine Similarity: {similarity2:.6f}")
    print(f"\n  Observation: These sentences share the same structure and two out of")
    print(f"  three words ('I', 'follow'). The difference lies in 'cricket' vs")
    print(f"  'baseball', both sports terms. The similarity score reflects the")
    print(f"  structural and lexical similarity.\n")

    return vectorizer, tfidf_matrix, df


# ============================================================================
# QUESTION 2: PPMI (POSITIVE POINTWISE MUTUAL INFORMATION) (5 points)
# ============================================================================

def calculate_ppmi(words: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Q2: Calculate Positive Pointwise Mutual Information for word pairs

    PPMI(x, y) = max(PMI(x, y), 0)
    PMI(x, y) = log2(p(x, y) / (p(x) * p(y)))

    Where:
    - p(x) = probability of word x occurring in the text
    - p(y) = probability of word y occurring in the text
    - p(x, y) = probability of x and y occurring together (adjacent)

    Args:
        words: List of strings representing the words in the text

    Returns:
        Dictionary mapping tuples of words to their PPMI values
    """
    # Count individual word occurrences
    word_counts = Counter(words)
    total_words = len(words)

    # Count adjacent word pairs (co-occurrences)
    pair_counts = Counter()
    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        pair_counts[pair] += 1

    total_pairs = sum(pair_counts.values())

    # Calculate PPMI for each pair
    ppmi_dict = {}

    for (word_x, word_y), pair_count in pair_counts.items():
        # Calculate probabilities
        p_x = word_counts[word_x] / total_words
        p_y = word_counts[word_y] / total_words
        p_xy = pair_count / total_pairs

        # Calculate PMI
        if p_x > 0 and p_y > 0 and p_xy > 0:
            pmi = math.log2(p_xy / (p_x * p_y))
            # PPMI = max(PMI, 0)
            ppmi = max(pmi, 0)
            ppmi_dict[(word_x, word_y)] = ppmi

    return ppmi_dict


def question_two():
    """
    Q2: Demonstrate PPMI calculation with examples
    """
    print("\n" + "=" * 80)
    print("QUESTION 2: PPMI CALCULATION")
    print("=" * 80 + "\n")

    # Example from assignment specification
    example_words = ['a', 'b', 'a', 'c']
    ppmi_result = calculate_ppmi(example_words)

    print("Example: words = ['a', 'b', 'a', 'c']")
    print("\nPPMI Results:")
    print("-" * 40)
    for pair, ppmi_value in sorted(ppmi_result.items()):
        print(f"  {pair}: {ppmi_value:.6f}")
    print("-" * 40)

    # Demonstrate with additional example
    print("\n" + "=" * 80)
    print("Additional Example for Demonstration:")
    print("=" * 80 + "\n")

    text_example = "the cat sat on the mat the dog sat on the log".split()
    ppmi_result2 = calculate_ppmi(text_example)

    print(f"Text: '{' '.join(text_example)}'")
    print("\nPPMI Results (sorted by PPMI value):")
    print("-" * 50)
    for pair, ppmi_value in sorted(ppmi_result2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair[0]:8s} -> {pair[1]:8s} : {ppmi_value:.6f}")
    print("-" * 50)

    return ppmi_result, ppmi_result2


# ============================================================================
# QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM (20 points)
# ============================================================================

class NerLstmModel:
    """
    Named Entity Recognition model using LSTM

    Architecture (as specified):
    - Embedding layer (using Word2Vec embeddings)
    - 3 LSTM layers
    - 1 fully-connected (dense) layer
    - 1 final fully-connected layer with softmax activation and 9 outputs
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 max_length: int = 100, num_tags: int = 9):
        """
        Initialize NER model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings (300 for Word2Vec)
            max_length: Maximum sequence length
            num_tags: Number of NER tags (9 for CoNLL2003)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_tags = num_tags
        self.model = None

    def build_model(self, embedding_matrix=None):
        """
        Q3.2: Build the LSTM model architecture

        Creates a Sequential model with:
        - Embedding layer
        - 3 LSTM layers
        - 1 fully-connected (dense) layer
        - 1 final fully-connected layer with softmax activation and 9 outputs

        Args:
            embedding_matrix: Pre-trained embedding matrix from Word2Vec
        """
        model = Sequential()

        # Embedding layer
        if embedding_matrix is not None:
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[embedding_matrix],
                input_length=self.max_length,
                trainable=False,  # Use pre-trained embeddings
                mask_zero=True
            ))
        else:
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                mask_zero=True
            ))

        # 3 LSTM layers
        model.add(LSTM(128, return_sequences=True, dropout=0.2))
        model.add(LSTM(64, return_sequences=True, dropout=0.2))
        model.add(LSTM(32, return_sequences=True, dropout=0.2))

        # 1 fully-connected (dense) layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))

        # Final fully-connected layer with softmax activation and 9 outputs
        model.add(Dense(self.num_tags, activation='softmax'))

        # Q3.3: Compile with cross entropy loss and Adam optimizer
        model.compile(
            loss='categorical_crossentropy',  # Cross entropy loss
            optimizer='adam',                  # Adam optimizer (recommended)
            metrics=['accuracy']
        )

        self.model = model
        return model

    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()


def prepare_ner_data(dataset):
    """
    Prepare CoNLL2003 data for NER training

    Args:
        dataset: CoNLL2003 dataset

    Returns:
        DataFrame with 'tokens' and 'ner_tags' columns
    """
    train_data = dataset['train']

    # Create DataFrame with only tokens and ner_tags
    df = pd.DataFrame({
        'tokens': [[token.lower() for token in tokens] for tokens in train_data['tokens']],
        'ner_tags': train_data['ner_tags']
    })

    return df


def create_embedding_matrix(word_to_idx: Dict, word2vec_model, embedding_dim: int = 300) -> np.ndarray:
    """
    Create embedding matrix from Word2Vec model

    Args:
        word_to_idx: Word to index mapping
        word2vec_model: Loaded Word2Vec model (Google News corpus)
        embedding_dim: Embedding dimension (300)

    Returns:
        Embedding matrix
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    for word, idx in word_to_idx.items():
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]
            found += 1
        else:
            # Random initialization for words not in Word2Vec
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

    print(f"Found {found}/{vocab_size} words in Word2Vec model ({100*found/vocab_size:.2f}%)")
    return embedding_matrix


def question_three():
    """
    Q3: Implement Named Entity Recognition using LSTM
    """
    print("\n" + "=" * 80)
    print("QUESTION 3: NAMED ENTITY RECOGNITION USING LSTM")
    print("=" * 80 + "\n")

    if not KERAS_AVAILABLE:
        print("ERROR: Keras/TensorFlow not available. Please install required packages.")
        return None

    # Load dataset
    print("Loading CoNLL2003 dataset...")
    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # Prepare data
    print("Preparing data...")
    df = prepare_ner_data(dataset)
    print(f"Total samples: {len(df)}")

    # Use subset for efficiency
    df = df.head(5000)
    print(f"Using {len(df)} samples for training\n")

    # Build vocabulary
    all_words = set()
    for tokens in df['tokens']:
        all_words.update(tokens)

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(sorted(all_words), start=2):
        word_to_idx[word] = idx

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    print(f"Vocabulary size: {len(word_to_idx)}")

    # Tag mapping (9 NER tags: 0-8)
    tag_to_idx = {i: i for i in range(9)}
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}

    print(f"Number of NER tags: {len(tag_to_idx)}")

    # Determine max length
    max_length = max(len(tokens) for tokens in df['tokens'])
    max_length = min(max_length, 100)
    print(f"Maximum sequence length: {max_length}\n")

    # Convert sentences to sequences
    X_sequences = []
    y_sequences = []

    for tokens, tags in zip(df['tokens'], df['ner_tags']):
        # Convert words to indices
        sent_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]
        X_sequences.append(sent_indices)
        y_sequences.append(tags)

    # Pad sequences
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', value=word_to_idx['<PAD>'])
    y_padded = pad_sequences(y_sequences, maxlen=max_length, padding='post', value=0)

    # Convert tags to categorical (one-hot encoding)
    y_categorical = np.array([to_categorical(seq, num_classes=9) for seq in y_padded])

    # Q3.1: Split data 80/20 using sklearn
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_test)} (20%)\n")

    # Q3.1: Load Word2Vec embeddings from Google News corpus
    print("Loading Word2Vec embeddings (Google News corpus)...")
    print("This may take a while on first run...\n")

    try:
        # Load pre-trained Word2Vec model
        word2vec_model = api.load("word2vec-google-news-300")
        print("Word2Vec model loaded successfully!\n")

        # Create embedding matrix
        embedding_matrix = create_embedding_matrix(word_to_idx, word2vec_model)
    except Exception as e:
        print(f"Warning: Could not load Word2Vec: {e}")
        print("Using random embeddings instead.\n")
        embedding_matrix = None

    # Q3.2: Build LSTM model
    print("Building LSTM model...")
    ner_model = NerLstmModel(
        vocab_size=len(word_to_idx),
        embedding_dim=300,
        max_length=max_length,
        num_tags=9
    )
    ner_model.build_model(embedding_matrix)
    ner_model.summary()

    # Q3.3: Train model with cross entropy loss, Adam optimizer, 10 epochs
    print("\n" + "=" * 80)
    print("TRAINING MODEL (10 epochs)")
    print("=" * 80 + "\n")

    history = ner_model.model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Q3.4: Evaluate model on test set
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80 + "\n")

    print("Evaluating model on test set...")
    y_pred = ner_model.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_test_classes = np.argmax(y_test, axis=-1)

    # Flatten predictions and true labels
    y_pred_flat = []
    y_test_flat = []

    for i in range(len(y_test_classes)):
        for j in range(len(y_test_classes[i])):
            y_pred_flat.append(y_pred_classes[i][j])
            y_test_flat.append(y_test_classes[i][j])

    # Q3.4: Calculate metrics
    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_flat, y_pred_flat, average='macro', zero_division=0
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Accuracy:               {accuracy:.6f}")
    print(f"  Macro-average Precision: {precision:.6f}")
    print(f"  Macro-average Recall:    {recall:.6f}")
    print(f"  Macro-average F1 Score:  {f1:.6f}")
    print("=" * 80 + "\n")

    return ner_model, history, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all assignment questions
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "CS 421: NLP - ASSIGNMENT 4" + " " * 32 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    results = {}

    # Question 1: TF-IDF and Cosine Similarity (25 points)
    print("\n[Running Question 1: TF-IDF and Cosine Similarity]")
    try:
        vectorizer, tfidf_matrix, df = question_one()
        results['q1'] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'dataframe': df
        }
    except Exception as e:
        print(f"\nError in Q1: {e}")
        import traceback
        traceback.print_exc()

    # Question 2: PPMI (5 points)
    print("\n[Running Question 2: PPMI Calculation]")
    try:
        ppmi1, ppmi2 = question_two()
        results['q2'] = {'ppmi_example1': ppmi1, 'ppmi_example2': ppmi2}
    except Exception as e:
        print(f"\nError in Q2: {e}")
        import traceback
        traceback.print_exc()

    # Question 3: NER with LSTM (20 points)
    print("\n[Running Question 3: Named Entity Recognition using LSTM]")
    try:
        model, history, metrics = question_three()
        results['q3'] = {
            'model': model,
            'history': history,
            'metrics': metrics
        }
    except Exception as e:
        print(f"\nError in Q3: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 80)
    print("*" + " " * 25 + "ASSIGNMENT COMPLETE!" + " " * 33 + "*")
    print("*" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = main()
