# CS 421 NLP Assignment 5: Constituency and Dependency Parsing

## Files Overview

| File | Description |
|------|-------------|
| `assignment5.py` | Main assignment file with all tasks |
| `dep_parser.py` | Standalone dependency parser module (Q3a) |
| `start_corenlp.bat` | Batch script to start CoreNLP server (Windows) |
| `stanford-corenlp-4.5.10/` | Stanford CoreNLP installation |

## Requirements

```bash
pip install nltk numpy
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('large_grammars')
```

## Running the Assignment

### Task 1 & 2 (Constituency Tree & CKY Parsing)
These can be run without CoreNLP:
```bash
python assignment5.py
```

### Task 3 (Dependency Parsing)
Requires CoreNLP server running:

1. **Start CoreNLP Server:**
   - Double-click `start_corenlp.bat` (Windows)
   - OR run manually:
     ```bash
     cd stanford-corenlp-4.5.10
     java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
     ```

2. **Wait for:** `StanfordCoreNLPServer listening at...` message

3. **Run the parser (in a new terminal):**
   ```bash
   python dep_parser.py
   ```

## Task Descriptions

### Q1: Constituency Tree
Creates a visual constituency tree for "Cat sat on the mat" using production rules:
- S → VP
- VP → NP V PP
- NP → DET ADJ N | DET N | N
- PP → P NP

### Q2: CKY Parsing
Implements CKY algorithm with:
- ATIS CFG grammar from NLTK
- Chomsky Normal Form conversion
- Back-pointer tracking for tree construction

Test sentences:
1. "What is the cheapest one way flight from columbus to indianapolis"
2. "Is there a flight from memphis to los angeles"
3. "What aircraft is this"
4. "Show american flights after twelve p.m. from miami to chicago"

### Q3a: Dependency Parsing
`get_dependency_parse(sentence)` function that:
- Takes a sentence string
- Returns CoNLL-formatted output (word, POS, head, relation)
- Uses Stanford CoreNLP via NLTK

### Q3b: Ambiguous Sentence Analysis
Analyzes:
1. "Flying planes can be dangerous"
2. "Amid the chaos I saw her duck"
3. Additional: "I made her duck orange sauce"
