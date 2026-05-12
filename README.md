
# Part 3: NLP and Sequence Modeling Mini Project

** Module 5 Assignment**

---

## Problem Statement
Build a complete NLP pipeline on a customer support text dataset. Compare traditional text vectorization with a sequence-based deep learning model (LSTM) for sentiment classification.

## Dataset
- **Source:** Synthetic Customer Support Text Classification Dataset ( https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJIV-wBvUYs?usp=sharing )
- **Total Records:** 1,500 customer support messages
- **Columns:** ticket_id, channel, customer_message, sentiment_label, word_count, urgent_flag
- **Target:** `sentiment_label` → positive / negative / neutral
- **Class Distribution:** Neutral: 524 | Negative: 497 | Positive: 479 (nearly balanced)

---

## Task 1 — Dataset Understanding

| Metric | Value |
|--------|-------|
| Total Records | 1,500 |
| Target Classes | 3 (positive, negative, neutral) |
| Avg Word Count | 12.72 words/message |
| Avg Char Length | 72.76 chars |
| Missing Values | None |
| Class Balance | Nearly balanced (~33% each) |

**Sample Records:**
- [NEGATIVE] "My order has been delayed and no one is responding to my emails."
- [POSITIVE] "The app experience is smooth and reliable. I appreciate the quick response."
- [NEUTRAL] "I am checking if there is any scheduled maintenance today."

---

## Task 2 — Text Preprocessing

Steps applied to every customer message:
1. **Lowercasing** — "Hello" and "hello" treated as the same word
2. **Remove special characters** — punctuation, numbers, symbols removed
3. **Tokenization** — sentence split into individual word tokens
4. **Remove stopwords** — "the", "is", "a", "please" etc. removed (152 stopwords)

**Result:** Average words reduced from 12.7 → 6.4 per message (noise removed)

---

## Task 3 — Text Vectorization

**Why convert text to numbers?** Machine learning models cannot process raw text — they only work with numbers. Vectorization creates a numerical representation of each word/sentence.

Three methods used:

| Method | Description | Shape |
|--------|-------------|-------|
| Bag of Words (BoW) | Counts word occurrences | (1200, 148) |
| TF-IDF | Weights words by importance | (1200, 426) |
| Tokenizer + Padding | Integer sequences for LSTM | (1200, 30) |

---

## Task 4 — Baseline Models

| Model | Vectorization | Test Accuracy |
|-------|--------------|---------------|
| Naive Bayes | Bag of Words | **100.00%** |
| Logistic Regression | TF-IDF (unigrams+bigrams) | **100.00%** |

Both baselines achieved 100% — confirming the dataset has strong, clean textual patterns per class.

---

## Task 5 — LSTM Sequence Model

### Architecture
```
Input (sequence of 30 word IDs)
    ↓ Embedding(5000 → 64 dims)    → Each word becomes a 64-dim vector
    ↓ LSTM(64 units)               → Reads sequence, captures context
    ↓ Dropout(0.4)                 → Prevents overfitting
    ↓ Dense(32, ReLU)              → Learns complex patterns
    ↓ Dense(3, Softmax)            → 3-class probability output
```

### Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **100.00%** |
| Test Loss | 0.0059 |
| Epochs Run | 8 (early stopping) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

---

## Task 6 — Attention and Transformer Reflection

### Why RNNs Struggle with Long-Term Dependencies
Basic RNNs pass a single hidden state word-by-word. Over long sequences, early information gets repeatedly overwritten — by word 50, the model has mostly forgotten word 1. This is the **vanishing gradient problem**: gradients shrink to near-zero during backpropagation across many steps, making it impossible to learn long-range relationships.

### How LSTMs Help with Memory
LSTMs introduce three **gates** — Forget, Input, and Output — plus a dedicated memory cell. The forget gate decides what to erase, the input gate decides what to write, and the output gate decides what to read. This allows LSTMs to selectively remember information across hundreds of words, solving the vanishing gradient problem that cripples plain RNNs.

### What Attention Solves in Sequence-to-Sequence Tasks
In tasks like translation, RNNs/LSTMs compress the entire input sentence into one fixed-size vector — a bottleneck that loses information for long inputs. **Attention** lets the model look at ALL input words at every output step, shining a spotlight on the most relevant parts. When translating "bank", attention checks whether nearby words are "river" or "money" to choose the right meaning.

### Why Transformers are Important for Modern NLP and Generative AI
Transformers (2017) replaced RNNs entirely as the dominant NLP architecture:
- **Self-attention** connects every word to every other word in one step — no sequential bottleneck
- **Fully parallelizable** — processes all words simultaneously, enabling massive scale
- **Powers everything modern:** GPT-4, ChatGPT (text generation), BERT (search), Whisper (speech), DALL-E (images), AlphaFold (protein folding)

The transformer's ability to scale to billions of parameters while efficiently capturing long-range dependencies is what enabled the generative AI revolution.

---

## Repository Structure
```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

## Dataset File Needed
- `customer_support_text_classification.csv` — place in same folder as notebook

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
# Run Kernel → Restart & Run All
```
