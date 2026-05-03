# From Bag-of-Words to BERT: Comparing Text Representations for Hate Speech Detection

An empirical comparison of three text representation strategies for hate speech detection on social media: TF-IDF, Word2Vec, and fine-tuned BERT.

## Overview

Hate speech detection is a critical NLP task, but how much does representation sophistication actually matter? This project benchmarks three increasingly complex approaches on the same dataset to quantify the practical gains of modern representations over classical ones.

Using the Davidson et al. (2017) dataset of ~25,000 tweets labeled as hate speech, offensive language, or neither, we find that fine-tuned BERT achieves the best macro F1 (0.76), but the gains are concentrated on the minority hate speech class. Surprisingly, Word2Vec underperforms TF-IDF — averaging word embeddings dilutes the lexical signals that matter most for this task.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|----------------|--------------|----------|
| TF-IDF + Logistic Regression | 0.9011 | 0.8096 | 0.6541 | 0.6825 |
| Word2Vec + Logistic Regression | 0.8786 | 0.7487 | 0.6075 | 0.6300 |
| BERT (fine-tuned) | **0.9118** | **0.7798** | **0.7442** | **0.7606** |

### Per-Class F1 Scores

| Class | TF-IDF + LR | Word2Vec + LR | BERT |
|-------|-------------|---------------|------|
| Hate Speech | 0.26 | 0.19 | **0.47** |
| Offensive Language | 0.94 | 0.93 | **0.95** |
| Neither | 0.85 | 0.77 | **0.87** |

### Key Findings

- **BERT nearly doubles** hate speech F1 over TF-IDF (0.47 vs. 0.26), with gains driven by contextual understanding of subtle distinctions between hate and offense
- **Word2Vec underperforms TF-IDF** — document-level averaging destroys the fine-grained lexical cues (e.g., specific slur variants) that distinguish hate speech from offensive language
- **High accuracy is misleading** — all models hit ~90% accuracy by correctly classifying the dominant offensive class, but macro F1 reveals the real gaps
- **Shared failure modes** across all models: implicit hate, sarcasm, in-group reclaimed language, and polysemous slang

## Dataset

[Davidson et al. (2017)](https://github.com/t-davidson/hate-speech-and-offensive-language) hate speech and offensive language dataset:
- **24,802 tweets** labeled via crowdsourcing (92% intercoder agreement)
- **3 classes:** Hate Speech (5.8%), Offensive Language (77.4%), Neither (16.8%)
- Collected using the Hatebase lexicon via the Twitter API

## Project Structure

```
.
├── NLP_Project.ipynb          # Full pipeline: EDA, training, evaluation, error analysis
├── report.docx                # Write-up (conference-style, 5 pages)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Running

The notebook runs on **Google Colab** with GPU:

1. Upload `NLP_Project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells

The pipeline will automatically download the dataset, load Word2Vec embeddings (~1.6 GB, cached to Drive), fine-tune BERT (~10 min on T4), and produce all evaluation outputs.

## Methods

### TF-IDF + Logistic Regression
Unigrams + bigrams, 10,000 max features, English stop words removed. Standard scikit-learn pipeline.

### Word2Vec + Logistic Regression
Pre-trained Google News vectors (300d). Documents represented as mean of constituent word vectors. Logistic Regression on the resulting 300-dim representations.

### Fine-tuned BERT
`bert-base-uncased` via Hugging Face Transformers. 3 epochs, lr=2e-5, batch size 16, weight decay 0.01. Classification head on [CLS] token.

## References

1. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *ICWSM 2017*.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
3. Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS 2013*.
4. Nirmal, A. et al. (2024). Towards Interpretable Hate Speech Detection using LLM-extracted Rationales. *WOAH Workshop, ACL 2024*.
5. Plaza-del-Arco, F. M. et al. (2023). Respectful or Toxic? Using Zero-Shot Learning with Language Models to Detect Hate Speech. *WOAH Workshop, ACL 2023*.
6. Roy, S. et al. (2023). Probing LLMs for Hate Speech Detection: Strengths and Vulnerabilities. *Findings of EMNLP 2023*.
