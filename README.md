# Sentiment Analysis on Product Reviews using Hugging Face Transformers

This project fine-tunes **DistilBERT** to perform binary sentiment classification (positive/negative) on the [IMDb movie reviews dataset](https://huggingface.co/datasets/imdb) using the Hugging Face `transformers` and `datasets` libraries.

The goal is to demonstrate **NLP skills** for a practical, real-world task and to provide a clean, reproducible codebase for portfolio purposes.

---

## 📌 Features
- Fine-tunes a pre-trained transformer model on real-world text data.
- Uses Hugging Face's `Trainer` API for training and evaluation.
- Includes accuracy and F1-score metrics.
- Inference pipeline for testing custom reviews.
- Fully reproducible in Google Colab.

---

## 📂 Example Inference

```python
from transformers import pipeline

pipeline = pipeline("text-classification", model="mhemon/seintiment-distilbert-base-uncased", device="cuda")
```

```python
pipeline("The product quality is amazing and delivery was fast!")
# [{'label': 'POSITIVE', 'score': 0.7757363319396973}]
```

## 📊 Result
| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.85  |
| F1 Score | 0.85  |

## Colab File
https://colab.research.google.com/drive/1hycSErDqoGt1mWz20Al3fRX6EbDbW2nF?usp=sharing

