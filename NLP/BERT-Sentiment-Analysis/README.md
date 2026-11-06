# BERT-Based Sentiment Analysis

A binary sentiment classification project using **BERT Base Uncased** and TensorFlow/Keras to classify IMDb movie reviews as positive or negative.

## Overview

This project fine-tunes a pretrained BERT model for binary sentiment classification using the IMDb movie review dataset.

## Model Architecture

- Pretrained model: BERT Base Uncased
- Transformer encoder layers: 12
- Hidden size: 768
- Attention heads: 12
- Maximum sequence length: 128
- Dropout: 0.3
- Classification head: Dense layer with 2 output neurons and Softmax activation

## Dataset

Balanced IMDb dataset:

| Split      | Negative | Positive | Total |
| ---------- | -------: | -------: | ----: |
| Training   |    4,500 |    4,500 | 9,000 |
| Validation |      500 |      500 | 1,000 |
| Testing    |    1,250 |    1,250 | 2,500 |

## Training Configuration

- Optimizer: Adam
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Loss: Sparse Categorical Crossentropy
- Early stopping based on validation loss

## Results

| Metric/usr/local/lib/python3.13/dist-packages/transformers/generation/tf_utils.py:465: UserWarning: \`seed_generator\` is deprecated and will be removed in a future version.&#xA;  warnings.warn("\`seed_generator\` is deprecated and will be removed in a future version.", UserWarning)&#xA; |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

```
```

|                          Score |        |
| -----------------------------: | ------ |
| Training accuracy (best epoch) | 90.60% |
|            Validation accuracy | 87.40% |
|                  Test accuracy | 86.04% |
|                      Test loss | 0.3068 |
|                 Macro F1-score | 0.86   |

## Technologies

- Python
- TensorFlow / Keras
- Hugging Face Transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Future Improvements

- Error analysis and model calibration
- Handling neutral sentiment
- Sarcasm detection
- Experimenting with larger training datasets
- Model deployment through an API

## Acknowledgements

- Hugging Face Transformers
- IMDb Dataset
- Google BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
