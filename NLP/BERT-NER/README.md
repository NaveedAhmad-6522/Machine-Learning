# BERT-Based Named Entity Recognition (NER)

A Named Entity Recognition system built using **BERT Base Cased** and **TensorFlow/Keras**, fine-tuned on the CoNLL-2003 dataset.

## Project Overview

This project identifies and classifies named entities in text into four categories:

- **PER:** Person
- **ORG:** Organization
- **LOC:** Location
- **MISC:** Miscellaneous

It uses the BIO tagging scheme:
- B: Beginning of an entity
- I: Inside or continuation of an entity
- O: Outside an entity

## Model Architecture

- Pretrained Model: `bert-base-cased`
- Framework: TensorFlow / Keras
- Encoder: BERT
- Dropout: 0.3
- Classification Layer: Dense (9 classes)
- Optimizer: Adam
- Learning Rate: 2e-5
- Training Epochs: 3
- Maximum Sequence Length: 128
- Batch Size: 16

The model uses WordPiece tokenization and aligns NER labels with the first subword of each original word. Special tokens, padding, and continuation subwords are excluded from the loss using an ignore index of `-100`.

## Dataset

**CoNLL-2003**, loaded from Hugging Face:

`lhoestq/conll2003`

The dataset contains annotated sentences with four named entity categories.

## Test Results

Evaluation was performed using the `seqeval` library.

| Metric | Score |
|---|---:|
| Precision | 88.99% |
| Recall | 91.07% |
| F1-score | 90.02% |

### Entity-Level Performance

| Entity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| LOC | 94% | 93% | 93% |
| MISC | 77% | 80% | 79% |
| ORG | 83% | 91% | 87% |
| PER | 97% | 94% | 96% |

## Repository Structure

```text
BERT-NER/
├── model/
│   └── ner_model.weights.h5
├── tokenizer/
│   └── Tokenizer files
├── config/
│   └── label_mapping.json
├── notebooks/
│   └── BERT_NER.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Loading the Model

The repository contains trained model weights rather than a complete serialized model.

To load the checkpoint:

1. Recreate the original `BERT_NER` model architecture.
2. Load the pretrained BERT encoder.
3. Initialize the dropout and classification layers.
4. Build the model using a sample input.
5. Load the saved weights using `load_weights()`.

The complete implementation is available in the notebook.

## Technologies Used

- Python
- TensorFlow
- Keras
- Hugging Face Transformers
- Hugging Face Datasets
- BERT
- CoNLL-2003
- Seqeval

## Future Improvements

- Improve performance on MISC entities.
- Experiment with different training configurations.
- Improve word-level inference and entity extraction.
- Explore other Transformer architectures.

## Author

Naveed Ahmad

## License

This project is available for educational and research purposes.