
# SAMSum Dialogue Summarization using Transformer

An experimental Transformer-based model built from scratch using TensorFlow and Keras to generate concise summaries of conversations from the SAMSum dataset.

## Overview

Dialogue summarization aims to produce short, meaningful summaries from conversations while preserving their key information.

This project explores an Encoder–Decoder Transformer architecture trained on the SAMSum dataset. The model takes a dialogue as input and generates a corresponding summary.

The implementation includes text preprocessing, word-level tokenization, Transformer layers, model training, and summary generation.

## Dataset

The project uses the **SAMSum Dataset**, which contains messenger-style conversations paired with human-written summaries.

Each data sample contains:

- `id`: Unique conversation identifier.
- `dialogue`: The original conversation.
- `summary`: The reference summary written by a human.

The dataset is divided into training, validation, and testing sets.

> The dataset files are not included in this repository.

## Model Architecture

The model follows an Encoder–Decoder Transformer architecture.

**Main components:**

- Word-level tokenization
- Token embeddings
- Positional information
- Multi-head self-attention
- Encoder and decoder blocks
- Feed-forward neural networks
- Cross-attention between encoder and decoder
- Dense output layer with softmax activation

### Current Configuration

| Parameter | Value |
|---|---:|
| Encoder layers | 2 |
| Decoder layers | 2 |
| Model dimension | 128 |
| Attention heads | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Maximum input length | 150 |
| Maximum output length | 50 |
| Optimizer | Adam |
| Learning rate | 0.0002 |

These values describe the current experimental configuration.

## Repository Structure

```text
SAMSum-Transformer/
│
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt
│
├── notebook/
│   └── samsum_transformer.ipynb
│
├── model/
│   ├── weights/
│   │   └── best_model.weights.h5
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.json
│   │   └── word_index.json
│   │
│   └── config/
│       └── config.json
│
├── results/
│   ├── training_history.json
│   └── generated_summaries.json
│
└── docs/
    └── experiment_notes.md
```

## Installation
Create and activate a virtual environment (recommended):

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the notebook:

```bash
jupyter notebook notebook/samsum_transformer.ipynb
```

2. Prepare the SAMSum dataset.
3. Run the preprocessing and tokenization steps.
4. Build and train the Transformer model.
5. Evaluate the model and generate summaries for sample dialogues.

To use the saved model for inference, load the trained weights and the corresponding tokenizer. The model configuration and tokenization settings must match those used during training.

## Results and Limitations

The model demonstrates the implementation of a complete Transformer-based dialogue summarization pipeline.

However, the generated summaries are still experimental. Some outputs may:

- Miss important details from the original conversation.
- Produce inaccurate or hallucinated information.
- Repeat phrases or generate irrelevant content.
- Fail to preserve the original meaning completely.

These limitations indicate that further improvements are needed in model capacity, training, preprocessing, and decoding.

## Future Improvements

- Experiment with deeper Encoder–Decoder architectures.
- Increase model dimensions and attention heads.
- Improve tokenization and vocabulary handling.
- Experiment with decoding strategies.
- Evaluate generated summaries using ROUGE scores.
- Improve factual consistency and summary quality.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## License

This project is intended for educational and research purposes.
