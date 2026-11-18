# SAMSum GPT Summarization — TensorFlow

A decoder-only GPT language model trained from randomly initialized weights for dialogue summarization on the SAMSum dataset.

## Project design

This is **not** an encoder–decoder model and does not fine-tune pretrained GPT weights. It uses a custom causal Transformer decoder implemented in TensorFlow/Keras. A byte-level BPE tokenizer is trained on the training split only.

## Four-notebook workflow

1. `01_gpt_decoder_architecture.ipynb` — defines and sanity-checks the GPT decoder.
2. `02_preprocess_and_train_bpe.ipynb` — loads SAMSum, trains byte-level BPE, and writes tokenized splits.
3. `03_train_gpt_from_scratch.ipynb` — loads the architecture module and processed examples, trains from scratch, and saves the best checkpoint.
4. `04_inference_and_evaluation.ipynb` — restores the best checkpoint, generates summaries, and reports ROUGE.

The reusable model implementation lives in `src/gpt_model.py`; Notebook 01 documents and tests the same implementation. Notebooks 02–04 communicate through files on disk, not shared notebook variables.

## Dataset

Place the SAMSum JSON files here:

```text
data/raw/train.json
data/raw/val.json
data/raw/test.json
```

Each record should contain `dialogue` and `summary` fields. Dataset files are intentionally excluded from version control.

## Install

```bash
pip install -r requirements.txt
```

Run notebooks in order from the repository root. On Kaggle, attach the project files and dataset, then update paths in the configuration or copy the project to `/kaggle/working/` before running. Kaggle input mounts are read-only; write tokenizers, processed data, and checkpoints under `/kaggle/working/`.

## Important training notes

- The model is randomly initialized; no pretrained language-model weights are used.
- BPE is trained only on the training dialogues and summaries to avoid validation/test leakage.
- The training sequence is `dialogue prompt + summary + EOS`. Loss is masked on prompt tokens and padding; only summary tokens contribute.
- The decoder uses causal attention and masks padding tokens as attention keys.
- The default configuration is a substantial starting point, not a performance guarantee. SAMSum is a relatively small dataset for training a GPT from scratch. Monitor validation loss and ROUGE, inspect generations, and adjust model size/training accordingly.
- A 6-layer, 512-dimensional model may exceed the memory available on some Kaggle accelerators. Reduce batch size, sequence length, or model dimensions if needed. Gradient accumulation is included in the training notebook.

## Output locations

- Tokenizer: `artifacts/tokenizer/tokenizer.json`
- Processed examples: `data/processed/*.jsonl`
- Best checkpoint: `artifacts/checkpoints/best.weights.h5`
- Training history: `results/training_history.json`
- Test predictions: `results/test_predictions.json`

## License and dataset terms

Check the SAMSum dataset's original license and terms before redistributing the dataset or derived artifacts. This repository does not bundle the dataset.
