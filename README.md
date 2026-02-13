# Prompt Injection Classifier

Fine-tuned **mDeBERTa-v3-base** for binary classification of prompt injection attacks targeting an Italian ERP chatbot.

## Problem

LLM-powered chatbots are vulnerable to prompt injection — malicious inputs designed to override system instructions, extract confidential data, or hijack the model's behavior. This classifier acts as a security layer that filters user inputs before they reach the main LLM.

## Approach

| | |
|---|---|
| **Base model** | [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) (multilingual DeBERTa with disentangled attention) |
| **Task** | Binary classification: `legitimate` (0) / `injection` (1) |
| **Dataset** | 13,017 Italian samples (domain-specific ERP chatbot queries) |
| **Split** | 80/10/10 stratified (train: 10,413 / val: 1,302 / test: 1,302) |
| **Loss** | Weighted cross-entropy (class weights auto-computed via `compute_class_weight`) |
| **Optimization** | AdamW, lr=2e-5, warmup 10%, weight decay 0.01, fp16, gradient accumulation (2 steps) |
| **Selection metric** | Recall — in a security system, minimizing false negatives (undetected injections) is the priority |
| **Early stopping** | Patience 3 on validation recall |
| **Export** | ONNX via Optimum for production inference |

### Why mDeBERTa?

mDeBERTa uses disentangled attention (separate content and position embeddings) which improves understanding of token relationships — useful for detecting subtle injection patterns. The multilingual variant handles Italian text natively without translation overhead.

## Results (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.85% |
| **Precision** | 97.93% |
| **Recall** | **99.85%** |
| **F1 Score** | 98.88% |

```
              precision    recall  f1-score   support

  legitimate       1.00      0.98      0.99       638
   injection       0.98      1.00      0.99       664

    accuracy                           0.99      1302
```

The model converged in ~1 epoch (600 steps) with early stopping, indicating the task is well-suited for a pretrained multilingual transformer of this size.

## Observations & Limitations

- The dataset is nearly balanced (ratio 0.96), so the weighted loss has minimal practical impact. It was included as a best practice for production pipelines where class distribution may shift over time.
- High metrics are partly explained by the domain-specific nature of the dataset (Italian chatbot queries for Odoo ERP). Generalization to other domains or languages would require additional fine-tuning and evaluation.
- `max_length=512` is conservative — the 99th percentile token length is well below this threshold (max observed: 270). A lower value (e.g. 128) would improve training and inference speed with no accuracy loss.
- Training was performed on Google Colab with a single GPU using mixed precision (fp16).

## Project Structure

```
├── fine_tune.ipynb      # Full pipeline: EDA, training, evaluation, export
├── README.md
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
```

Open `fine_tune.ipynb` in Google Colab (GPU recommended) or Jupyter, and update the dataset path:

```python
df = pd.read_csv('path/to/your/dataset.csv')  # columns: domanda, label
```

## Stack

Python, PyTorch, Hugging Face Transformers, Optimum, ONNX Runtime, scikit-learn, pandas, matplotlib, seaborn
