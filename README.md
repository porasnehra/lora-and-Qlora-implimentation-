# LoRA & QLoRA Implementation on IMDB Movie Reviews
### Parameter-Efficient Fine-Tuning · NLP · Custom Keras Layers · PEFT · Python

---

> A hands-on implementation of **Low-Rank Adaptation (LoRA)** and **Quantized Low-Rank Adaptation (QLoRA)** applied to the **IMDB 50K Movie Reviews** dataset — a sentiment classification task using custom Keras layers and the Hugging Face PEFT library.

---

## WHAT IS LORA / QLORA?

Full fine-tuning updates every weight in the model — expensive in both compute and memory. LoRA and QLoRA are **parameter-efficient fine-tuning (PEFT)** techniques that freeze the original model weights and inject small trainable low-rank matrices, drastically reducing trainable parameters while maintaining performance.

| Method | Core Idea | Memory Saving |
|--------|-----------|---------------|
| **LoRA** | Inject trainable low-rank matrices (A, B) into frozen layers | High |
| **QLoRA** | LoRA + 4-bit NormalFloat quantization of base model weights | Very High |

---

## HOW LORA WORKS

```
Original weight update:   W = W0 + dW        (full fine-tuning — updates everything)

LoRA approximation:       W = W0 + B * A     (low-rank decomposition)

Where:
  W0  = frozen pre-trained weight matrix  (d x d)
  A   = trainable matrix                  (d x r)
  B   = trainable matrix                  (r x d)
  r   = rank hyperparameter               (r << d, e.g. 4, 8, 16)

Only A and B are trained — W0 is never updated.
```

---

## HOW QLORA EXTENDS LORA

```
Base Model Weights
      ↓
4-bit NormalFloat Quantization (NF4)
      ↓
Frozen Quantized Base + LoRA Adapters (A, B in fp16)
      ↓
Train only A and B — base weights dequantized at compute time
      ↓
Sentiment Prediction: Positive / Negative
```

---

## DATASET

**IMDB Dataset of 50,000 Movie Reviews**

| Split | Size | Classes |
|-------|------|---------|
| Train | 25,000 reviews | Positive / Negative |
| Test | 25,000 reviews | Positive / Negative |

- Binary sentiment classification (Positive / Negative)
- Reviews vary widely in length — requires tokenization and padding
- Balanced dataset: 50% positive, 50% negative

Example:
```
Review: "This film was absolutely brilliant. The acting was superb and the plot kept me engaged throughout."
Label:  Positive

Review: "A complete waste of time. Poorly written script with no character development whatsoever."
Label:  Negative
```

---

## STACK

```
Python · TensorFlow · Keras · PEFT (Hugging Face)
NumPy · Matplotlib · Google Colab (GPU runtime)
```

---

## PROJECT STRUCTURE

```
lora-and-Qlora-implimentation-/
└── lora_qlora.ipynb     # Full implementation: data loading, LoRA, QLoRA, evaluation
```

---

## IMPLEMENTATION HIGHLIGHTS

### Custom LoRA Layer (Keras)
```python
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, units, rank=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank  # r << d — keeps trainable params minimal

    def build(self, input_shape):
        d = input_shape[-1]
        # Only A and B are trainable — W0 is frozen
        self.A = self.add_weight(shape=(d, self.rank), trainable=True, name="lora_A")
        self.B = self.add_weight(shape=(self.rank, self.units), trainable=True, name="lora_B")
        self.W0 = self.add_weight(shape=(d, self.units), trainable=False, name="frozen_W")

    def call(self, x):
        # Effective weight = frozen base + low-rank update
        return tf.matmul(x, self.W0 + tf.matmul(self.A, self.B))
```

### Key Hyperparameters

| Parameter | LoRA | QLoRA |
|-----------|------|-------|
| Rank (r) | 4–16 | 4–16 |
| Alpha | 16–32 | 16–32 |
| Dropout | 0.05 | 0.05 |
| Quantization | None | 4-bit NF4 |
| Task | Sentiment Classification | Sentiment Classification |

---

## GETTING STARTED

### Option 1 — Run on Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/porasnehra/2191c8b1fb6692a7a8b8a231e88576d6/lora-qlora.ipynb)

> Enable GPU runtime: Runtime → Change runtime type → T4 GPU

### Option 2 — Run Locally

```bash
# Clone
git clone https://github.com/porasnehra/lora-and-Qlora-implimentation-.git
cd lora-and-Qlora-implimentation-

# Install dependencies
pip install tensorflow peft numpy matplotlib datasets

# Launch notebook
jupyter notebook lora_qlora.ipynb
```

---

## KEY CONCEPTS COVERED

- Why full fine-tuning is memory-inefficient for large models
- Low-rank matrix decomposition: W = W0 + B * A
- Building a custom LoRA layer from scratch in Keras
- Text preprocessing: tokenization, padding, vocabulary building on IMDB
- Applying Hugging Face PEFT for QLoRA quantization
- Trainable parameter comparison: full fine-tuning vs LoRA vs QLoRA
- Impact of rank (r) on accuracy vs parameter efficiency trade-off

---

## WHY IMDB + LORA?

LoRA was originally proposed for fine-tuning large language models — so applying it to a text sentiment task is far more representative of real-world use cases than image benchmarks. The IMDB dataset provides a clean, well-understood environment to observe how low-rank adapters capture task-specific knowledge without touching the base weights.

---

## AUTHOR

**Poras Nehra** · B.Tech CS @ MMEC, Mullana
[GitHub](https://github.com/porasnehra) · [LinkedIn](https://linkedin.com/in/poras-nehra-142170367)

---

> This project is part of a structured GenAI/ML learning roadmap covering deep learning fundamentals, fine-tuning techniques, RAG pipelines, and agentic AI systems.

---

⭐ Star this repo if it helped you understand LoRA and QLoRA!
