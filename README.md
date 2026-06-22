# LoRA & QLoRA Implementation on CIFAR-10
### Parameter-Efficient Fine-Tuning · Custom Keras Layers · PEFT · Python

---

> A hands-on implementation of **Low-Rank Adaptation (LoRA)** and **Quantized Low-Rank Adaptation (QLoRA)** for parameter-efficient fine-tuning, applied to the CIFAR-10 image classification dataset using custom Keras layers and the Hugging Face PEFT library.

---

## WHAT IS LORA / QLORA?

Full fine-tuning of deep neural networks updates every weight — expensive in both compute and memory. LoRA and QLoRA are **parameter-efficient fine-tuning (PEFT)** techniques that freeze the original model weights and inject small trainable low-rank matrices, drastically reducing trainable parameters while maintaining performance.

| Method | Core Idea | Memory Saving |
|--------|-----------|--------------|
| **LoRA** | Inject trainable low-rank matrices (A, B) into frozen layers | High |
| **QLoRA** | LoRA + 4-bit NormalFloat quantization of base model weights | Very High |

---

## HOW LORA WORKS

```
Original weight update:   W = W0 + dW        (full fine-tuning)

LoRA approximation:       W = W0 + B * A     (low-rank decomposition)

Where:
  W0  = frozen pre-trained weight matrix  (d x d)
  A   = trainable matrix                  (d x r)
  B   = trainable matrix                  (r x d)
  r   = rank (hyperparameter, r << d)

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
Double Quantization (optional — quantize the quantization constants)
      ↓
Train only A, B — dequantize W0 at compute time
```

This enables fine-tuning large models on consumer GPUs that would otherwise run out of memory.

---

## DATASET

**CIFAR-10** — 60,000 32x32 colour images across 10 classes:

| Class | Class | Class | Class | Class |
|-------|-------|-------|-------|-------|
| Airplane | Automobile | Bird | Cat | Deer |
| Dog | Frog | Horse | Ship | Truck |

- 50,000 training images
- 10,000 test images

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
└── lora_qlora.ipynb     # Full implementation notebook (LoRA + QLoRA)
```

---

## IMPLEMENTATION HIGHLIGHTS

### Custom LoRA Layer (Keras)
```python
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, units, rank=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank   # r << d — keeps trainable params small

    def build(self, input_shape):
        d = input_shape[-1]
        # A and B are the only trainable matrices
        self.A = self.add_weight(shape=(d, self.rank), trainable=True, name="lora_A")
        self.B = self.add_weight(shape=(self.rank, self.units), trainable=True, name="lora_B")
        self.W0 = self.add_weight(shape=(d, self.units), trainable=False, name="frozen_W")

    def call(self, x):
        # W_effective = W0 + B @ A  (low-rank update injected at inference)
        return tf.matmul(x, self.W0 + tf.matmul(self.A, self.B))
```

### Key Hyperparameters
| Parameter | LoRA | QLoRA |
|-----------|------|-------|
| Rank (r) | 4–16 | 4–16 |
| Alpha | 16–32 | 16–32 |
| Dropout | 0.05 | 0.05 |
| Quantization | None | 4-bit NF4 |
| Target Modules | Dense layers | Dense layers |

---

## GETTING STARTED

### Option 1 — Run on Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/porasnehra/2191c8b1fb6692a7a8b8a231e88576d6/lora-qlora.ipynb)

### Option 2 — Run Locally
```bash
# Clone
git clone https://github.com/porasnehra/lora-and-Qlora-implimentation-.git
cd lora-and-Qlora-implimentation-

# Install dependencies
pip install tensorflow peft numpy matplotlib

# Launch notebook
jupyter notebook lora_qlora.ipynb
```

---

## KEY CONCEPTS COVERED

- Why full fine-tuning is memory-inefficient for large models
- Low-rank matrix decomposition (W = W0 + B*A)
- Building a custom LoRA layer from scratch in Keras
- Applying PEFT library for QLoRA quantization
- Trainable parameter count comparison: full fine-tuning vs LoRA vs QLoRA
- Impact of rank (r) on accuracy vs parameter efficiency trade-off

---

## AUTHOR

**Poras Nehra** · B.Tech CS @ MMEC, Mullana
[GitHub](https://github.com/porasnehra) · [LinkedIn](https://linkedin.com/in/poras-nehra-142170367)

---

> This project is part of a structured GenAI/ML learning roadmap covering deep learning fundamentals, fine-tuning techniques, and agentic AI systems.

---

⭐ Star this repo if it helped you understand LoRA and QLoRA!
