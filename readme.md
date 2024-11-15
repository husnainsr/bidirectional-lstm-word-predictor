# Bidirectional LSTM Word Predictor 🤖 

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/docs/transformers/index) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![GPU](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

A neural network-based system that predicts missing words in sentences using both forward and backward LSTM models. 🔮

💾 Installation
---------------

### 📋 Prerequisites

-   **Python Version:** Ensure you have Python **3.11.7** installed. You can download it from the [official website](https://www.python.org/downloads/release/python-3117/).

### 📥 Clone this Repo

```bash
git clone https://github.com/husnainsr/bidirectional-lstm-word-predictor.git
```


Creating a virtual environment helps manage dependencies and avoid conflicts.

```bash
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

### 📦 Install Dependencies

Ensure you have `pip` installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```



## 📊 Data Preprocessing

### 📚 Dataset
- Uses the RACE dataset, which contains high-quality reading comprehension passages
- Sentences are extracted from articles and filtered to remove very short sequences
- Training set: 10,000 sentences
- Validation set: 1,000 sentences

### ⚙️ Input Processing
- Creates fill-in-the-blank examples by randomly selecting words from the latter half of sentences
- Tokenization using BERT tokenizer (bert-base-uncased)
- Sequences are padded/truncated to max_length=512
- For each sentence:
  - Forward model receives text before the blank
  - Backward model receives reversed text after the blank

## 🏗️ Model Architecture

### 🧠 LSTM Predictor
- Embedding layer (vocab_size × 128)
- LSTM layers:
  - Input size: 128
  - Hidden size: 256
  - Number of layers: 2
  - Dropout: 0.3
- Final linear layer maps to vocabulary size
- Attention mask handling for proper padding

### 🔧 Training Parameters
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Early stopping patience: 5 epochs
- Maximum epochs: 50

## 📈 Training Results

### ⏩ Forward Model
- Final training accuracy: 18.67%
- Best validation accuracy: 14.50% (Epoch 8)
- Early stopping triggered after 13 epochs
- Notable challenges with forward prediction accuracy

### ⏪ Backward Model
- Final training accuracy: 36.85%
- Best validation accuracy: 18.80% (Epoch 9)
- Early stopping triggered after 14 epochs
- Generally better performance than forward model

## 🔍 Observations

### 📊 Model Performance
1. Backward model consistently outperformed the forward model
2. Both models showed signs of overfitting:
   - Forward model: Training acc 18.67% vs Val acc ~14%
   - Backward model: Training acc 36.85% vs Val acc ~18%

### 🎯 Prediction Behavior
- Models tend to predict common words (e.g., "the", "a", "to")
- Confidence scores are relatively low (typically 0.15-0.20)
- Combined bidirectional approach helps improve prediction quality

### ⚠️ Challenges
1. Limited vocabulary coverage
2. Difficulty with context-specific predictions
3. Training instability and early convergence
4. High computational requirements for larger datasets

## 🚀 Future Improvements
1. Increase training data size
2. Experiment with different model architectures
3. Implement better tokenization strategies
4. Add beam search for prediction
5. Incorporate pre-trained embeddings

## 💻 Requirements
- PyTorch
- Transformers (Hugging Face)
- CUDA-capable GPU recommended

---
<p align="center">
  Made with ❤️ using PyTorch and Hugging Face
</p>

<p align="center">
    Husnain Sattar i211354
</p>
