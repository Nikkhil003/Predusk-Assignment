# Iris Classification - Neural Network from Scratch in PyTorch

This project demonstrates how to build and train a **two-layer neural network from scratch using PyTorch for classifying iris flowers based on sepal and petal measurements.

---

## 📊 Dataset

We use the **Iris dataset**:
- 150 samples
- 4 features per sample:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- 3 classes: Setosa, Versicolor, Virginica

---

## 🧠 Model Overview

- **Architecture**:
  - Input: 4 units
  - Hidden layer: 32 units, ReLU
  - Output: 3 units (logits for class probabilities)

- **Manual Components**:
  - Weight initialization: Xavier
  - Training loop: forward → loss → backward → SGD
  - Accuracy tracking across epochs

- **Loss**: CrossEntropyLoss  
- **Optimizer**: Manual SGD (no high-level optimizer)  
- **Training Duration**: 50 epochs

---

## 📈 Output

- Final training and test accuracy (expected: ~81% / ~76%)
- Accuracy vs Epoch plot
- Detailed data preprocessing and normalization

---

## 🛠️ Setup & Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/iris-classifier.git
cd iris-classifier
```
### 2. Install Dependencies
``` bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 3.How to Run
```bash
python iris_classifier.py
```
If you're using a Jupyter notebook:
```bash
jupyter notebook iris_classifier.ipynb
```

