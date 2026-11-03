# Deep Learning Lab 1: Regression & Multi-Class Classification with PyTorch

## 📌 Objective
The objective of this lab is to design, train, tune, and evaluate Deep Neural Network (DNN / MLP) architectures using **PyTorch** for both **Regression** and **Multi-Class Classification** tasks, integrating `GridSearchCV` hyperparameter optimization, SMOTE data augmentation, and regularization techniques (L1, L2, Dropout).

---

## Tools & Libraries
* **Deep Learning:** PyTorch (`torch.nn`, `Dataset`, `DataLoader`, Kaiming & Xavier initialization)
* **Hyperparameter Tuning & Evaluation:** Scikit-Learn (`GridSearchCV`, `StandardScaler`, `classification_report`) & `Skorch`
* **Data Augmentation:** `imbalanced-learn` (`SMOTE`)
* **EDA & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Environment:** Google Colab

---

## Part 1: Regression (NYSE Stock Prices Dataset)
* **Dataset:** [NYSE Dataset (Kaggle)](https://www.kaggle.com/datasets/dgawlik/nyse) (`851,264` rows, 7 features). Predicts `close` price from `open`, `low`, `high`, and `volume`.
* **Architecture:** 3-layer MLP (`4 -> 32 -> 16 -> 1`) with `ReLU` activations, Kaiming Uniform initialization on hidden layers, and Xavier Uniform initialization on the linear output layer.
* **Best Hyperparameters (`GridSearchCV`):**
  * Optimizer: `Adam` | Learning Rate: `0.001` | Epochs: `40` | Hidden Size: `32`
* **Results Comparison:**

| Model Variant | Regularization Applied | Final Test MSE | Final Test $R^2$ Score |
| :--- | :--- | :---: | :---: |
| **Baseline MLP** | None | **0.000509** | **0.9994 (99.94%)** |
| **Regularized MLP** | Dropout (`0.1`) + L2 (`1e-4`) + L1 (`1e-5`) | 0.008245 | 0.9906 (99.06%) |

* **Key Takeaway:** Because `open`, `low`, and `high` have a near-perfect linear correlation ($1.00$) with `close`[cite: 1], the baseline MLP converged within 2 epochs without overfitting[cite: 2]. Adding Dropout and L1/L2 penalties slightly constrained capacity on noise-free linear features.

---

## Part 2: Multi-Class Classification (Predictive Maintenance Dataset)
* **Dataset:** [Machine Predictive Maintenance Classification (Kaggle)](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (`10,000` rows, 6 target failure classes).
* **Preprocessing & SMOTE Balancing:**
  * Dropped identifiers (`UDI`, `Product ID`), one-hot encoded product `Type`, label-encoded `Failure Type`, and standardized sensor features.
  * Used **SMOTE** to balance the severe class imbalance (`9,652` No Failure vs. `18` Random Failures), expanding the dataset to `57,912` balanced samples (`9,652` per class).
* **Architecture:** 3-layer MLP (`7 -> 128 -> 64 -> 6`) optimized with `CrossEntropyLoss`.
* **Best Hyperparameters (`GridSearchCV`):**
  * Optimizer: `Adam` | Learning Rate: `0.01` | Epochs: `40` | Hidden Size: `128` (CV Accuracy: `98.83%`)
* **Performance Metrics:**

| Dataset Split / Variant | Accuracy | Macro Sensitivity (Recall) | Macro Precision | Macro F1-Score | Cross-Entropy Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Train Set)** | 99.37% | 99.37% | 99.37% | 99.37% | 0.0261 |
| **Baseline (Test Set)** | **99.37%** | **99.38%** | **99.37%** | **99.37%** | **0.0346** |
| **Regularized (Test Set)** | 98.58% | 98.58% | 98.59% | 98.58% | 0.0564 |

---

## Synthesis: Key Learnings
1. **End-to-End PyTorch Workflow:** Building custom `Dataset` and `DataLoader` pipelines alongside modular `nn.Module` classes with proper weight initialization (`kaiming_uniform_` and `xavier_uniform_`) is essential for fast convergence.
2. **Importance of Data Balancing:** Without SMOTE, a classifier on the Predictive Maintenance dataset could achieve 96.5% accuracy by simply predicting `"No Failure"` every time while failing to detect actual breakdowns. Balancing the classes yielded 99%+ F1-score across all 6 failure modes.
3. **Hyperparameter Tuning & Regularization:** Combining PyTorch with `GridSearchCV` demonstrated the superiority of adaptive learning rate optimizers (`Adam`) over vanilla `SGD` for tabular MLPs. Furthermore, testing Dropout and L1/L2 regularization highlighted that regularization is most beneficial when a network suffers from high variance (overfitting).


---
---

# Deep Learning Lab 2: Computer Vision Architectures in PyTorch (GPU Mode)
**CNN vs. Faster R-CNN vs. Fine-Tuned AlexNet & VGG16 vs. Vision Transformer (ViT)**

## 📌 Objective
The objective of this lab is to design, train, fine-tune, and compare multiple Deep Learning architectures for Computer Vision on the **MNIST dataset** using **PyTorch** in **GPU mode (`Tesla T4`)**:
1. **Part 1:** Build a custom **CNN**, adapt **Faster R-CNN** for digit detection/classification, and fine-tune pretrained **AlexNet** and **VGG16** architectures.
2. **Part 2:** Implement a **Vision Transformer (ViT)** from scratch (Patchification, Linear Projection, `[CLS]` token, Sinusoidal Positional Encoding, and Multi-Head Self-Attention blocks) and compare its performance with the CNN/R-CNN family.

---

## Environment & Hardware
* **Framework:** PyTorch & Torchvision
* **Hardware Accelerator:** NVIDIA Tesla T4 GPU (`cuda:0`) on Google Colab
* **Dataset:** MNIST (`60,000` training images, `10,000` test images, $28 \times 28$ grayscale digits `0–9`)

---

## Experimental Results & Architecture Comparison

| Model Architecture | Test Accuracy (%) | Macro F1-Score | Test Loss | GPU Training Time (s) | Notes / Configuration |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Custom CNN** | **98.99%** | **0.9899** | **0.0290** | 89.3s | 5 epochs, full 60k train / 10k test ($28 \times 28$) |
| **Faster R-CNN** | 98.40% | 0.9835 | 0.7823* | 38.5s | 2 epochs, 2k train / 500 test subset (Bounding Box + RoI) |
| **Vision Transformer (ViT)** | 96.95% | 0.9693 | 0.0992 | 114.3s | 5 epochs, full 60k train / 10k test ($7 \times 7$ patches of $4 \times 4$) |
| **AlexNet (Fine-Tuned)** | 94.90% | 0.9486 | 0.1608 | **31.6s** | 3 epochs, 10k train / 2k test, frozen backbone ($64 \times 64$) |
| **VGG16 (Fine-Tuned)** | 93.75% | 0.9394 | 0.2178 | 48.6s | 3 epochs, 10k train / 2k test, frozen backbone ($64 \times 64$) |

*\*Note: Faster R-CNN loss is a multi-task sum of RPN objectness, RPN box regression, RoI classification, and RoI box regression losses.*

---

## Key Findings & Conclusions

### 1. Custom CNN vs. Faster R-CNN
* **Custom CNN** achieved the highest accuracy (**98.99%**) and lowest loss (**0.0290**).
* **Faster R-CNN** achieved **98.40%** accuracy while simultaneously localizing each digit with a bounding box. However, because Faster R-CNN evaluates region proposals and bounding-box regressions, it is ~30x more computationally expensive per image than a standard CNN classifier.

### 2. Transfer Learning (`AlexNet` & `VGG16`) vs. Task-Specific CNN
* Pretrained models like **AlexNet** (`60M` parameters) and **VGG16** (`138M` parameters) are optimized for complex $224 \times 224$ RGB natural scenes. On $28 \times 28$ grayscale digits (upscaled to $64 \times 64$ to survive 5 max-pooling stages) with frozen intermediate feature layers, they achieved **94.90%** and **93.75%** accuracy. A lightweight, custom 2-layer CNN trained end-to-end is both more accurate and vastly more parameter-efficient for MNIST.

### 3. Vision Transformer (ViT) vs. Convolutional Networks
* Built from scratch, our **ViT** achieved **96.95% accuracy**, outperforming the fine-tuned `AlexNet` and `VGG16` models.
* Compared to the **Custom CNN (98.99%)**, ViT required slightly more training time (**114.3s** vs. **89.3s**) and had a ~2% lower accuracy after 5 epochs. This illustrates the core difference between architectures: CNNs leverage **inductive biases** (translation invariance and local $3 \times 3$ receptive fields) to learn digit strokes immediately, whereas ViT must learn spatial relationships between $4 \times 4$ patches from scratch using global **Multi-Head Self-Attention**.

---

## Synthesis
This lab provided hands-on experience implementing the evolution of Computer Vision in PyTorch on GPU: from classic **Convolutional Neural Networks (LeNet-style)** and deep ImageNet backbones (**AlexNet, VGG16**) to two-stage object detectors (**Faster R-CNN**) and modern attention-based **Vision Transformers (ViT)**.
