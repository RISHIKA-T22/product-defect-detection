# Automated Product Defect Detection

## 1. Project Overview

This project implements an automated product defect detection system using Computer Vision and Transfer Learning.

The system classifies product images into two categories:

- Defective
- Normal

A pretrained ResNet18 model is used for binary image classification.

---

## 2. Objective

The objective is to:

- Prepare and preprocess the image dataset.
- Resize images to 224 × 224.
- Apply data augmentation only to training images.
- Split the dataset into 80% training and 20% validation.
- Train a binary image classification model using transfer learning.
- Evaluate the model using Accuracy, Precision, Recall, F1-Score and Confusion Matrix.
- Analyze incorrectly classified images.

---

## 3. Dataset

The dataset contains two classes:

- `def_front` - Defective products
- `ok_front` - Normal products

For this practical assessment, 100 images were selected:

- 50 Defective images
- 50 Normal images

The selected dataset was divided into:

- Training: 80 images
- Validation: 20 images

The original dataset was not modified.

---

## 4. Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Scikit-learn
- Matplotlib

---

## 5. Image Preprocessing

The following preprocessing steps were used:

1. Images were resized to 224 × 224 pixels.
2. Images were converted to tensors.
3. ImageNet normalization was applied.
4. Training images were augmented using:
   - Random Horizontal Flip
   - Random Rotation
5. No random augmentation was applied to validation images.

---

## 6. Train-Validation Split

The dataset was divided using an 80:20 split.

| Dataset | Number of Images |
|---|---:|
| Training | 80 |
| Validation | 20 |
| Total | 100 |

---

## 7. Transfer Learning Model

A pretrained ResNet18 model was used.

### Model Configuration

- Pretrained ResNet18 weights were used.
- Most pretrained feature extraction layers were frozen; the final residual block (`layer4`) was unfrozen for fine-tuning.
- The final fully connected layer was replaced.
- The final layer outputs two classes:
  - Defective
  - Normal

### Trainable and Frozen Layers

**Frozen:**
- ResNet18 layers up to `layer4` (initial conv layers and residual blocks 1–3)

**Trainable:**
- `layer4` (final residual block)
- Final fully connected classification layer

---

## 8. Training

### Training Configuration

- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 8
- Batch Size: 16
- Device: CPU

Training and validation loss and accuracy were recorded for every epoch.

### Training Results

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.6008 | 70.00% | 0.5313 | 70.00% |
| 2 | 0.3504 | 82.50% | 0.4325 | 70.00% |
| 3 | 0.2799 | 88.75% | 0.3829 | 70.00% |
| 4 | 0.2069 | 93.75% | 0.3883 | 70.00% |
| 5 | 0.1513 | 96.25% | 0.5905 | 65.00% |
| 6 | 0.1612 | 93.75% | 0.7058 | 60.00% |
| 7 | 0.0743 | 97.50% | 0.5577 | 75.00% |
| 8 | 0.0844 | 97.50% | 0.2322 | 80.00% |

---

## 9. Final Performance

The model was evaluated on the 20 validation images.

| Metric | Result |
|---|---:|
| Accuracy | 80.00% |
| Precision | 100.00% |
| Recall | 60.00% |
| F1-Score | 75.00% |

---

## 10. Confusion Matrix

The confusion matrix obtained from the validation data is:

```text
[[6, 4],
 [0, 10]]
```

---

## 11. Error Analysis

4 misclassifications were found on the validation set, all false negatives (actual `def_front` predicted as `ok_front`). The 3 saved examples:

| # | Actual | Predicted | Observation | Likely Reason |
|---|---|---|---|---|
| 1 | def_front | ok_front | Faint scratch/wear marks on outer rim | Low-contrast defect blends into the metallic surface texture |
| 2 | def_front | ok_front | No clearly visible surface defect | Likely a subtle internal/dimensional flaw, hard to detect visually |
| 3 | def_front | ok_front | Very faint marking near inner rim edge | Fine detail lost after resizing to 224×224 |

**Conclusion:** The model relies primarily on overall shape and symmetry. Small, low-contrast casting defects (hairline scratches, minor blowholes) remain harder to detect given the limited number of defective training samples (40), even after fine-tuning `layer4`.

---

## 12. Files

- `main.py` — full pipeline (preprocessing, training, evaluation, error analysis)
- `model/trained_model.pth` — saved model weights
- `results/` — confusion matrix + misclassified image plots