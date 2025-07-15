# Pneumonia Detection from Chest X-Rays using CNN

This project involves building a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Pneumonia** or **Normal**, using the publicly available dataset from Kaggle. It was developed as part of an academic assignment focused on medical image classification.

---

## 📂 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset is structured as:
```bash
chest_xray/
└── train/
├── NORMAL/
└── PNEUMONIA/
```

> *Only the `train` directory was provided. Validation split is handled programmatically using `ImageDataGenerator`.*

---

## 🧪 Objective

To classify chest X-ray images into:
- **NORMAL**: No signs of pneumonia
- **PNEUMONIA**: Evidence of pneumonia present

---

## 🔧 Preprocessing

- Images resized to **150x150**
- Pixel values normalized to [0, 1]
- Data augmentation:
  - Shear
  - Zoom
  - Horizontal flip
- 80/20 split for training and validation using `ImageDataGenerator`

---

## 🧠 Model Architecture

A custom CNN built using TensorFlow/Keras:

- `Conv2D` layers with ReLU activations
- `MaxPooling2D` layers
- `Dropout` for regularization
- Fully-connected `Dense` layers
- Output layer with `sigmoid` activation (binary classification)

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🚀 How to Run (in Google Colab)

1. Upload the `.zip` file containing the dataset.
2. Unzip the file inside Colab.
3. Run the cells to preprocess, train, and evaluate the model.

---

## 📈 Results

Evaluation is done on the validation set (20% of training data). Metrics are printed in a classification report along with a confusion matrix heatmap.

---

## 📝 Notes

- No separate test set was provided. Validation was performed by splitting the training data.
- For more robust evaluation, consider adding a dedicated test set or using k-fold cross-validation.

---

## 🧩 Future Work

- Use pretrained models (Transfer Learning with MobileNet or ResNet)
- Explore Grad-CAM for visualizing attention areas
- Hyperparameter tuning for performance improvement

---

## 📌 Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```
### 📚 Acknowledgements

    Dataset by Paul Mooney on Kaggle

    Built as part of the academic curriculum for deep learning in medical imaging
  
