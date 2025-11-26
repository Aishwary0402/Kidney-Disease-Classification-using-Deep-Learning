# Kidney Disease Classification using Deep Learning & Hybrid Models 🩺🧠

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)

## 📖 Overview

This project utilizes Deep Learning techniques to classify kidney medical images into four distinct categories. The goal is to assist in the automated diagnosis of kidney abnormalities using Computer Vision.

The project evolves from a custom Convolutional Neural Network (CNN) to Transfer Learning approaches, culminating in a **Hybrid Model** that combines **ResNet50V2** and **EfficientNetV2B2** for improved accuracy.

### 🎯 Classification Classes
The model classifies images into the following 4 classes:
1.  **Cyst**
2.  **Normal**
3.  **Stone**
4.  **Tumor**

---

## ⚙️ Methodology & Pipeline

### 1. Data Preprocessing
* **Resizing:** All images are resized to `224x224` pixels.
* **Grayscale to RGB:** Dataset is loaded in grayscale but converted to RGB dynamically to satisfy Transfer Learning input requirements.
* **Class Balancing:** `compute_class_weight` is used to calculate weights for the classes to handle dataset imbalance during training.

### 2. Data Augmentation
To prevent overfitting and improve generalization, the following augmentations are applied dynamically:
* Random Flip (Horizontal)
* Random Rotation (0.25)
* Random Zoom (0.25)
* Random Brightness/Contrast adjustments

### 3. Model Architectures Implemented
We experimented with three distinct approaches:

* **Model A: Improved Custom CNN:** A standard CNN with BatchNormalization, MaxPooling, and Dropout layers. (Baseline)
* **Model B: Transfer Learning (ResNet50):** Using pre-trained ImageNet weights with a custom classification head.
* **Model C: Hybrid Ensemble (Final Model):**
    * Combines feature extraction from **ResNet50V2** and **EfficientNetV2B2**.
    * Features are concatenated and passed through a dense classification head.
    * Uses L2 Regularization to prevent overfitting.

### 4. Training Strategy (Fine-Tuning)
For the Hybrid Model, a two-phase training strategy was employed:
1.  **Phase 1 (Feature Extraction):** The base models are frozen, and only the top classification layers are trained.
2.  **Phase 2 (Fine-Tuning):** The top layers of the base models are unfrozen, and the entire model is retrained with a very low learning rate (`1e-5`) to adapt the pre-trained features to the specific kidney dataset.

---

## 📊 Results

The models were evaluated using Accuracy, Confusion Matrices, and Classification Reports (Precision, Recall, F1-Score).

| Model | Approach | Approx. Accuracy |
| :--- | :--- | :--- |
| Custom CNN | Baseline | ~25% |
| ResNet50 | Transfer Learning | ~71% |
| **Hybrid (ResNet + EffNet)** | **Fine-Tuning** | **~73%** |

*Note: Accuracy improved significantly using the Hybrid architecture and fine-tuning strategy compared to the baseline CNN.*

---

## 🛠️ Technologies Used

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn

---

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Aishwary0402/Kidney_Disease.git](https://github.com/Aishwary0402/Kidney_Disease.git)
    cd Kidney_Disease
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Dataset Setup**
    * Ensure your dataset is organized into folders: `train`, `test`, and `valid`.
    * Update the `DATASET_DIR` variable in the notebook to point to your local dataset path.

4.  **Run the Notebook**
    Launch Jupyter Notebook and run `Kidney_Disease.ipynb`.

---

## 📈 Future Improvements
* Implement Vision Transformers (ViT) for comparison.
* Collect more data to improve class balance naturally.
* Deploy the model using Flask or Streamlit for a web-based demo.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Aishwary0402/Kidney_Disease/issues).
