# 🎨 Lab Color Prediction using Deep Learning (ΔE CMC Optimized)

A deep learning-based system designed to predict **CIE LAB color values** from spectral reflectance and pigment composition data, optimized using the **ΔE CMC perceptual color difference metric**.

The project integrates a **K-Fold cross-validation training pipeline** with an **interactive Streamlit dashboard** for real-time evaluation and visualization.

---

# 🎯 Problem Statement

Accurate color prediction is critical in industries such as **paint manufacturing, textiles, and printing**, where even small perceptual differences can lead to significant quality issues.

Traditional regression models minimize numerical error (MSE), but:

* ❌ They do not align with **human color perception**
* ❌ Small numeric errors may still produce **visibly different colors**

This project addresses the problem by:

> Optimizing predictions using **ΔE CMC**, a perceptual color difference metric aligned with human vision.

---

# 🧠 Project Objective

To develop a machine learning model that:

* Predicts **LAB color space values (L, a, b)**
* Minimizes **perceptual color error (ΔE CMC)**
* Ensures **robust generalization using K-Fold Cross Validation**
* Provides **interactive visualization via a web dashboard**

---

# ⚙️ How It Works

The system follows a structured ML pipeline:

```
Dataset Input (Spectral + Pigments)
            ↓
Data Cleaning & Preprocessing
            ↓
Feature Scaling (StandardScaler)
            ↓
Deep Neural Network (LeakyReLU)
            ↓
ΔE CMC Loss Optimization
            ↓
K-Fold Cross Validation (10 folds)
            ↓
Performance Aggregation
            ↓
Streamlit Dashboard Visualization
```

---

# 🧠 Model Architecture

```
Input Layer
   ↓
Dense (768) + LeakyReLU
   ↓
Dense (512) + LeakyReLU
   ↓
Dense (256) + LeakyReLU
   ↓
Dense (128) + LeakyReLU
   ↓
Output Layer (3 → L, a, b)
```

### Key Design Choices:

* Activation: **LeakyReLU (α = 0.05)**
* Optimizer: **Adam (lr = 0.0005)**
* Loss Function: **ΔE CMC (Perceptual Loss)**
* Validation: **10-Fold Cross Validation**

---

## ⚙️ Model Development & Hyperparameter Tuning

Before finalizing the model architecture, extensive experimentation was performed using **hyperparameter tuning**.

Instead of fixing the network design manually, we explored multiple configurations by varying:

* Number of hidden layers
* Number of neurons per layer
* Activation functions (ReLU, LeakyReLU, etc.)
* Optimizers (Adam, RMSprop, etc.)
* Learning rates

The tuning process was carried out using a systematic search strategy (e.g., KerasTuner / manual experimentation), where different architectures were trained and evaluated based on **ΔE CMC performance**.

### 🎯 Outcome

Based on empirical results, the following configuration provided the best performance:

* Dense Layers: **768 → 512 → 256 → 128**
* Activation: **LeakyReLU (α = 0.05)**
* Optimizer: **Adam (learning rate = 0.0005)**
* Loss Function: **ΔE CMC**

This approach ensured that the final model is:

* Data-driven
* Performance-optimized
* Not based on arbitrary design choices

---

💡 This step was crucial in achieving high accuracy (~90%+) and stable generalization across folds.

---

# 📊 Evaluation Metrics

* Mean ΔE (Color Difference)
* Median ΔE
* Accuracy (ΔE ≤ 0.6 threshold)
* 95% Confidence Interval

---

# 📂 Dataset Description

The dataset contains:

### 🔹 Input Features

1. **Spectral Reflectance Values**

   * Range: `R400 – R700`
   * Represents how light reflects across wavelengths

2. **Pigment Composition**

   * Columns like: `B352S, B610, B616, ...`
   * Represents chemical composition

### 🔹 Target Output

* **CIE LAB Color Values**

  * L → Lightness
  * a → Green–Red axis
  * b → Blue–Yellow axis

### 🔹 Preprocessing Steps

* Removal of non-numeric columns (`RGB`, `ImitationID`)
* Conversion to numeric values
* Handling missing values
* Feature scaling

---

# 🖥️ Streamlit Web Application

An interactive dashboard is built using **Streamlit**.

### Features:

* 📂 Upload dataset (CSV)
* 🚀 Run full **K-Fold training**
* 📊 Visualize model performance
* 🎯 Adjust ΔE threshold dynamically

## 📈 Model Performance Visualizations

### 🔹 ΔE Distribution

![DeltaE Distribution]([deltaE distribution.png](https://github.com/HIMANSHI549/Akzonobel_lab/blob/master/deltaE%20distribution.png))

### 🔹 Accuracy vs Threshold

![Accuracy Curve]([images/accuracy.png](https://github.com/HIMANSHI549/Akzonobel_lab/blob/master/accuracy%20vs%20threshold.png))

### 🔹 True vs Predicted (L Channel)

![Scatter Plot]([images/scatter.png](https://github.com/HIMANSHI549/Akzonobel_lab/blob/master/true%20vs%20pred%20L.png))

### 🔹 Training Loss(all folds)

![Scatter Plot](### 🔹 True vs Predicted (L Channel)

![Scatter Plot]([images/scatter.png](https://github.com/HIMANSHI549/Akzonobel_lab/blob/master/true%20vs%20pred%20L.png))


# 📁 Project Structure

```
lab-color-prediction/
│
├── app.py                # Streamlit dashboard
├── model.py              # K-Fold training pipeline
├── code.py               # Standalone script (Colab/Kaggle)
├── requirements.txt
├── README.md
│
└── data/
    └── merged_trainval.csv
```

---

# ▶️ Usage

## 1. Run Streamlit App

```bash
streamlit run app.py
```

---

## 2. Run Standalone Code (Colab/Kaggle)

```bash
python code.py
```

---

# 📄 `code.py` (Standalone Execution File)

This file allows users to:

* Run full training
* Reproduce results
* Validate performance outside Streamlit

```python
import pandas as pd
from model import LabColorModel

# Load dataset
df = pd.read_csv("merged_trainval.csv")

# Clean data
df = df.drop(['RGB','R','G','B'], axis=1, errors='ignore')
df = df.drop(['ImitationID'], axis=1, errors='ignore')

lab_cols = ["L_D65-10°","a_D65-10°","b_D65-10°"]
spec_cols = [c for c in df.columns if c.startswith('R') and len(c)==4]

pig_cols = ["B352S","B610","B616","B622","B623","B625","B628",
            "B641","B642","B648","B649","B651","B653",
            "B662","B664","B671","B673"]

for col in spec_cols + pig_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[spec_cols + pig_cols] = df[spec_cols + pig_cols].fillna(0)

X = df[spec_cols + pig_cols].values
y = df[lab_cols].values

# Train model
model = LabColorModel()
deltaE, _, _ = model.train_kfold(X, y)

print("Mean ΔE:", deltaE.mean())
print("Accuracy (≤0.6):", (deltaE <= 0.6).mean())
```

---

# 📦 Requirements

* Python 3.9+
* TensorFlow
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

Install using:

```bash
pip install -r requirements.txt
```

---

# 📈 Results

* Mean ΔE: ~0.3–0.5
* Accuracy (ΔE ≤ 0.6): ~90%+
* Strong correlation between predicted and actual LAB values

---

# 💡 Applications

* 🎨 Paint & Coating Industry (AkzoNobel use-case)
* 🖨️ Printing & Color Calibration
* 👕 Textile Industry
* 🧪 Pigment Formulation

---

# ⚠️ Limitations

* Model depends on dataset quality and distribution
* Does not handle unseen pigment combinations well
* Computationally expensive due to K-Fold training

---

## 🔮 Future Work

* Develop an inverse model to **predict pigment composition from target LAB color values**
* Predict the **optimal combination of 6 pigments and their proportions** required to generate a desired color
* Formulate the problem as a **multi-output regression / optimization task**
* Incorporate **constraints (sum of proportions = 1, non-negativity)**
* Explore **hybrid approaches (Deep Learning + Optimization algorithms)**
* Build an **interactive color formulation tool** for real-time industrial use
* Extend the system for **multi-illuminant and cross-material color matching**


---

# 👩‍💻 Author

**Himanshi and Aayushi**
B.Tech Final Year Project

---

# 📜 Disclaimer

This project is developed for academic and research purposes.
It is not intended for direct industrial deployment without further validation.
