import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import LabColorModel

st.set_page_config(layout="wide")

st.title("🎨 Lab Color Prediction (K-Fold Evaluation)")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    # CLEAN
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

    X = df[spec_cols + pig_cols].values.astype(np.float32)
    y = df[lab_cols].values.astype(np.float32)

    if st.button("Run K-Fold Training"):

        model = LabColorModel()

        deltaE, y_test, y_pred = model.train_kfold(X, y)

        st.success("Training Completed!")

        # Metrics
        st.metric("Accuracy (≤0.6)", f"{np.mean(deltaE <= 0.6)*100:.2f}%")
        st.write("Mean ΔE:", np.mean(deltaE))

        col1, col2 = st.columns(2)

        # Histogram
        with col1:
            fig, ax = plt.subplots()
            ax.hist(deltaE, bins=50)
            ax.set_title("DeltaE Distribution")
            st.pyplot(fig)

        # Scatter
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test[:,0], y_pred[:,0], alpha=0.4)
            ax.plot([y_test[:,0].min(), y_test[:,0].max()],
                    [y_test[:,0].min(), y_test[:,0].max()])
            ax.set_title("True vs Predicted L")
            st.pyplot(fig)
