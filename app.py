import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from model import LabColorModel

st.set_page_config(layout="wide")
st.title("🎨 Lab Color Prediction (K-Fold + ΔE CMC)")

# create images folder
os.makedirs("images", exist_ok=True)

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.write(df.head())

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

        st.metric("Accuracy (≤0.6)", f"{np.mean(deltaE <= 0.6)*100:.2f}%")
        st.write("Mean ΔE:", np.mean(deltaE))

        col1, col2 = st.columns(2)

        # =============================
        # Histogram
        # =============================
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.hist(deltaE, bins=50)
            ax1.set_title("DeltaE Distribution")

            fig1.savefig("images/deltaE.png", dpi=300, bbox_inches='tight')
            st.pyplot(fig1)

        # =============================
        # Scatter
        # =============================
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test[:,0], y_pred[:,0], alpha=0.4)

            ax2.plot([y_test[:,0].min(), y_test[:,0].max()],
                     [y_test[:,0].min(), y_test[:,0].max()])

            ax2.set_title("True vs Predicted L")

            fig2.savefig("images/scatter.png", dpi=300, bbox_inches='tight')
            st.pyplot(fig2)

        # =============================
        # Accuracy Curve
        # =============================
        thresholds = np.linspace(0, 2, 50)
        accs = [np.mean(deltaE <= t) for t in thresholds]

        fig3, ax3 = plt.subplots()
        ax3.plot(thresholds, accs)
        ax3.axvline(0.6)
        ax3.set_title("Accuracy vs Threshold")

        fig3.savefig("images/accuracy.png", dpi=300, bbox_inches='tight')
        st.pyplot(fig3)
