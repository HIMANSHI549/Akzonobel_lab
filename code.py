import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam


class LabColorModel:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None

        self.all_deltaE = []
        self.all_y_test = []
        self.all_y_pred = []
        self.fold_losses = []

    # =============================
    # LOAD & PREPROCESS
    # =============================
    def load_data(self):
        self.df = pd.read_csv(self.file_path)

        self.df = self.df.drop(['RGB','R','G','B'], axis=1, errors='ignore')
        self.df = self.df.drop(['ImitationID'], axis=1, errors='ignore')

        self.lab_cols = ["L_D65-10°","a_D65-10°","b_D65-10°"]

        self.pig_cols = ["B352S","B610","B616","B622","B623","B625","B628",
                         "B641","B642","B648","B649","B651","B653",
                         "B662","B664","B671","B673"]

        self.spec_cols = [c for c in self.df.columns if c.startswith('R') and len(c)==4]

        for col in self.spec_cols + self.pig_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        self.df[self.spec_cols + self.pig_cols] = self.df[self.spec_cols + self.pig_cols].fillna(0)
        self.df.dropna(subset=self.lab_cols, inplace=True)

        # FIX dtype
        self.X = self.df[self.spec_cols + self.pig_cols].values.astype(np.float32)
        self.y = self.df[self.lab_cols].values.astype(np.float32)

        print("Data Loaded:", self.X.shape)

    # =============================
    # DELTA E
    # =============================
    def delta_e_cmc(self, y_true, y_pred):

        eps = 1e-8

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        L1,a1,b1 = y_true[:,0], y_true[:,1], y_true[:,2]
        L2,a2,b2 = y_pred[:,0], y_pred[:,1], y_pred[:,2]

        C1 = tf.sqrt(a1**2 + b1**2 + eps)
        C2 = tf.sqrt(a2**2 + b2**2 + eps)

        dL = L1 - L2
        dC = C1 - C2

        da = a1 - a2
        db = b1 - b2

        dH = tf.sqrt(tf.maximum(da**2 + db**2 - dC**2, eps))

        SL = tf.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1 + eps))
        SC = (0.0638 * C1) / (1 + 0.0131 * C1 + eps) + 0.638

        H1 = tf.atan2(b1, a1) * (180.0 / np.pi)
        H1 = tf.where(H1 < 0, H1 + 360, H1)

        T = tf.where(
            (H1 >= 164) & (H1 <= 345),
            0.56 + tf.abs(0.2 * tf.cos((H1 + 168) * np.pi/180)),
            0.36 + tf.abs(0.4 * tf.cos((H1 + 35) * np.pi/180))
        )

        F = (C1**4) / (C1**4 + 1900 + eps)
        SH = SC * (F * T + 1 - F) + eps

        deltaE = tf.sqrt(
            (dL/(2*SL + eps))**2 +
            (dC/(SC + eps))**2 +
            (dH/(SH + eps))**2
        )

        return deltaE

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(self.delta_e_cmc(y_true, y_pred))

    # =============================
    # MODEL
    # =============================
    def build_model(self, input_dim):
        model = Sequential([
            Input(shape=(input_dim,)),

            Dense(768), LeakyReLU(0.05),
            Dense(512), LeakyReLU(0.05),
            Dense(256), LeakyReLU(0.05),
            Dense(128), LeakyReLU(0.05),

            Dense(3, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=self.loss_fn
        )

        return model

    # =============================
    # TRAIN
    # =============================
    def train(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        fold = 1

        for train_idx, test_idx in kf.split(self.X):

            print(f"\nFOLD {fold}")

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = self.build_model(X_train.shape[1])

            history = model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=32,
                verbose=0
            )

            self.fold_losses.append(history.history['loss'])

            y_pred = model.predict(X_test, verbose=0)

            # ensure float32
            y_test_tf = tf.cast(y_test, tf.float32)
            y_pred_tf = tf.cast(y_pred, tf.float32)

            deltaE = self.delta_e_cmc(y_test_tf, y_pred_tf).numpy()

            self.all_deltaE.extend(deltaE)
            self.all_y_test.append(y_test)
            self.all_y_pred.append(y_pred)

            print("Mean ΔE:", np.mean(deltaE))
            print("Accuracy (≤0.6):", np.mean(deltaE <= 0.6))

            fold += 1

        self.all_deltaE = np.array(self.all_deltaE)
        self.all_y_test = np.vstack(self.all_y_test)
        self.all_y_pred = np.vstack(self.all_y_pred)

    # =============================
    # VISUALS (UPGRADED)
    # =============================
    def plot_results(self):

        plt.figure(figsize=(18, 12))

        # 1. Loss
        plt.subplot(2,2,1)
        for loss in self.fold_losses:
            plt.plot(loss, alpha=0.6)
        plt.title("Training Loss (All Folds)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        # 2. DeltaE
        plt.subplot(2,2,2)
        plt.hist(self.all_deltaE, bins=60)
        plt.axvline(0.6, linestyle='--')
        plt.title("DeltaE Distribution")
        plt.xlabel("DeltaE")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 3. Accuracy curve
        plt.subplot(2,2,3)
        thresholds = np.linspace(0, 2, 50)
        acc = [np.mean(self.all_deltaE <= t) for t in thresholds]
        plt.plot(thresholds, acc)
        plt.axvline(0.6, linestyle='--')
        plt.title("Accuracy vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.grid(True)

        # 4. True vs Predicted
        plt.subplot(2,2,4)
        plt.scatter(self.all_y_test[:,0], self.all_y_pred[:,0], alpha=0.4)

        min_v = self.all_y_test[:,0].min()
        max_v = self.all_y_test[:,0].max()
        plt.plot([min_v, max_v], [min_v, max_v], linestyle='--')

        plt.title("True vs Predicted L")
        plt.xlabel("True L")
        plt.ylabel("Predicted L")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # =============================
    # METRICS
    # =============================
    def evaluate(self):

        print("\nFINAL RESULTS")
        print("Mean ΔE:", np.mean(self.all_deltaE))
        print("Median ΔE:", np.median(self.all_deltaE))
        print("Accuracy (≤0.6):", np.mean(self.all_deltaE <= 0.6))

        print("\n95% Confidence Interval:")
        print(np.percentile(self.all_deltaE, 2.5),
              np.percentile(self.all_deltaE, 97.5))


# =============================
# RUN
# =============================

model = LabColorModel("/kaggle/input/datasets/himanshi549/dataset-1/merged_trainval.csv")

model.load_data()
model.train()
model.evaluate()
model.plot_results()
