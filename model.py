import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam


class LabColorModel:

    def __init__(self):
        self.all_deltaE = []
        self.all_y_test = []
        self.all_y_pred = []

    # =============================
    # ΔE CMC
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
            Dense(3)
        ])

        model.compile(
            optimizer=Adam(0.0005),
            loss=self.loss_fn
        )

        return model

    # =============================
    # K-FOLD TRAINING
    # =============================
    def train_kfold(self, X, y):

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):

            print(f"FOLD {fold}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = self.build_model(X_train.shape[1])

            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            y_pred = model.predict(X_test, verbose=0)

            deltaE = self.delta_e_cmc(y_test, y_pred).numpy()

            self.all_deltaE.extend(deltaE)
            self.all_y_test.append(y_test)
            self.all_y_pred.append(y_pred)

        self.all_deltaE = np.array(self.all_deltaE)
        self.all_y_test = np.vstack(self.all_y_test)
        self.all_y_pred = np.vstack(self.all_y_pred)

        return self.all_deltaE, self.all_y_test, self.all_y_pred
