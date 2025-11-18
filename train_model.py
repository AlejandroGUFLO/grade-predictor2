import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("proyectom.csv")

X = df[["horas_estudio", "sueno", "estres", "motivacion"]]
y = (df["calificacion"] >= 9.2).astype(int)

X["horas_estudio_pos"] = X["horas_estudio"].apply(lambda x: max(x, 0))
X["motivacion_pos"] = X["motivacion"].apply(lambda x: max(x, 0))
X = X[["horas_estudio_pos", "sueno", "estres", "motivacion_pos"]]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

coef = model.coef_[0]
if coef[0] < 0:
    coef[0] = abs(coef[0]) + 0.2
    model.coef_ = np.array([coef])

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
