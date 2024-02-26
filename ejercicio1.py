import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class Perceptron(): 
    def __init__(self, aprendizaje=0.1, n_iter=50):
        self.aprendizaje = aprendizaje
        self.n_iter = n_iter

    def fit(self, X, y):      
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
        self.errors_ = []   

        for _ in range(self.n_iter):
            errors = 0
            for xi, label in zip(X, y):
                update = self.aprendizaje * (label-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


n_iter = input('Cuantas iteraciones?(10)\n')
aprendizaje = input('Que taza de aprendizaje quiere?(0.1)\n')


print("Cargando archivo de entrenamiento...")
df = pd.read_csv('OR_trn.csv', header=None)

X = df.iloc[0:2000, [0,1]].values
y = df.iloc[0:2000, 2].values
y = np.where(y == -1, -1, 1)

ppn = Perceptron(aprendizaje=0.1, n_iter=10) 
print("Entrenando...")
ppn.fit(X, y) 

print("Mostrando resultados...")

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel("Iteracion")
plt.ylabel("Numero de errores")

plt.show()

OR_trn = np.array(df.iloc[0:2000, [0,2]])

point1, point2 = [-1, 0], [1, 0]
plt.axline(point1, point2)
plt.plot(OR_trn[:, 0], OR_trn[:, 1], "bo", label="OR_trn")

plt.xlabel("B")
plt.ylabel("A")
plt.legend(loc='upper left')
plt.show()


print("Cargando archivo de prueba...")
df = pd.read_csv('OR_tst.csv', header=None)

X = df.iloc[0:2000, [0,1]].values
y = df.iloc[0:2000, 2].values
y = np.where(y == -1, -1, 1)

ppn = Perceptron(aprendizaje=0.1, n_iter=10) 
print("Probando...")
ppn.fit(X, y) 

print("Mostrando resultados...")

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel("Iteracion")
plt.ylabel("Numero de errores")

plt.show()

OR_trn = np.array(df.iloc[0:2000, [0,2]])

point1, point2 = [-1, 0], [1, 0]
plt.axline(point1, point2)
plt.plot(OR_trn[:, 0], OR_trn[:, 1], "bo", label="OR_trn")

plt.xlabel("B")
plt.ylabel("A")
plt.legend(loc='upper left')
plt.show()
