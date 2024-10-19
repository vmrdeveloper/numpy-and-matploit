import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Creamos los array
dias = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
temperatura = np.array([22, 24, 25, 27, 30, 32, 28, 25, 26, 27])

#calculamos la temperatura media, la maxima y la minima
maxtemp = np.max(temperatura)
midtemp = np.average(temperatura)
mintemp = np.min(temperatura)

print(maxtemp)
print(midtemp)
print(mintemp)

plt.plot(dias, temperatura, marker='o')
plt.xlabel("Dias de la semana")
plt.ylabel("Temperatura promedio diaria")
plt.title("Promedio de temperatura segun los dias de la semana")
plt.show()