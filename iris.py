from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import cdist
import random

iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
    
T =[[6.8,3.2,4.7,1.5], [5,3.5,1.4,0.2], [100, 1, 1, 1]]
pred = kmeans.predict(T)

pred


////////////////////////////////////////////
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Doc du lieu
df = pd.read_csv('data_linear.csv')

Dt =pd.DataFrame(df['Diện tích'])
Gn = pd.DataFrame(df['Giá'])

model = linear_model.LinearRegression()
model_fit = model.fit(Dt,Gn)

model.coef_

model.intercept_
model.score(Dt,Gn)
#Dự đoán
X = [[50]]
y_pred = model_fit.predict(X)
y_pred
df.plot(kind='scatter',x = 'Diện tích', y ='Giá')
plt.plot(Dt,model_fit.predict(Dt),color = 'red',lineWidth=2)
plt.scatter(X,model.coef_*X+model.intercept_,color='black')


////////////////////////

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets
from sklearn import linear_model

df = datasets.load_iris()

from sklearn.model_selection import train_test_split
# Tach tap du lieu
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.1)
model = linear_model.LogisticRegression()
model_fit = model.fit(X_train, y_train)
model.coef_
model.intercept_
T =[[5,3.5,1.4,0.2]]
pred = model.predict(T)
print(pred)

n = int(input())
a = []

for i in range(0, n):
    a.append([])
    for j in range(0, 4):
        x = float(input())
        a[i].append(x)

pred = model.predict(a)
pred

////////////////////////////////////////

import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv(r"/content/drive/MyDrive/Bai_Tap_May_Hoc/Linear Registion/Data/data_linear .csv")

x = np.array(data['Diện tích'])
Y = np.array(data['Giá'])
X = np.zeros(shape=(len(X),2))
X[:,1]=x
X
model = linear_model.LinearRegression()
model.fit(X,Y)

result = model.predict(X)
for i in range(len(result)):
  print(round(result[i],2))

import matplotlib.pyplot as plt

plt.plot(x,Y)
plt.show()


