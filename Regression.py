
import pandas as pd
import seaborn as sns

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Fiyat':[70, 90, 95, 120, 110]
    })

sns.lmplot(x='Metrekare', y='Fiyat', data = df)

"""y = wx + b

y = label (Fiyat)
w = Weight of x
x = Feature (Metrekare)
b = bias
"""

from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.DataFrame(data = {
    'Metrekare': [100, 150, 120, 300, 230],
    'Fiyat': [70, 90, 95, 120, 110]
    })

X = df.iloc[:, :-1].values # feature
y = df.iloc[:, -1].values # label

reg = LinearRegression().fit(X, y)

print(f"Score: {reg.score(X, y)}")
print(f"Coefficient: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

metrekare = int(input("Lütfen fiyat tahminini yapacağınız metrekareyi giriniz: "))

print(f"{metrekare} metrekare evin tahmini fiyatı: {reg.predict(np.array([[metrekare]]))}")

metrekare * reg.coef_[0] + float(reg.intercept_)

"""# Multiple Lineer Regresyon"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]
    })

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(df['Metrekare'], df['Bina Yaşı'], df['Fiyat'], cmap=cm.jet, linewidth=0.2)
ax.set_xlabel('Metrekare')
ax.set_ylabel('Fiyat')
ax.set_zlabel('Bina Yaşı')
plt.show()

"""y = w1x1 + w2x2 + c

y = label (Fiyat)
w1 = Weight of x1
x1 = Feature 1 (Metrekare)
w2 = Weight of x2
x2 = Feature 2 (Bina Yaşı)
c = bias
"""

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]
    })

X = df.iloc[: , :-1]
y = df.iloc[: , -1]

reg = LinearRegression().fit(X, y)

print(f"Score: {reg.score(X, y)}")
print(f"Coefficient: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

"""# Non-Lineer - Polynomial Regresyon"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 160, 200, 230],
    'Yükseklik':[50, 60, 70, 80, 90],
    'Genişlik': [10, 20, 20, 25, 40],
    'Fiyat':[70, 90, 95, 120, 110]
    })

df['Alan'] = df['Yükseklik'] * df['Genişlik']
df.drop(columns = ['Yükseklik', 'Genişlik'])

plt.plot(df['Alan'], df['Fiyat'])

import matplotlib.pyplot as plt

df = pd.DataFrame(data = {
    'Metrekare': range(1,101, 3)
    })

df['Fiyat'] = df['Metrekare'] ** 3

plt.plot(df['Metrekare'], df['Fiyat'])

"""# Regresyon Modelimizde Hata Kavramı"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Fiyat':[70, 90, 95, 120, 110]
    })

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

reg = LinearRegression().fit(X, y)

toplam_hata = 0
for idx in range(len(df)):
  gercek_fiyat = df.loc[idx]['Fiyat']
  metrekare = df.loc[idx]['Metrekare']

  tahmin_fiyat = reg.predict([[metrekare]])[0]

  # Hatali gosterim abs eklenmeli, olmamasi gerekiyor!
  print(f"{df.loc[idx]['Metrekare']} metrekare için gerçek fiyat: {gercek_fiyat}. Tahmin edilen fiyat: {tahmin_fiyat}. HATA: {gercek_fiyat - tahmin_fiyat}")
  toplam_hata = gercek_fiyat - tahmin_fiyat

print(f"\nToplam Hata: {toplam_hata}")

df.head()

"""# R2 (R squared)"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]})

df.head()

X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

X

y

reg = LinearRegression().fit(X, y)

y_pred = []

for idx in range(len(df)):
  metrekare = df.loc[idx]["Metrekare"]
  bina_yasi = df.loc[idx]["Bina Yaşı"]

  y_pred.append(reg.predict([[metrekare,bina_yasi]]))

print(reg.score(X, y)) # X -> Test verisi, y -> Test verilerinin gerçek label değerleri
print(r2_score(y, y_pred)) # y -> Gerçek label değerleri, y_pred -> model tarafından tahmin edilen y değerleri

"""# Mean Absolute Error (MAE)"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]
    })

X = df.iloc[:,:-1].values # feature values (take columns until last column)
y = df.iloc[:,-1].values # label values (take only last column)

reg = LinearRegression().fit(X, y)

y_pred = []

for idx in range(len(df)):
  metrekare = df.loc[idx]["Metrekare"]
  bina_yasi = df.loc[idx]["Bina Yaşı"]

  y_pred.append(reg.predict([[metrekare,bina_yasi]]))


print(mean_absolute_error(y, y_pred))

"""# Mean Squared Error (MSE)"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]})

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

reg = LinearRegression().fit(X, y)

y_pred = []

for idx in range(len(df)):
  metrekare = df.loc[idx]["Metrekare"]
  bina_yasi = df.loc[idx]["Bina Yaşı"]

  y_pred.append(reg.predict([[metrekare,bina_yasi]]))


print(mean_squared_error(y, y_pred))

"""# Verimizi Eğitim - Test - Validasyon Setlerine Ayırmak (train-test-validation split)"""

#train-test split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.DataFrame(data = {'Metrekare':[100, 150, 120, 300, 230, 175, 220, 270, 190, 220],
                          'Bina Yaşı':[5, 2, 6, 10, 3, 7, 6, 8, 9, 4],
                          'Fiyat':[70, 90, 95, 120, 110, 120, 95, 140, 220, 100]})

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train)

X_train

X_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)

print(X_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

print(X_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

print(X_train)

#train-test-validation split
df = pd.DataFrame(data = {'Metrekare':[100, 150, 120, 300, 230, 175, 220, 270, 190, 220],
                          'Bina Yaşı':[5, 2, 6, 10, 3, 7, 6, 8, 9, 4],
                          'Fiyat':[70, 90, 95, 120, 110, 120, 95, 140, 220, 100]})

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

X_train, X_validasyon, y_train, y_validasyon = train_test_split(X_train, y_train, test_size=2)


print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

print(X_validasyon.shape)
print(y_validasyon.shape)

print(X_train)

"""# L1 (Lasso) Regülarizasyon"""

from sklearn.linear_model import Lasso

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]})

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = Lasso(alpha=0.1)
model.fit(X, y)

print(model.coef_)
print(model.score(X, y))
print(model.intercept_)

print(model.predict([[300, 9]]))



"""# 2 (Ridge) Regülarizasyon"""

from sklearn.linear_model import Ridge

df = pd.DataFrame(data = {
    'Metrekare':[100, 150, 120, 300, 230],
    'Bina Yaşı':[5, 2, 6, 10, 3],
    'Fiyat':[70, 90, 95, 120, 110]
    })

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

model = Ridge(alpha=0.1)
model.fit(X, y)

print(model.coef_)
print(model.score(X, y))
print(model.intercept_)

model.predict([[300, 9]])

"""# Iris Dataset ile Gerçek Hayat Senaryosu
# Gathering the Data
"""

from sklearn.datasets import load_iris

iris = load_iris()

iris.feature_names

iris.data[:10]

iris.target_names

iris.target[:10]

feature_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
label_df = pd.DataFrame(data= iris.target, columns= ['species'])

feature_df.head()

label_df.head()

#concat dataframe to gather full dataset
iris_df = pd.concat([feature_df, label_df], axis= 1)

iris_df.head()

"""# Veriye İstatistiksel Bir Bakış"""

iris_df.describe().T

iris_df.info()

iris.target_names

import seaborn as sns

sns.pairplot(iris_df, hue = "species")

"""# Modelimizi Eğitmek"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X = iris_df.select_dtypes("float64").drop("sepal length (cm)", axis = 1)
y = iris_df["sepal length (cm)"]

X.head()

y.head()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

linear_model = LinearRegression()
ridge_model = Ridge() #L2
lasso_model = Lasso() #L1

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

print(linear_model.score(X_train, y_train))
print(ridge_model.score(X_train, y_train))
print(lasso_model.score(X_train, y_train))

"""# Metrikler"""

lin_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)

pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso": lasso_pred}

pred_dict

for key, value in pred_dict.items():
  print("Model:", key)
  print("R2 Score:", r2_score(y_test, value))
  print('Mean Absolute Error:', mean_absolute_error(y_test, value))
  print('Mean Squared Error:', mean_squared_error(y_test, value))
  print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, value)))
  print()

X.head()

"""# Custom Prediction"""

value_to_predict = [[3.7, 5, 0.9]]

lin_pred = linear_model.predict(value_to_predict)
ridge_pred = ridge_model.predict(value_to_predict)
lasso_pred = lasso_model.predict(value_to_predict)

pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso": lasso_pred}

pred_dict