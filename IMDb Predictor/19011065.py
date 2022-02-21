# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:45:46 2022

@author: elifm
"""
#Gerekli kutuphanelerin import edilmesi
import pandas as pd
import numpy as np

#veri setinin okunmasi
veriler=pd.read_csv('movie_metadata.csv')
print(veriler)

#veri setindeki bos degerlere ortalama degerlerin yazilmasi
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
 
boslar = veriler.iloc[:,2:6].values
print(boslar)
imputer = imputer.fit(boslar[:,0:4])
boslar[:,0:4] = imputer.transform(boslar[:,0:4]) 
print(boslar)

imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar2 = veriler.iloc[:,7:9].values
print(boslar2)
imputer2 = imputer2.fit(boslar2[:,0:2])
boslar2[:,0:2]=imputer2.transform(boslar2[:,0:2])
print(boslar2)

imputer3 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar3 = veriler.iloc[:,12:14].values
print(boslar3)
imputer3 = imputer3.fit(boslar3[:,0:2])
boslar3[:,0:2]=imputer3.transform(boslar3[:,0:2])
print(boslar3)

imputer4 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar4 = veriler.iloc[:,18:19].values
print(boslar4)
imputer4 = imputer4.fit(boslar4[:,0:1])
boslar4[:,0:1]=imputer4.transform(boslar4[:,0:1])
print(boslar4)

imputer5 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar5 = veriler.iloc[:,22:23].values
print(boslar5)
imputer5 = imputer5.fit(boslar5[:,0:1])
boslar5[:,0:1]=imputer5.transform(boslar5[:,0:1])
print(boslar5)

imputer6 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar6 = veriler.iloc[:,24:25].values
print(boslar6)
imputer6 = imputer6.fit(boslar6[:,0:1])
boslar6[:,0:1]=imputer6.transform(boslar6[:,0:1])
print(boslar6)

imputer7 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar7 = veriler.iloc[:,25:26].values
print(boslar7)
imputer7 = imputer7.fit(boslar7[:,0:1])
boslar7[:,0:1]=imputer7.transform(boslar7[:,0:1])
print(boslar7)

imputer8 = SimpleImputer(missing_values=np.nan, strategy='mean')
boslar8 = veriler.iloc[:,26:28].values
print(boslar8)
imputer8 = imputer8.fit(boslar8[:,0:2])
boslar8[:,0:2]=imputer8.transform(boslar8[:,0:2])
print(boslar8)

#sutunlarin DataFrame'e cevrilmesi
sonuc=pd.DataFrame(data=boslar, index=range(5043), columns=['critics','duration','dirFbLikes','a3fbLikes'])
print(sonuc)

sonuc2=pd.DataFrame(data=boslar2, index=range(5043), columns=['a1fbLikes','gross'])
print(sonuc2)

sonuc3=pd.DataFrame(data=boslar3, index=range(5043), columns=['votedUsers','castFbLikes'])
print(sonuc3)

sonuc4=pd.DataFrame(data=boslar4, index=range(5043), columns=['userReviews'])
print(sonuc4)

sonuc5=pd.DataFrame(data=boslar5, index=range(5043), columns=['budget'])
print(sonuc5)

sonuc6=pd.DataFrame(data=boslar6, index=range(5043), columns=['a2fbLikes'])
print(sonuc6)

sonuc7=pd.DataFrame(data=boslar7, index=range(5043), columns=['imdb'])
print(sonuc7)

sonuc8=pd.DataFrame(data=boslar8, index=range(5043), columns=['aspectratio','movieFbLikes'])
print(sonuc8)

#DataFrame'lerin bagimsiz degiskenleri elde etmek icin birlestirilmesi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)
s=pd.concat([s,sonuc3], axis=1)
print(s)
s=pd.concat([s,sonuc4], axis=1)
print(s)
s=pd.concat([s,sonuc5], axis=1)
print(s)
s=pd.concat([s,sonuc6], axis=1)
print(s)
s=pd.concat([s,sonuc8], axis=1)
print(s)

#veri setinin train ve test olarak ikiye bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc7,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

#MLR ile tahmin yapilmasi
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

tahmin=lr.predict(X_test)

tahmin=sc.inverse_transform(tahmin)

others=s
imdb=sonuc7

#verilere 1'lerden olusan constant degerlerin eklenmesi
import statsmodels.api as sm
X = np.append(arr = np.ones((5043,1)).astype(int), values=others, axis=1)

#her bir bagimsiz degiskenin p-value'sunun gorulmesi
X_l = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(imdb,X_l).fit()
print(model.summary())

X_l = X[:,[0,1,2,3,4,5,6,7,8,9,11,12,13]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(imdb,X_l).fit()
print(model.summary())

#R-squared degerinin hesaplanmasi
import sklearn.metrics as metrics
r2 = metrics.r2_score(y_test, tahmin)

print("Results of sklearn.metrics:")

print("R-Squared:", r2)