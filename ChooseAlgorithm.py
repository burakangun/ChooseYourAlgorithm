import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Verileri yükleme
veriler = pd.read_csv('veriler.csv')
# Test
#print(veriler)

x = veriler.iloc[:,1:4].values # bağımsız değişkenler
y = veriler.iloc[:,4:].values # bağımlı değişkenler
# Eğer başka veri kümeleri üzerinde çalışmak istiyorsan yukarıdaki sütunları değiştirebilirsin.

# Verilerin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split # Kütüphaneyi dahil ettik

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler # Kütüphaneyi dahil ettik

sc = StandardScaler() # StandartScaler sınıfından bir obje oluşturuyoruz.

X_train = sc.fit_transform(x_train) # Modellemeyi skaladan geçirilmiş veriler üzerinden yapacağız.
X_test = sc.transform(x_test)


def wannaSee(func): # Decorator fonksiyon
    def wrapper():
        liste = [x_train,y_train,x_test,y_test]
        cevap = int(input("Train ve Test veri kümelerini görmek ister misin? (Evet için '1' Hayır için '2' yazınız."))
        if cevap==1:
            for i in liste:
                print("Train ve Test Kümeleri")
                print(i)
            func()
        elif cevap==2:
            func()
    return wrapper

@wannaSee
def lojistikRegresyon():
    from sklearn.linear_model import LogisticRegression # Kütüphaneyi dahil ettik

    logr = LogisticRegression() # LogistigRegression sınıfından bir obje oluşturuyoruz.
    logr.fit(X_train,y_train)

    y_pred = logr.predict(X_test)
    print(y_pred)

@wannaSee
def kNN():
    from sklearn.neighbors import KNeighborsClassifier # Kütüphaneyi dahil ettik
    k = int(input("k değerinin kaç olmasını istiyorsunuz ? (Kaç tane komşuya bakılsın)"))

    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski') # Diğer parametreleri denemek isterseniz -> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    knn.fit(X_train, y_train) # Modeli eğitiyoruz.

    y_pred = knn.predict(X_test)
    print("Tahmin sonucu : ", y_pred)


@wannaSee
def SVM():
    from sklearn.svm import SVC

    svc = SVC(kernel='linear') # Diğer parametreleri denemek isterseniz -> https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    svc.fit(X_train,y_train) # Modeli eğitiyoruz.

    y_pred = svc.predict(X_train, y_train)
    print("Tahmin sonucu : ", y_pred)

@wannaSee
def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    gnb.fit(X_train,y_train) # Modeli eğitiyoruz.

    y_pred = gnb.predict(X_test, y_train)
    print("Tahmin sonucu : ", y_pred)

@wannaSee
def DTC():
    from sklearn.tree import DecisionTreeClassifier

    dtc = DecisionTreeClassifier(criterion='entropy') # Diğer parametreleri denemek isterseniz -> https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    dtc.fit(X_train,y_train) # Modeli eğitiyoruz.

    y_pred = dtc.predict(X_test)
    print("Tahmin sonucu : ", y_pred)

@wannaSee
def RFC():
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
    rfc.fit(X_train,y_train)

    y_pred = rfc.predict(X_test)
    print("Tahmin sonucu : ", y_pred)



algoritma = int(input("""
Hangi Sınıflandırma Algoritmasını Kullanmak İstiyorsunuz ?
1- Lojistik Regresyon
2- K-NN (k-Nearest Neighbor)
3- SVR (Support Vector Regression)
4- Naive Bayes
5- DTC(Decision Tree Classifier)-Karar Ağacı
6- Rassal Orman

Çıkmak için '0'(sıfır)  basınız !
"""))

while(True):
    if algoritma==1:
        lojistikRegresyon()
    elif algoritma==2:
        kNN()
    elif algoritma==3:
        SVM()
    elif algoritma==4:
        NaiveBayes()
    elif algoritma==5:
        DTC()
    elif algoritma==6:
        RFC()
    elif algoritma==0:
        break
    else:
        print("Yanlış bir değer girdiniz lütfen tekrar deneyin !")
        continue

