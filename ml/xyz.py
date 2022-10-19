import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

pima=pd.read_csv(r"C:\Users\Node-2\Downloads\data.csv")
X=np.array(pima.iloc[:,:-1])
y=np.array(pima.iloc[:,-1])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_sate=1)
clf=MLPClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy",metrics.accuracy_score(y_test,y_pred))

        
