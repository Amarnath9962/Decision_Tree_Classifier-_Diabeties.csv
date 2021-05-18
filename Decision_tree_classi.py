import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
df.fillna(method='bfill',inplace = True)

x = df.drop(['Outcome'],axis = 1)
y = df['Outcome'].values

from sklearn.preprocessing import StandardScaler # Because its having a Numerical data

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier(criterion='entropy',max_depth= None)
Tree.fit(X_train,Y_train)

Y_pred = Tree.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(round(accuracy_score(Y_test,Y_pred),2)*100)
print(classification_report(Y_test,Y_pred))
# New data Prediction In the Decision Tree Classifier

# import numpy as np
New_1 = list([[6,148,72,35,0,33.6,0.627,50],[10,120,70,25,1,35,7,0.525],[7,85,55,35,12,56,0.254,85]])
for x,data in enumerate(New_1):
    New_d = np.array(data).reshape(1,-1)
    Y_New=Tree.predict(New_d)
    # print(Y_New)
    if Y_New == 1:
        print('Condition ',x+1,end=' ')
        print("Yes You hav a desease")
    else:
        print('Condition ', x + 1, end=' ')
        print("Dont worry you dont have desease")