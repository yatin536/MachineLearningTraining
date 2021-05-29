import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
ds=pd.read_csv("SummerTrainingML\\ML\\MultipleLinearRegression\\50_Startups.csv")
X=ds.iloc[:,:-1]
y=ds["Profit"]
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#print(X_train)
#print(y_train)
States_tr=X_train["State"]
#print(States_tr)
Labe=LabelEncoder()
states_labe=Labe.fit_transform(States_tr)
#print(states_labe)
states_labe=states_labe.reshape(-1,1)
#print(states_labe)
OHE=OneHotEncoder()
states_final=OHE.fit_transform(states_labe)
states_final=states_final.toarray()
states_final=states_final[:,0:2]
#print(states_final)
#states_final.reshape(30,4)
#print(states_final,shape)

X_train=X_train.iloc[:,:-1]
#print(X_train.shape)
X_train=np.hstack((X_train,states_final))

#print(X_train)


mind=LinearRegression()
mind.fit(X_train,y_train)
print(mind.coef_)
