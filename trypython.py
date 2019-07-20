import keras 
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
dataset =pd.read_csv("HTRU_2.csv")
dataset.head()

X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]
scaler= StandardScaler()   #Initialisation  to zero
X=scaler.fit(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)		

model= Sequential()
model.add(Dense(4,input_dim=8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,input_dim=8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=50)

# score =model.evaluate(X_train, y_train)
# y_pred=model.predict(X_test)
# y_pred =(y_pred>0.5)