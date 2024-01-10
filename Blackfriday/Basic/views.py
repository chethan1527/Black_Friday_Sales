from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def Prog(request):
    if request.method == "POST":
        data = request.POST
        algo = data.get('alg')
        t1 = data.get('t1')
        t2 = data.get('t2')
        t3 = data.get('t3')
        t4 = data.get('t4')
        t5 = data.get('t5')
        t6 = data.get('t6')
        t7 = data.get('t7')
        t8 = data.get('t8')
        t9 = data.get('t9')
        if algo == '0':  # Check for '0', which corresponds to BDtree
            result, RMSE = BDtree(t1, t2, t3, t4, t5, t6, t7, t8, t9)
        elif algo == '1':  # Check for '1', which corresponds to BRandom
            result, RMSE = BRandom(t1, t2, t3, t4, t5, t6, t7, t8, t9)
        else:
            result, RMSE = Blinear(t1, t2, t3, t4, t5, t6, t7, t8, t9)
        return render(request, "Prog.html", context={'result': result, 'RMSE': RMSE})
    return render(request, "Prog.html")


def BDtree(t1,t2,t3,t4,t5,t6,t7,t8,t9):
        path="C:\\Users\\chait\\OneDrive\\Documents\\Desktop\\chethan\\internship projects\\BlackFridaySales.csv"
        data=pd.read_csv(path)
        df = data.copy()
        lr = LabelEncoder()
        df['Gender'] = lr.fit_transform(df['Gender'])
        df['Age'] = lr.fit_transform(df['Age'])
        df['City_Category'] = lr.fit_transform(df['City_Category'])
        df['Stay_In_Current_City_Years'] = lr.fit_transform(df['Stay_In_Current_City_Years'])
        df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
        df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')
        df = df.drop(["User_ID","Product_ID"],axis=1)
        X = df.drop("Purchase",'columns')
        y=df['Purchase']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        from sklearn.tree import DecisionTreeRegressor
        # create a regressor object 
        regressor = DecisionTreeRegressor(random_state = 0)  
        regressor.fit(X_train, y_train)
        dt_y_pred = regressor.predict(X_test)
        result=regressor.predict([[t1,t2,t3,t4,t5,t6,t7,t8,t9]])
        RMSE=sqrt(mean_squared_error(y_test, dt_y_pred))
        return result[0],RMSE

def BRandom(t1,t2,t3,t4,t5,t6,t7,t8,t9):
        path="C:\\Users\\chait\\OneDrive\\Documents\\Desktop\\chethan\\internship projects\\BlackFridaySales.csv"
        data=pd.read_csv(path)
        df = data.copy()
        lr = LabelEncoder()
        df['Gender'] = lr.fit_transform(df['Gender'])
        df['Age'] = lr.fit_transform(df['Age'])
        df['City_Category'] = lr.fit_transform(df['City_Category'])
        df['Stay_In_Current_City_Years'] = lr.fit_transform(df['Stay_In_Current_City_Years'])
        df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
        df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')
        df = df.drop(["User_ID","Product_ID"],axis=1)
        X = df.drop("Purchase",'columns')
        y=df['Purchase']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        from sklearn.ensemble import RandomForestRegressor
        # create a regressor object 
        RFregressor = RandomForestRegressor(random_state = 0)  
        RFregressor.fit(X_train, y_train)
        y_pred = RFregressor.predict(X_test)
        sqrt(mean_squared_error(y_test, y_pred))
        result=RFregressor.predict([[t1,t2,t3,t4,t5,t6,t7,t8,t9]])
        RMSE=sqrt(mean_squared_error(y_test, y_pred))
        return result[0],RMSE

def Blinear(t1,t2,t3,t4,t5,t6,t7,t8,t9):
        path="C:\\Users\\chait\\OneDrive\\Documents\\Desktop\\chethan\\internship projects\\BlackFridaySales.csv"
        data=pd.read_csv(path)
        df = data.copy()
        lr = LabelEncoder()
        df['Gender'] = lr.fit_transform(df['Gender'])
        df['Age'] = lr.fit_transform(df['Age'])
        df['City_Category'] = lr.fit_transform(df['City_Category'])
        df['Stay_In_Current_City_Years'] = lr.fit_transform(df['Stay_In_Current_City_Years'])
        df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
        df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')
        df = df.drop(["User_ID","Product_ID"],axis=1)
        X = df.drop("Purchase",'columns')
        y=df['Purchase']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        result=lr.predict([[t1,t2,t3,t4,t5,t6,t7,t8,t9]])
        RMSE=sqrt(mean_squared_error(y_test, y_pred))
        return result[0],RMSE


def Index(request):
    return render(request,'Index.html')
     
        
# Create your views here.
