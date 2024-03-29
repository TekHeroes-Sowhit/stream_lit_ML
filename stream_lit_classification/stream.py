import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
st.title("Classification Instance")

st.write("""
         # Explore different classifier
         """)
dataset_name=st.sidebar.selectbox("select Dataset",("Iris","Breast Cancer","Wine Dataset"))
classifier_name=st.sidebar.selectbox("Select classifier",("KNN","SVM","Random Forest"))


def get_dataset(dataset_name):
    if dataset_name=="iris":
        data=dataset_name.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y

x,y=get_dataset(dataset_name)
st.write("Shape is",x.shape)
st.write()

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        k=st.sidebar.slider("K",1,15)
        params["C"]=C
    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
    return params

params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
        clf=SVC(c=params["C"])
    else:
        clf=RandomForestClassifier()
        return clf

clf=get_classifier(classifier_name,params)

## Classification
x_train,x_test,y_train,y_test=train_test_split(x,y)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f" accuracy={accuracy}")
#PLOT
pca=PCA(2)
x_projected=pca.fit_transform(x)
x1=x_projected[:,0]
x2=x_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("PCA1")
plt.xlabel("PCA2")
## plt.show
st.pyplot()
