from sklearn import preprocessing 
import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib 


kmeans_model = joblib.load("Customer_Segmentation.sav")

df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Customer Segmentation")
st.header("Input Customer Data")


st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

with st.form("my_form"):
        
    customer_income = st.number_input(label='Annual Income (k$)', min_value=0, max_value=200) #value=50)
    customer_spending = st.number_input(label='Spending Score (1-100)', min_value=0, max_value=100) #value=50)

    data = [[customer_income, customer_spending]]
    submitted = st.form_submit_button("Predict")


if submitted:
         
        clust=kmeans_model.predict(data)[0]
        print('Data Belongs to Cluster',clust)

        cluster_df=df[df['Cluster']==clust]

        plt.rcParams["figure.figsize"] = (20,3)
        for c in cluster_df.drop(['Cluster'],axis=1):
           fig, ax = plt.subplots()
           grid = sns.histplot(cluster_df, x=c, kde=True)
           plt.title(f'Histogram of {c} for cluster{clust}')
           #plt.show()
           st.pyplot(fig)
