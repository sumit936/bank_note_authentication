import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.proto.PlotlyChart_pb2 import Figure
from plotly import graph_objects as go
import seaborn as sns
import pickle
import time
sns.set()
st.set_option('deprecation.showPyplotGlobalUse', False)


def home():
        if st.checkbox('show data'):
            st.dataframe(data)
        
        st.markdown(
            '''
            ### Histogram
            '''
        )
        col = st.selectbox('choose column to plot',['variance','skewness','curtosis','entropy'])
        layout = go.Layout(
            title = go.layout.Title(text = col.capitalize(),)
        )

        fig = go.Figure(data= go.Histogram(x = data[col]),layout = layout)
        st.plotly_chart(fig)

        st.markdown(
            '''
            ### Line chart
            '''
        )
        st.line_chart(data.iloc[:,:-1])

        st.markdown(
            '''
            ### Box Plot
            '''
        )

        col2 = st.selectbox('choose column',['variance','skewness','curtosis','entropy'])
        plt.figure(figsize = (5,3))
        sns.boxplot(data[col2])
        plt.tight_layout()
        st.pyplot()

def predict(val):
    prediction = classifier.predict(val)
    return prediction

def prediction():
        st.header('Know your prediction')
        variance = st.text_input('Variance')
        skewness = st.text_input('Skewness')
        curtosis = st.text_input('Curtosis')
        entropy = st.text_input('Entropy')
        val = [[variance,skewness, curtosis, entropy]]
        if st.button('Predict'):
            result = predict(val)
            my_bar = st.progress(0,)
            for i in range(100):
                time.sleep(0.000001)
                my_bar.progress(i+1)
            st.success(f'The prediction is class: {result[0]}')


if __name__=='__main__':
    #Loading Classifier
    pickle_in = open('BNA_classifier.pkl','rb')
    classifier = pickle.load(pickle_in)

    html = """
    <div style="background-color:blue;padding:5px">
    <h2 style = "color:white;text-align:center;">Bank Note Aunthenticator ML App </h2>
    </div><br>
    """

    #Title of the page
    st.markdown(html, unsafe_allow_html=True)
    st.image('banknote-authentication.jpeg')

    nav = st.sidebar.radio('Navigation',['Home','Prediction'])

    data = pd.read_csv('BankNote_Authentication.csv')
    if nav=='Home':
        home()
    if nav=='Prediction':
        prediction()
    
    