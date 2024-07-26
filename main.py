import streamlit as st
#pip install scikit-learn
import pickle
import numpy as np

#load model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title('Laptop Price predictor')

#brand
company=st.selectbox('Brand',df['Company'].unique())

#type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

#RAM
RAM=st.selectbox('RAM(GB)',[2,4,6,8,12,16,32])

#Weight
weight=st.number_input('weight of Laptop')

#touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

#ips
ips=st.selectbox('IPS',['Yes','No'])

#screen size
screen_size=st.number_input('Screen Size')

#resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1440','2304x1440'])
#cpu
cpu=st.selectbox('CPU',df['cpu brand'].unique())

#hdd
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
#ssd
ssd=st.selectbox('SSD(in GB)',[0,128,256,512,1024,2048])
#gpu
gpu=st.selectbox('GPU',df['Gpu brand'].unique())

#os
os=st.selectbox('OS',df['os'].unique())

if st.button('Predict Laptop Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=='Yes':
        ips==1
    else:
        ips==0
    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size
    query=np.array([company,type,RAM,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query=query.reshape(1,12)
    st.title("Predicted price:-" + str(int(np.exp(pipe.predict(query)[0]))))





