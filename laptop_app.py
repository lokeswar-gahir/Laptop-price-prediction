import streamlit as st
import pickle
import sklearn
import numpy as np

pipe = pickle.load(open("pipe.pkl","rb"))
df = pickle.load(open("df.pkl","rb"))

st.title("Laptop Price Predictor")
st.write("Enter the following form to get the predicted price of your configured personal computer :)")

# brand
company = st.selectbox("Brand",df["Company"].unique())

# type of laptop
type = st.selectbox("Type", df["TypeName"].unique())

# RAM
ram = st.selectbox("RAM(in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input("Weight")

# TouchScreen
touchscreen = st.selectbox("Touchscreen",["No","Yes"])

# IPS
ips = st.selectbox("IPS",["No","Yes"])

# Screen size
screen_size = st.number_input("Screen Size(in inches)")

# Screen resolution
resolution = st.selectbox("Screen Resolution",["1920x1080","1366x768","1600x900","3840x2160","3200x1800","2880x1800","2560x1600","2560x1440","2304x1440"])

# CPU
cpu = st.selectbox("CPU",df["Cpu brand"].unique())

# HDD
hdd = st.selectbox("HDD(in GB)",[0, 32, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox("SSD(in GB)",[0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1024])

# GPU brand
gpu = st.selectbox("GPU",df["Gpu brand"].unique())

# os
os = st.selectbox("OS",df["os"].unique())

if st.button('Pridict Price'):
    st.title("Result:\n")
    #query
    try:
        ppi=None

        if touchscreen=="Yes":
            touchscreen=1
        else:
            touchscreen=0
        
        if ips=="Yes":
            ips=1
        else:
            ips=0

        res_x = int(resolution.split("x")[0])
        res_y = int(resolution.split("x")[1])
        ppi = ((res_x**2)+(res_y**2))**0.5/screen_size
        query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
        query = query.reshape(1,12)
        y_pred = pipe.predict(query)
        output = np.exp(y_pred)
        st.title(f"The price of the configured laptop is: {round(output[0],2)}.")
    except Exception as e:
        if screen_size == 0:
            st.write("!!! Screen size can not be 0.")
        if weight == 0:
            st.write("!!! Please Enter the weight.")
        st.write("ERROR: ",e)
