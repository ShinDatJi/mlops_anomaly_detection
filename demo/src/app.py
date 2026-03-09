import streamlit as st
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
import os
import prediction as prediction
# import prediction

st.set_page_config(layout="wide")
st.title("MVTec Anomaly Detection - MLOps")
st.sidebar.title("Table of contents")
pages=["Welcome", "Introduction", "Demo", "Architecture", "Conclusions"]
page=st.sidebar.radio("Go to", pages)

if page=="Welcome":
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.write("MVTec AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad")
            st.write("Project on GitHub: https://github.com/ShinDatJi/mlops_anomaly_detection")
        with cols[1]:
            with st.container(horizontal_alignment="right"):
                st.text("Fernando Sotres")
                st.text("Isaak Gerber")

        st.image("./demo/figures/mvtec_header.webp", caption="src: https://www.mvtec.com/company/research/datasets/mvtec-ad")

elif page=="Introduction":
    st.header("Introduction")
    with st.container():
        st.image("./demo/figures/introduction.png")

elif page == "Demo":
    st.header("Demo")
    with st.container():
        sel_cat = st.selectbox("Select category", ["bottle", "capsule", "grid", "toothbrush", "zipper"])
        sel_ver = st.text_input("Select version", "pretrained")
        defective = st.checkbox("Use defective image", True)
        predict = st.button("Predict random")
        if predict:
            fig = prediction.predict_random(sel_cat, sel_ver, defective)
            st.pyplot(fig)

elif page=="Architecture":
    st.header("Architecture")
    with st.container():
        st.image("./demo/figures/architecture.png")
    
elif page=="Conclusions":
    st.header("Conclusions")
    st.divider()
    st.subheader("Improvements")
    st.write("User management API backed by user database")
    st.write("Data API for uploading training data by customers")
    st.write("Unit, integration, and e2e tests")
    st.write("CI/CD pipelines for testing and deployment")
    st.write("nginx for load balancing and scaling")
    st.write("User management and roles for used tools")
    st.write("Package model and patching with e.g. bentoml")
