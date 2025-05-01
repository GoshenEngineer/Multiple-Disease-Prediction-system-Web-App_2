# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:31:32 2025

@author: Admin
"""
#importing the dependencies
import numpy as np
#loading the saved model
import pickle
#deploying the model
import streamlit as st
#importing streamlit option menu
from streamlit_option_menu import option_menu


#loading the saved models
#for the breast_cancer model

breast_cancer_model = pickle.load(open('Breast_Cancer_model.sav', 'rb'))
#loading the saved scaler for the breast cancer model
breast_cancer_scaler = pickle.load(open('Breast_Cancer_scaler.sav', 'rb'))
#for hepatitis model
hepatitis_disease_model = pickle.load(open('Hepatitis_Disease_model.sav', 'rb'))
#loading the saved scaler for the breast cancer model
hepatitis_scaler = pickle.load(open('scaler_Hepatitis_Disease.sav', 'rb'))
#for the parkinsons model
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))


#sidebar for navigate
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System using ML',
                           
                           
                           ['Breast Cancer Prediction', 'Hepatitis Prediction',
                            'Parkinsons Prediction'],
                               icons=["hospital", "lungs", "brain-circuit"],  # New icons
                                default_index=0)
      
#Breast Cancer Prediction Page
if (selected == 'Breast Cancer Prediction'):
    
    #page title
    st.title('Breast Cancer Prediction using ML')
    
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean = st.text_input('Radius Mean')
        smoothness_mean = st.text_input('Smoothness Mean')
        concavity_mean = st.text_input('Concavity Mean')
        symmetry_mean = st.text_input('Symmetry Mean')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')
    
    with col2:
        texture_mean = st.text_input('Texture Mean')
        compactness_mean = st.text_input('Compactness Mean')
        concave_points_mean = st.text_input('Concave Points Mean')
        radius_se = st.text_input('Radius SE')
        texture_se = st.text_input('Texture SE')
    
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean')
        area_mean = st.text_input('Area Mean')
        perimeter_se = st.text_input('Perimeter SE')
        area_se = st.text_input('Area SE')
        smoothness_se = st.text_input('Smoothness SE')
    
    with col1:
        compactness_se = st.text_input('Compactness SE')
        concavity_se = st.text_input('Concavity SE')
        concave_points_se = st.text_input('Concave Points SE')
        symmetry_se = st.text_input('Symmetry SE')
        fractal_dimension_se = st.text_input('Fractal Dimension SE')
    
    with col2:
        radius_worst = st.text_input('Radius Worst')
        texture_worst = st.text_input('Texture Worst')
        perimeter_worst = st.text_input('Perimeter Worst')
        area_worst = st.text_input('Area Worst')
        smoothness_worst = st.text_input('Smoothness Worst')
    
    with col3:
        compactness_worst = st.text_input('Compactness Worst')
        concavity_worst = st.text_input('Concavity Worst')
        concave_points_worst = st.text_input('Concave Points Worst')
        symmetry_worst = st.text_input('Symmetry Worst')
        fractal_dimension_worst = st.text_input('Fractal Dimension Worst')

    
    # code for prediction
    breast_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Breast Cancer Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array =np.asarray([radius_mean, texture_mean, perimeter_mean,
           area_mean, smoothness_mean, compactness_mean, concavity_mean,
           concave_points_mean, symmetry_mean, fractal_dimension_mean,
           radius_se, texture_se, perimeter_se, area_se, smoothness_se,
           compactness_se, concavity_se, concave_points_se, symmetry_se,
           fractal_dimension_se, radius_worst, texture_worst,
           perimeter_worst, area_worst, smoothness_worst,
           compactness_worst, concavity_worst, concave_points_worst,
           symmetry_worst, fractal_dimension_worst])
        
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)
        input_data_scaled =  breast_cancer_scaler.transform(input_data_reshaped)
        breast_prediction =  breast_cancer_model.predict(input_data_scaled)
        if(breast_prediction[0]== 0):
           breast_diagnosis = 'You don\'t have Breast Cancer'
        else:
           breast_diagnosis = 'You have Breast Cancer'
    st.success(breast_diagnosis)
    
    
if (selected == 'Hepatitis Prediction'):
    
    #page title
    st.title('Hepatitis Disease Prediction using ML')
  
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    # General Patient Information
    with col1:
        age = st.text_input('Age')
        sex = st.text_input('Sex')
        steroid = st.text_input('Steroid Use (1 = no; 2 = yes)')
        antivirals = st.text_input('Antiviral Treatment (1 = no; 0 = yes)')
        fatigue = st.text_input('Fatigue (1 = no; 2 = yes)')
        malaise = st.text_input('Malaise (1 = no; 2 = yes)')
    # Symptoms & Clinical Findings
    with col2:
        anorexia = st.text_input('Anorexia (1 = no; 2 = yes)')
        liver_Big = st.text_input('Liver Big (1 = no; 2 = yes)')
        liver_Firm = st.text_input('Liver Firm (1 = no; 2 = yes)')
        spleen_Palpable = st.text_input('Spleen Palpable (1 = no; 2 = yes)')
        spiders = st.text_input('Spiders (1 = no; 2 = yes)')
        ascites = st.text_input('Ascites (1 = no; 2 = yes)')   
    # Lab Tests & Advanced Clinical Markers
    with col3:
        varices = st.text_input('Varices (1 = no; 2 = yes)')
        bilirubin = st.text_input('Bilirubin Level')
        alk_Phosphate = st.text_input('Alkaline Phosphate Level')
        SGOT = st.text_input('Serum Glutamic-Oxaloacetic Transaminase (SGOT) Level')
        albumin = st.text_input('Albumin Level')
        protime = st.text_input('Prothrombin Time')
        histology = st.text_input('Histology Exam Result (1 = no; 2 = yes)')

        
    # code for prediction
    hepat_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Hepatitis Disease Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array = np.asarray([age, sex, steroid, antivirals, fatigue, malaise, anorexia,
            liver_Big, liver_Firm, spleen_Palpable, spiders, ascites, varices,
            bilirubin, alk_Phosphate, SGOT, albumin, protime, histology])
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)
        input_data_scaled = hepatitis_scaler.transform(input_data_reshaped)
        hepatitis_prediction =  hepatitis_disease_model.predict(input_data_scaled)
        if(hepatitis_prediction[0]== 1):
           hepat_diagnosis = 'The person "Won\'t survive'
        else:
           hepat_diagnosis = 'The person will survive'
    st.success(hepat_diagnosis)
    
    
    
if (selected == 'Parkinsons Prediction'):
    #page title
    st.title('Parkinsons Prediction using ML')
    
    
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo = st.text_input('Average vocal fundamental frequency')
    with col2:
        MDVP_Fhi = st.text_input('Maximum vocal fundamental frequency')
    with col3:    
        MDVP_Flo = st.text_input('Minimum vocal fundamental frequency')
    with col1:    
        MDVP_Jitter_ = st.text_input(' The percentage variation in the fundamental frequency, indicating voice instability.')
    with col2:    
        MDVP_RAP =st.text_input('Relative Average Perturbation, another measure of frequency variation')
    with col3:    
        MDVP_PPQ = st.text_input('Pitch Period Perturbation Quotient')
    with col1:    
        Jitter_DDP  = st.text_input('Jitter (Difference of Differences)')
    with col2:    
        MDVP_Shimmer = st.text_input('Variation in amplitude (loudness) of the voice')
    with col3:
        MDVP_Shimmer_dB = st.text_input('Shimmer in decibels, another measure of amplitude variation')   
    with col1:
        Shimmer_DDA = st.text_input('A measure of amplitude variation over dynamic time')
    with col2:
        HNR = st.text_input('Harmonics-to-Noise Ratio')
    with col3:
        RPDE = st.text_input('Recurrence-based Plotting of Dynamic Entropy')
    with col1:
        DFE = st.text_input('The Detrended Fluctuation Exponent (DFE)')
    with col2:
        spread1 = st.text_input('Fundamental Frequency Variation 1')
    with col3:
        spread2 = st.text_input('Fundamental Frequency Variation 2')
    with col1:
        D2 = st.text_input('Fractal Dimension')
    with col2:
        PPE = st.text_input('Peak-to-Peak Entropy')

    # code for prediction
    park_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Parkinson Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array =np.asarray([MDVP_Fo,MDVP_Fhi, MDVP_Flo ,MDVP_Jitter_  ,MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                                         MDVP_Shimmer_dB, Shimmer_DDA, HNR, RPDE, DFE, spread1, spread2, D2,PPE])
        
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)

        parkinson_prediction =  parkinsons_model.predict(input_data_reshaped)
        if(parkinson_prediction[0]== 0):
           park_diagnosis = 'The person is not Parkinson Disease'
        else:
           park_diagnosis = 'The person is having Parkinson Disease'
    st.success(park_diagnosis)      