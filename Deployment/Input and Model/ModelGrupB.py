import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.header("IVF Prediction App")

# Input bar 1
#height = st.number_input("Enter Height")
# Input bar 2
#weight = st.number_input("Enter Weight")
#st.radio

col1, col2, col3, col4, col5 = st.columns([1,0.75,1.5,1.5,1])

# Dropdown input
with col1:
    Age = st.slider("Patient Age at Treatment", 18,100,25)
    Surrogate = st.radio("Patient acting as Surrogate", ("No", "Yes"))
    pgd = st.radio("PGD", ("No", "Yes"))
    electemb = st.radio("Elective Single Embryo Transfer", ("No", "Yes"))
    Freshcyc = st.radio("Fresh Cycle", ("No", "Yes"))
    Frozencyc= st.radio("Frozen Cycle", ("No", "Yes"))

with col2:
    Typetreat = st.radio("Type of treatment - IVF or DI",("IVF","DI"))
    Donemb = st.radio("Donated embryo", ("No", "Yes"))
    EggsThawed= st.radio("Eggs Thawed", ("0", ">=1"))
    EggsMixDonor= st.radio("Eggs Mixed With Donor sperm", ("0", ">=1"))
    Eggsource = st.radio("Egg Source",("Patient","Donor"))

with col3:
    Previvf = st.number_input("Total Number of Previous IVF cycles",step=1,min_value=0)
    IVFpreg = st.number_input('Total number of previous IVF pregnancies',step=1,min_value=0)
    Prevdi = st.number_input("Total Number of Previous DI cycles",step=1,min_value=0)
    DIpreg = st.number_input('Total number of previous DI pregnancies',step=1,min_value=0)
    Typeinfertil = st.selectbox("Type of Infertility", ("Female Primary", "Female Secondary", "Male Primary", 
                                "Male Secondary","Couple Primary", "Couple Secondary"))
    Causeinfertil = st.selectbox("Cause of Infertility", ("Tubal disease","Ovulatory Disorder", "Male Factor",
                                    "Patient Unexplained","Endometriosis","Cervical factors",
                                    "Female Factors","Partner Sperm Concentration","Partner Sperm Morphology",
                                    "Partner Sperm Motility","Partner Sperm Immunological factors"))

with col4:
    Mainreas = st.selectbox("Main Reason for Producing Embroys Storing Eggs",("Treatment Now ","For Donation ","Treatment Now ,For Donation ","For Storing Embryos ","For Storing Eggs "))
    Specifictreat = st.selectbox("Specific Treatment Type",("ICI","IVF","Unknown","IUI"))
    Embthawed = st.number_input("Total Embryos Thawed",step=1,min_value=0)
    Embtrans = st.number_input("Embryos Transfered",step=1,min_value=0)
    EmbtransMicro = st.number_input("Embryos Transfered from Eggs Micro-injected", step=1,min_value=0)
    Spermfrom = st.selectbox("Sperm From",("Partner","Donor","Not Assigned","Partner & Donor"))
with col5:
    Freshcollect = st.number_input("Fresh Egss Collected",step=1,min_value=0)
    #Freshstored = st.selectbox("Fresh Eggs Stored",("0","1-50",">50"))
    Freshstored = st.number_input("Fresh Eggs Stored",step=1,min_value=0)
    Eggsmix = st.number_input("Eggs Mixed with partner sperm",step=1,min_value=0)
    Totalembryo  = st.number_input("Total Embryos Created",step=1,min_value=0)
    Egginjected  = st.number_input("Eggs Micro-injected",step=1,min_value=0)
    Embrystored  = st.number_input("Embryos Stored for use by patient",step=1,min_value=0)

# If button is pressed
if st.button("Predict Your Live Birth from our Current Program"):
    
    # Unpickle classifier
    #clf = joblib.load("clf_xgb.pkl")
    clf= pickle.load(open("clf_xgb.pkl", 'rb'))

    # Store inputs into dataframe
    df_input = pd.DataFrame([[Age, Prevdi, Previvf,IVFpreg,DIpreg,Mainreas,Donemb,Surrogate,Typetreat,Specifictreat,pgd,electemb,Eggsource,Spermfrom,Freshcyc,Frozencyc,EggsThawed,
                    Freshcollect,Freshstored,Eggsmix,EggsMixDonor,Totalembryo,Egginjected,Embthawed,Embtrans,EmbtransMicro,Embrystored,Typeinfertil,Causeinfertil]], 
                     columns = ['Patient Age at Treatment', 'Total Number of Previous IVF cycles','Total Number of Previous DI cycles', 'Total number of IVF pregnancies',
                     'Total number of DI pregnancies','Main Reason for Producing Embroys Storing Eggs', 'Donated embryo','Patient acting as Surrogate', 
                     'Type of treatment - IVF or DI','Specific treatment type', 'PGD', 'Elective Single Embryo Transfer','Egg Source', 'Sperm From', 'Fresh Cycle', 
                     'Frozen Cycle','Eggs Thawed', 'Fresh Eggs Collected', 'Fresh Eggs Stored','Eggs Mixed With Partner Sperm', 'Eggs Mixed With Donor sperm',
                     'Total Embryos Created', 'Eggs Micro-injected', 'Total Embryos Thawed','Embryos Transfered', 'Embryos Transfered from Eggs Micro-injected',
                     'Embryos Stored For Use By Patient', 'Type of Infertility','Cause of Infertility']
                     )

    def feature_engineering_features (df_selected):

        df_selected_feature=df_selected

        bins = [-1,0,1,2,3,4,5,100]
        labels = [0,1,2,3,4,5,6]
        df_selected_feature['Total Number of Previous IVF cycles'] = pd.cut(df_selected_feature['Total Number of Previous IVF cycles'], bins=bins, labels=labels,include_lowest=True)
        df_selected_feature['Total Number of Previous DI cycles'] = pd.cut(df_selected_feature['Total Number of Previous DI cycles'], bins=bins, labels=labels,include_lowest=True)

        YesNoMap = {"No":0,"Yes":1}
        df_selected_feature['Donated embryo'] = df_selected_feature['Donated embryo'].replace(YesNoMap)
        df_selected_feature['Patient acting as Surrogate'] = df_selected_feature['Patient acting as Surrogate'].replace(YesNoMap)
        df_selected_feature['PGD'] = df_selected_feature['PGD'].replace(YesNoMap)
        df_selected_feature['Elective Single Embryo Transfer'] = df_selected_feature['Elective Single Embryo Transfer'].replace(YesNoMap)
        df_selected_feature['Fresh Cycle'] = df_selected_feature['Fresh Cycle'].replace(YesNoMap)
        df_selected_feature['Frozen Cycle'] = df_selected_feature['Frozen Cycle'].replace(YesNoMap)

        ZeroOneMap = {"0":0,">=1":1}
        df_selected_feature['Eggs Thawed'] = df_selected_feature['Eggs Thawed'].replace(ZeroOneMap)
        df_selected_feature['Eggs Mixed With Donor sperm'] = df_selected_feature['Eggs Mixed With Donor sperm'].replace(ZeroOneMap)

        bins = [-1,0,1,2,3,100]
        labels = [0,1,2,3,4]
        df_selected_feature['Total Embryos Thawed'] = pd.cut(df_selected_feature['Total Embryos Thawed'], bins=bins, labels=labels,include_lowest=True)
        df_selected_feature['Embryos Transfered'] = pd.cut(df_selected_feature['Embryos Transfered'], bins=bins, labels=labels,include_lowest=True)
        df_selected_feature['Embryos Transfered from Eggs Micro-injected'] = pd.cut(df_selected_feature['Embryos Transfered from Eggs Micro-injected'], bins=bins, labels=labels,include_lowest=True)

        bins = [18,34,37,39,42,44,50,100]
        labels = [0,1,2,3,4,5,6]
        df_selected_feature['Patient Age at Treatment'] = pd.cut(df_selected_feature['Patient Age at Treatment'], bins=bins, labels=labels,include_lowest=True)

        TMPESmap = {'Treatment Now ':0,'For Donation ':1,'Treatment Now ,For Donation ':2,'For Storing Embryos ':3,'For Storing Eggs ':4}
        df_selected_feature['Main Reason for Producing Embroys Storing Eggs'] =df_selected_feature['Main Reason for Producing Embroys Storing Eggs'].replace(TMPESmap)

        Type= {"Female Primary":1, "Female Secondary":2, "Male Primary":3, "Male Secondary":4,"Couple Primary":5, "Couple Secondary":6}
        df_selected_feature['Type of Infertility'] =df_selected_feature['Type of Infertility'].replace(Type)

        Cause= { "Tubal disease":1,"Ovulatory Disorder":2, "Male Factor":3,"Patient Unexplained":4,"Endometriosis":5,"Cervical factors":6,
            "Female Factors":7,"Partner Sperm Concentration":8,"Partner Sperm Morphology":9,"Partner Sperm Motility":10,"Partner Sperm Immunological factors":11}
        df_selected_feature['Cause of Infertility'] =df_selected_feature['Cause of Infertility'].replace(Cause)

        #Treat Type of treatment - IVF or DI
        TTIODmap = {'IVF':1,'DI':2}
        df_selected_feature['Type of treatment - IVF or DI'] =df_selected_feature['Type of treatment - IVF or DI'].replace(TTIODmap)

        #Treat Specific treatment type
        TSTTmap = {'ICI':1, 'IVF':2, 'Unknown':3, 'IUI':4}
        df_selected_feature['Specific treatment type'] = df_selected_feature['Specific treatment type'].replace(TSTTmap)

        # Treat Egg source 
        TESmap = {'Patient':0,'Donor':1}
        df_selected_feature['Egg Source'] = df_selected_feature['Egg Source'].replace(TESmap)

        #Treat sperm from
        TSFmap = {'Partner':0,'Donor':1,'not assigned':2,'Partner & Donor':3}
        df_selected_feature['Sperm From'] = df_selected_feature['Sperm From'].replace(TSFmap)

        bins = [-1,0,10,20,50,np.inf]
        labels = [0,1,2,3,4]
        #Treat Fresh Eggs Collected
        df_selected_feature['Fresh Eggs Collected'] = pd.cut(df_selected_feature['Fresh Eggs Collected'], bins=bins, labels=labels,include_lowest=True)
         #Treat Eggs Mixed With Partner Sperm
        df_selected_feature['Eggs Mixed With Partner Sperm'] = pd.cut(df_selected_feature['Eggs Mixed With Partner Sperm'], bins=bins, labels=labels,include_lowest=True)
        #Treat Eggs Micro-injected
        df_selected_feature['Eggs Micro-injected'] = pd.cut(df_selected_feature['Eggs Micro-injected'], bins=bins, labels=labels,include_lowest=True)
        

        #Treat Fresh Eggs Stored
        bins = [-1,0,50,np.inf]
        labels = [0,1,2]
        df_selected_feature['Fresh Eggs Stored']  = pd.cut(df_selected_feature['Fresh Eggs Stored'] , bins=bins, labels=labels,include_lowest=True)

        #Treat Total Embryos Created
        bins = [-1,0,5,10,20,np.inf]
        labels = [0,1,2,3,4]
        df_selected_feature['Total Embryos Created'] = pd.cut(df_selected_feature['Total Embryos Created'] , bins=bins, labels=labels,include_lowest=True)

        #Treat Embryos Stored For Use By Patient
        bins = [-1,0,10,50,np.inf]
        labels = [0,1,2,3]
        df_selected_feature['Embryos Stored For Use By Patient'] = pd.cut(df_selected_feature['Embryos Stored For Use By Patient'] , bins=bins, labels=labels,include_lowest=True)

        return df_selected_feature

    Data_input = feature_engineering_features(df_input)
    
    #X = X.replace(["Brown", "Blue"], [1, 0])
    features= ['Patient Age at Treatment','Total Number of Previous IVF cycles', 'Total Number of Previous DI cycles', 'Total number of IVF pregnancies',
            'Total number of DI pregnancies','Main Reason for Producing Embroys Storing Eggs','Donated embryo','Patient acting as Surrogate', 'Type of treatment - IVF or DI',
            'Specific treatment type', 'PGD', 'Elective Single Embryo Transfer', 'Egg Source', 'Sperm From', 'Fresh Cycle', 'Frozen Cycle', 'Eggs Thawed', 'Fresh Eggs Collected',
            'Fresh Eggs Stored', 'Eggs Mixed With Partner Sperm', 'Eggs Mixed With Donor sperm', 'Total Embryos Created', 'Eggs Micro-injected', 'Total Embryos Thawed', 'Embryos Transfered',
            'Embryos Transfered from Eggs Micro-injected', 'Embryos Stored For Use By Patient', 'Type of Infertility', 'Cause of Infertility']
    #input_data = Data_input.transpose().reindex(features).transpose().astype(float)
    input_data = Data_input.transpose().reindex(features).transpose().astype(int)
    # Check untuk data input anda
    # st.text(input_data.transpose())

    # Get prediction
    prediction = clf.predict(input_data)
    # Output prediction

    #st.text(f"This instance is a {prediction}")
    if prediction == 1:
        st.text("Prediction Result: Successful")
    else:
        st.text("Prediction Result: Failed")