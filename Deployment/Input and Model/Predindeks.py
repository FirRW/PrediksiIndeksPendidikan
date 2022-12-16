import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Title
st.header("Prediksi Indeks Pendidikan di Jawa Barat")


col1, col2, col3, col4 = st.columns([1,0.75,1.5,1.5])

# Dropdown input
with col1:
    status_kegiatan_buta_aksara= st.slider("Kegiatan Pemberantasan Buta Aksara (%)", 0,100,25)
    ketersediaan_pkbm = st.number_input("Jumlah Ketersediaan Pusat Kegiatan Belajar",step=1,min_value=0)
    ketersediaan_tbm = st.number_input("Jumlah Perpustakaan",step=1,min_value=0)
    jumlah_angkatan_kerja = st.number_input("Jumlah Angkatan Kerja ",step=1,min_value=0)
    kepadatan_penduduk = st.number_input("Total Kepadatan Penduduk",step=1,min_value=0)

with col2:
    Jumlah_SD = st.number_input("Jumlah SMD",step=1,min_value=0)
    Jumlah_SMP = st.number_input("Jumlah SMP",step=1,min_value=0)
    Jumlah_SMA = st.number_input("Jumlah SMA",step=1,min_value=0)
    Jumlah_SMK= st.number_input("Jumlah SMK",step=1,min_value=0)

with col3:
    Jumlah_Guru_SD = st.number_input("Jumlah Guru SD",step=1,min_value=0)
    Jumlah_Guru_SMP = st.number_input("Jumlah Guru SMP",step=1,min_value=0)
    Jumlah_Guru_SMA = st.number_input("Jumlah Guru SMA",step=1,min_value=0)
    Jumlah_Guru_SMK = st.number_input("Jumlah Guru SMK",step=1,min_value=0)

with col4:
    waktu_tempuh_sd_terdekat = st.number_input("Waktu Tempuh SD Terdekat(dalam menit)",step=0.1,min_value=0.0)
    waktu_tempuh_smp_terdekat = st.number_input("Waktu Tempuh SMP Terdekat(dalam menit)",step=0.1,min_value=0.0)
    waktu_tempuh_sma_terdekat = st.number_input("Waktu Tempuh SMA Terdekat(dalam menit)",step=0.1,min_value=0.0)
    Garis_Kemiskinan = st.number_input("Garis Kemiskinan Menurut Kab/Kota (Rupiah/Perkapita/Perbulan)",step=1,min_value=0)


# If button is pressed
if st.button("Prediksi Indeks Pendidikan Berdasarkan data Anda"):
    
    # Unpickle model
    xgb_cv_model= pickle.load(open("xgb_cv.pkl", 'rb'))
    scaler_fitur= pickle.load(open("scaler_fitur.pkl", 'rb'))
    scaler_target= pickle.load(open("scaler_target.pkl", 'rb'))

    # Store inputs into dataframe
    df_input = pd.DataFrame([[status_kegiatan_buta_aksara, ketersediaan_pkbm, ketersediaan_tbm,jumlah_angkatan_kerja,kepadatan_penduduk,Jumlah_SD,Jumlah_SMP,Jumlah_SMA,Jumlah_SMK,
                            Jumlah_Guru_SD,Jumlah_Guru_SMP,Jumlah_Guru_SMA,Jumlah_Guru_SMK,waktu_tempuh_sd_terdekat,waktu_tempuh_smp_terdekat,waktu_tempuh_sma_terdekat,Garis_Kemiskinan]], 
                     columns = ['status_kegiatan_buta_aksara', 'ketersediaan_pkbm','ketersediaan_tbm', 'jumlah_angkatan_kerja',
                     'kepadatan_penduduk','Jumlah SD', 'Jumlah SMP', 'Jumlah SMA', 'Jumlah SMK', 'Jumlah Guru SD', 'Jumlah Guru SMP', 'Jumlah Guru SMA', 'Jumlah Guru SMK',
                      'waktu_tempuh_sd_terdekat', 'waktu_tempuh_smp_terdekat', 'waktu_tempuh_sma_terdekat', 'Garis Kemiskinan Menurut Kab/Kota (Rupiah/Perkapita/Perbulan)']
                     )

    def feature_engineering_features (df_selected):

        df_selected_feature=df_selected

        features= ['status_kegiatan_buta_aksara',
                    'ketersediaan_pkbm',
                    'ketersediaan_tbm',
                    'jumlah_angkatan_kerja',
                    'kepadatan_penduduk',
                    'Jumlah SD',
                    'Jumlah SMP',
                    'Jumlah SMA',
                    'Jumlah SMK',
                    'Jumlah Guru SD',
                    'Jumlah Guru SMP',
                    'Jumlah Guru SMA',
                    'Jumlah Guru SMK',
                    'waktu_tempuh_sd_terdekat',
                    'waktu_tempuh_smp_terdekat',
                    'waktu_tempuh_sma_terdekat',
                    'Garis Kemiskinan Menurut Kab/Kota (Rupiah/Perkapita/Perbulan)']

        df_selected_feature = df_selected_feature.transpose().reindex(features).transpose().astype(float)

        df_selected_feature['status_kegiatan_buta_aksara']= df_selected_feature['status_kegiatan_buta_aksara']/100
        
       # inputdata = pd.DataFrame(scaler_fitur.transform(df_selected_feature[features]),columns = features)

        return df_selected_feature

    Data_input = feature_engineering_features(df_input)

    #st.text(Data_input.transpose())
    # Get prediction
    #Prediction:
    result_prediction= xgb_cv_model.predict(Data_input)
    data_hasil =pd.DataFrame(result_prediction,columns= ['indeks_pendidikan'])
    inversed_result = scaler_target.inverse_transform(data_hasil)

    # Output prediction
    st.text(print("Hasil prediksi nilai indeks pendidikan berdasarkan kondisi anda", inversed_result))