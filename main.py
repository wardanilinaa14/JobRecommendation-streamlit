import json
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# List Unique Category
with open('kategori_list.json','r') as files:
    data_json = json.load(files)
    locations = data_json.get('lokasi')
    states = data_json.get('negara_bagian')
    countries = data_json.get('negara')
    cities = data_json.get('kota')
    idpos = data_json.get('kode_pos')
    names = data_json.get('nama_kemampuan_lowongan')
    specializations = data_json.get('spesialisasi')
    industries = data_json.get('nama_industri_lowongan')
    salaries = data_json.get('range_gaji')
    benefits = data_json.get('deskripsi_benefit')

# Import Final Data
df = pd.read_csv('final_dataaa.csv')
used_cols = [col for col in df.columns if col not in ['id_pekerjaan','url']]
df['combined_features'] = df[used_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

def recommend_jobs(user_input, df, tfidf_matrix):
    # Transform user input menjadi vektor TF-IDF
    user_tfidf = vectorizer.transform([user_input])

    # Hitung cosine similarity antara input pengguna dan semua pekerjaan
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Dapatkan indeks pekerjaan dengan similarity tertinggi
    similar_indices = cosine_similarities.argsort()[-10:][::-1]  # ambil 10 pekerjaan teratas

    # Ambil pekerjaan yang sesuai berdasarkan indeks
    recommendations = df.iloc[similar_indices]

    # Tambahkan kolom cosine similarity ke DataFrame rekomendasi
    recommendations['cosine_similarity'] = cosine_similarities[similar_indices]
    return recommendations

def main():
    st.set_page_config(page_title='Model Deployment 7B', layout='wide')
    # Display logo and title
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg"
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="{logo_url}" width="50" style="margin-right: 10px;">
        <h1>Job Recommendation System</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write('Please Input Your Data :')
    
    selected_location = st.selectbox('Location',locations)
    selected_states = st.selectbox('State',states)
    selected_countries = st.selectbox('Country',countries)
    selected_cities = st.selectbox('City',cities)
    selected_idpos = st.selectbox('Postal Code',idpos)
    selected_names = st.selectbox('Position',names)
    selected_specializations = st.selectbox('Specialization',specializations)
    selected_industries = st.selectbox('Industry',industries)
    selected_salaries = st.selectbox('Salary',salaries)
    selected_benefits = st.selectbox('Benefit',benefits)
    
    user_data = pd.DataFrame({
        'lokasi' : [selected_location], 'negara_bagian' : [selected_states],'negara' : [selected_countries],'kota' : [selected_cities], 'kode_pos':[selected_idpos],
        'nama_kemampuan_lowongan' : [selected_names], 'spesialisasi' : [selected_specializations],'nama_industri_lowongan' : [selected_industries],
        'range_gaji' : [selected_salaries], 'deskripsi_benefit' : [selected_benefits]
    })
    user_data['combined_features'] = user_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    if st.button('Find me jobs:'):
        # run recommend_jobs with user_data
        # ex :
        recommendations = recommend_jobs(str(user_data['combined_features'].values), df, tfidf_matrix)
        recommendations = recommendations.drop(['id_pekerjaan', 'combined_features'], axis=1)
        # change the columns name to be more pretty
        # ex : 
        recommendations.rename(columns={'lokasi':'Location'}, inplace=True)
        recommendations.rename(columns={'negara_bagian':'States'}, inplace=True)
        recommendations.rename(columns={'negara':'Country'}, inplace=True)
        recommendations.rename(columns={'kota':'City'}, inplace=True)
        recommendations.rename(columns={'kode_pos':'Postal code'}, inplace=True)
        recommendations.rename(columns={'nama_kemampuan_lowongan':'Position'}, inplace=True)
        recommendations.rename(columns={'spesialisasi':'Specialization'}, inplace=True)
        recommendations.rename(columns={'nama_industri_lowongan':'Industry'}, inplace=True)
        recommendations.rename(columns={'range_gaji':'Salary'}, inplace=True)
        recommendations.rename(columns={'deskripsi_benefit':'Benefit'}, inplace=True)
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
