!pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score


# Fungsi untuk memuat dataset
@st.cache_data
def load_dataset(file):
    try:
        df = pd.read_csv(file)
        if 'clear' not in df.columns or 'Label' not in df.columns:
            st.error("Dataset harus memiliki kolom 'clear' dan 'Label'")
            return None
        return df.dropna(subset=['clear', 'Label'])
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

# Fungsi untuk melatih model
@st.cache_resource
def train_models(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df['clear'], df['Label'], test_size=0.4, random_state=42
    )

    pipelines = {
        "TF-IDF + Naive Bayes": make_pipeline(TfidfVectorizer(), MultinomialNB()),
        "CountVectorizer + Naive Bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
        "TF-IDF + Decision Tree": make_pipeline(TfidfVectorizer(), DecisionTreeClassifier(random_state=42)),
        "CountVectorizer + Decision Tree": make_pipeline(CountVectorizer(), DecisionTreeClassifier(random_state=42)),
        "TF-idf + SVM": make_pipeline(TfidfVectorizer(),svm.SVC(kernel='linear')),
        "CountVectorizer + SVM": make_pipeline(CountVectorizer(),svm.SVC(kernel='linear'))

    }

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)

    return pipelines, x_test, y_test

# Fungsi untuk prediksi dan akurasi
def predict_and_display(pipelines, x_test, y_test, input_text):
    results = []
    for name, pipeline in pipelines.items():
        prediction = pipeline.predict([input_text])[0]
        accuracy = accuracy_score(y_test, pipeline.predict(x_test))
        results.append((name, prediction, accuracy))
    return results


# GUI dengan Streamlit
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="wide")
st.title("🧠 Sentiment Analysis App")

# Sidebar untuk memuat dataset
st.sidebar.header("📂 Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_dataset(uploaded_file)
    if df is not None:
        st.sidebar.success("Dataset berhasil dimuat!")
        pipelines, x_test, y_test = train_models(df)

        st.write("### 🗂️ Preview Dataset")
        st.dataframe(df.head())

        st.write("### 📊 Masukkan Teks untuk Analisis Sentimen")
        input_text = st.text_area("Masukkan teks ulasan di sini")

        if st.button("🔍 Analisis Sentimen"):
            if input_text:
                results = predict_and_display(pipelines, x_test, y_test, input_text)
                st.write("### 📊 Hasil Prediksi:")
                for name, prediction, accuracy in results:
                    st.write(f"**Model:** {name}")
                    st.write(f"**Prediksi:** {prediction}")
                    st.write(f"**Akurasi:** {accuracy:.2f}")
                    st.write("---")
            else:
                st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
else:
    st.info("Silakan upload dataset untuk memulai analisis.")

# Footer
st.sidebar.write("---")
