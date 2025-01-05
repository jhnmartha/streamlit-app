import streamlit as st
import numpy as np
import joblib

# Fungsi untuk memuat model
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file {model_path} tidak ditemukan.")
        st.stop()

# Fungsi prediksi
def predict_with_model(model, input_data, label_mapping):
    try:
        prediction = model.predict(input_data)
        return label_mapping.get(prediction[0], "Label tidak dikenali")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")

# Label mapping
label_mapping_fish = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conconius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}
label_mapping_fruit = {
    0: "Orange",
    1: "Grapefruit"
}
label_mapping_pumpkin = {
    0: "Çerçevelik",
    1: "Ürgüp Sivrisi"
}

# Sidebar
st.sidebar.title("MEMPREDIKSI DATASET")
learning_type = st.sidebar.radio(
    "Pilih Jenis Algoritma:",
    ["Supervised Learning", "Unsupervised Learning"]
)

if learning_type == "Supervised Learning":
    with st.sidebar.expander("Pilih Model dan Dataset", expanded=True):
        model_choice = st.selectbox(
            "Model:",
            ["Random Forest Regression", "Perceptron", "SVM"]
        )
        dataset_choice = st.selectbox(
            "Dataset:",
            ["Fish", "Fruit", "Pumpkin Seeds"]
        )

    if dataset_choice == "Fish":
        st.title(f"Prediksi Ikan Algoritma {model_choice}")
        with st.expander("Masukkan Data", expanded=True):
            length = st.number_input("Length (cm):", min_value=0.0, step=0.1)
            weight = st.number_input("Weight (gram):", min_value=0.0, step=0.1)
            wlratio = st.number_input("Weight Length Ratio:", min_value=0.0, step=0.1)

        model_path = {
            "Random Forest Regression": "rf_fish-scale.pkl",
            "Perceptron": "perceptron_fish-scale.pkl",
            "SVM": "svm_fish-scale.pkl"
        }

        if st.button("Prediksi Ikan"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[length, weight, wlratio]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_fish)
            st.success(f"Hasil Prediksi: **{hasil_prediksi}**")

    elif dataset_choice == "Fruit":
        st.title(f"Prediksi Buah Algoritma {model_choice}")
        with st.expander("Masukkan Data", expanded=True):
            diameter = st.number_input("Diameter (cm):", min_value=0.0, step=0.1)
            weight = st.number_input("Weight (gram):", min_value=0.0, step=0.1)
            red = st.number_input("Red (RGB):", min_value=0.0, step=1.0)
            green = st.number_input("Green (RGB):", min_value=0.0, step=1.0)
            blue = st.number_input("Blue (RGB):", min_value=0.0, step=1.0)

        model_path = {
            "Random Forest Regression": "rf_fruit-scale.pkl",
            "Perceptron": "perceptron_fruit-scale.pkl",
            "SVM": "svm_fruit-scale.pkl"
        }

        if st.button("Prediksi Buah"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[diameter, weight, red, green, blue]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_fruit)
            st.success(f"Hasil Prediksi: **{hasil_prediksi}**")

    elif dataset_choice == "Pumpkin Seeds":
        st.title(f"Prediksi Biji Labu Algoritma {model_choice}")
        with st.expander("Masukkan Data", expanded=True):
            area = st.number_input("Area:", min_value=0.0, step=0.1)
            perimeter = st.number_input("Perimeter:", min_value=0.0, step=0.1)
            major_axis_length = st.number_input("Major Axis Length:", min_value=0.0, step=0.1)
            minor_axis_length = st.number_input("Minor Axis Length:", min_value=0.0, step=0.1)
            convex_area = st.number_input("Convex Area:", min_value=0.0, step=0.1)
            equiv_diameter = st.number_input("Equiv Diameter:", min_value=0.0, step=0.1)
            eccentricity = st.number_input("Eccentricity:", min_value=0.0, step=0.1)
            solidity = st.number_input("Solidity:", min_value=0.0, step=0.1)
            extent = st.number_input("Extent:", min_value=0.0, step=0.1)
            roundness = st.number_input("Roundness:", min_value=0.0, step=0.1)
            aspect_ration = st.number_input("Aspect Ratio:", min_value=0.0, step=0.1)
            compactness = st.number_input("Compactness:", min_value=0.0, step=0.1)

        model_path = {
            "Random Forest Regression": "rf_pumpkin-scale.pkl",
            "Perceptron": "perceptron_pumpkin-scale.pkl",
            "SVM": "svm_pumpkin-scale.pkl"
        }

        if st.button("Prediksi Biji Labu"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[area, perimeter, major_axis_length, minor_axis_length,
                                    convex_area, equiv_diameter, eccentricity, solidity,
                                    extent, roundness, aspect_ration, compactness]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_pumpkin)
            st.success(f"Hasil Prediksi: **{hasil_prediksi}**")

elif learning_type == "Unsupervised Learning":
    st.title("Clustering Menggunakan Algoritma K-Means")
    with st.expander("Dataset Yang Digunakan", expanded=True):
        st.write("Wine Clustering:")
        data_to_cluster = np.random.rand(100, 3) 

    n_clusters = st.slider("Jumlah Cluster:", min_value=2, max_value=10, value=3)

    if st.button("Clustering"):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_to_cluster)

        st.subheader(f"Hasil Clustering dengan {n_clusters} Cluster")
        st.write("Cluster Labels:", clusters)
        st.write("Centroid Cluster:", kmeans.cluster_centers_)
else:
    st.write("Pilih tipe pembelajaran dari sidebar untuk melanjutkan.")
