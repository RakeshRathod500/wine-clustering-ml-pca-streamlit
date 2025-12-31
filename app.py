import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =========================================================
# LOAD MODELS
# =========================================================
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")

# =========================================================
# CLUSTER NAMES & MEANING
# =========================================================
CLUSTER_NAMES = {
    0: "🍃 Light & Fresh Wines",
    1: "⚖️ Balanced Medium-Body Wines",
    2: "🍷 Rich & Full-Bodied Wines"
}

CLUSTER_DESC = {
    0: "Lower alcohol & color intensity. Crisp, fresh, and easy-drinking wines.",
    1: "Moderate alcohol, phenols, and balance. Well-rounded profile.",
    2: "High alcohol, rich color, strong phenols. Intense & complex wines."
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wine Intelligence | PCA Clustering",
    page_icon="🍷",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
body {background-color:#fafafa;}
.title {font-size:42px;font-weight:800;color:#7b1e1e;}
.subtitle {font-size:18px;color:#555;}
.card {
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0 6px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}
.footer {
    text-align:center;
    font-size:14px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🍷 Wine Intelligence")
st.sidebar.markdown("""
**Created by**  
### 👤 Rakesh Rathod  

🎯 Data Science | ML | Analytics  
📊 PCA • K-Means • Visualization  
🚀 Streamlit Deployment
""")

st.sidebar.markdown("---")

MODE = st.sidebar.radio(
    "🧭 Select Mode",
    ["📂 Upload Dataset", "✍️ Manual Wine Input"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application demonstrates **unsupervised learning** "
    "using **PCA + K-Means** for wine segmentation."
)

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('<div class="title">🍷 Wine Clustering using PCA & K-Means</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">End-to-end ML application with real-time clustering & insights</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# MODE 1: CSV UPLOAD
# =========================================================
if MODE == "📂 Upload Dataset":
    st.markdown("### 📂 Upload Wine Dataset")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Dataset Preview")
        st.dataframe(df.head())

        # Handle Type column safely
        X = df.drop("Type", axis=1) if "Type" in df.columns else df.copy()

        # Transform
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # Predict
        labels = kmeans.predict(X_pca)
        named_labels = [CLUSTER_NAMES[i] for i in labels]

        # Metrics
        sil = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)

        c1, c2 = st.columns(2)
        c1.metric("📈 Silhouette Score", round(sil, 3))
        c2.metric("📉 Davies–Bouldin Index", round(db, 3))

        # Plot
        st.markdown("### 📊 PCA Cluster Visualization")
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(
            X_pca[:,0], X_pca[:,1],
            c=labels, cmap="viridis", alpha=0.8
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Wine Clusters in PCA Space")
        st.pyplot(fig)

        # Result table
        result_df = df.copy()
        result_df["Cluster Group"] = named_labels

        st.markdown("### 🧾 Clustered Dataset")
        st.dataframe(result_df.head())

        # Download
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Clustered Dataset",
            csv,
            "wine_clusters.csv",
            "text/csv"
        )

# =========================================================
# MODE 2: MANUAL INPUT
# =========================================================
if MODE == "✍️ Manual Wine Input":
    st.markdown("### ✍️ Enter Wine Chemical Properties")

    with st.form("wine_form"):
        c1, c2, c3 = st.columns(3)

        alcohol = c1.number_input("Alcohol", 0.0)
        malic = c1.number_input("Malic Acid", 0.0)
        ash = c1.number_input("Ash", 0.0)
        alcalinity = c1.number_input("Alcalinity", 0.0)

        magnesium = c2.number_input("Magnesium", 0)
        phenols = c2.number_input("Phenols", 0.0)
        flavanoids = c2.number_input("Flavanoids", 0.0)
        dilution = c2.number_input("Dilution", 0.0)

        nonflavanoids = c3.number_input("Nonflavanoids", 0.0)
        proanthocyanins = c3.number_input("Proanthocyanins", 0.0)
        color = c3.number_input("Color Intensity", 0.0)
        hue = c3.number_input("Hue", 0.0)
        proline = c3.number_input("Proline", 0)

        submitted = st.form_submit_button("🔮 Predict Wine Cluster")

    if submitted:
        input_df = pd.DataFrame([[
            alcohol, malic, ash, alcalinity, magnesium,
            phenols, flavanoids, nonflavanoids,
            proanthocyanins, color, hue, dilution, proline
        ]], columns=[
            "Alcohol","Malic","Ash","Alcalinity","Magnesium",
            "Phenols","Flavanoids","Nonflavanoids",
            "Proanthocyanins","Color","Hue","Dilution","Proline"
        ])

        X_scaled = scaler.transform(input_df)
        X_pca = pca.transform(X_scaled)
        cid = kmeans.predict(X_pca)[0]

        st.success(f"🍷 **Predicted Category:** {CLUSTER_NAMES[cid]}")
        st.info(CLUSTER_DESC[cid])

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="footer">🚀 Built with ❤️ by <b>Rakesh Rathod</b> | PCA + K-Means | Streamlit ML App</div>',
    unsafe_allow_html=True
)
