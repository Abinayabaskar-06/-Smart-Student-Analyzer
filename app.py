import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Student Analyzer", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Different card colors */
.card1 { background: linear-gradient(45deg, #8E2DE2, #4A00E0); }
.card2 { background: linear-gradient(45deg, #11998e, #38ef7d); }
.card3 { background: linear-gradient(45deg, #396afc, #2948ff); }
.card4 { background: linear-gradient(45deg, #f7971e, #ffd200); color:black; }

/* Upload box */
.upload-box {
    border: 2px dashed #888;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
            
/* Make ALL labels white */
label {
    color: white !important;
    font-weight: bold !important;
}

/* Input text color */
input {
    color: black !important;
}

/* Also fix number input text */
div[data-baseweb="input"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="big-title">🎓 Smart Student Analyzer</p>', unsafe_allow_html=True)
st.markdown("### 🚀 AI-Based Student Segmentation System")

# ---------------- USER INPUT ----------------
st.subheader("🧾 Enter Student Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    study_hours = st.number_input("Study Hours", 0, 12, 5)

with col2:
    attendance = st.number_input("Attendance (%)", 0, 100, 75)

with col3:
    assignment = st.number_input("Assignment Score", 0, 100, 70)

with col4:
    marks = st.number_input("Marks", 0, 100, 65)

# ---------------- DATASET ----------------
data = {
    'StudyHours': [2,3,5,7,8,1,4,6,9,10],
    'Attendance': [60,65,70,80,90,55,75,85,95,98],
    'Assignment': [50,55,65,75,85,45,70,80,90,95],
    'Marks': [40,50,60,70,85,35,65,78,88,92]
}

df = pd.DataFrame(data)

# Add user input
new_data = pd.DataFrame({
    'StudyHours': [study_hours],
    'Attendance': [attendance],
    'Assignment': [assignment],
    'Marks': [marks]
})

df = pd.concat([df, new_data], ignore_index=True)

# ---------------- MODEL ----------------
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

# ---------------- OUTPUT ----------------
st.subheader("🎯 Prediction Result")

user_cluster = df.iloc[-1]['Cluster']

if user_cluster == 0:
    st.success("Cluster 0 → 📉 Low Performance Student")
elif user_cluster == 1:
    st.success("Cluster 1 → 📊 Average Student")
else:
    st.success("Cluster 2 → 📈 High Performance Student")

# ---------------- GRAPH ----------------
st.subheader("📊 Visualization")

# Create two columns
col1, col2 = st.columns(2)

# ---------- LEFT SIDE (SCATTER) ----------
with col1:
    st.markdown("#### 📈 Clustering")

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Light background
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    # Colors
    colors = ['purple', 'green', 'orange']
    cluster_colors = [colors[int(i)] for i in df['Cluster']]

    ax.scatter(df['StudyHours'], df['Marks'], c=cluster_colors, s=50)

    ax.set_xlabel("Study Hours", fontsize=8, color="black")
    ax.set_ylabel("Marks", fontsize=8, color="black")

    ax.tick_params(axis='both', labelsize=7, colors='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)


# ---------- RIGHT SIDE (BAR CHART) ----------
with col2:
    st.markdown("#### 📊 Feature Distribution")

    avg_values = df[['StudyHours', 'Attendance', 'Assignment', 'Marks']].mean()

    fig2, ax2 = plt.subplots(figsize=(4, 2.5))

    # Light background
    ax2.set_facecolor("#ffffff")
    fig2.patch.set_facecolor("#ffffff")

    colors = ['purple', 'green', 'orange', 'blue']

    ax2.bar(avg_values.index, avg_values.values, color=colors)

    ax2.set_ylabel("Avg Value", fontsize=8, color="black")
    ax2.set_xlabel("Features", fontsize=8, color="black")

    ax2.tick_params(axis='both', labelsize=7, colors='black')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    st.pyplot(fig2)

# ---------------- TABLE ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df)

# ---------------- DOWNLOAD ----------------
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Result", csv, "student_clusters.csv")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ for Education")