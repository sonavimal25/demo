import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # type: ignore
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
import io

st.set_page_config(page_title="Churn & Segmentation Dashboard", layout="wide")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Customer-Churn-Records.csv")
    return df

df = load_data()

st.title("Customer Churn & Segmentation Dashboard")
st.markdown("Analyze churn risk and group customers by behavior using machine learning.")

# ---------------- Preprocessing ----------------
def preprocess_for_churn(df):
    df_churn = df.copy()
    df_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    le = LabelEncoder()
    for col in ['Geography', 'Gender', 'Card Type']:
        df_churn[col] = le.fit_transform(df_churn[col])
    return df_churn

def churn_model_pipeline(df_churn):
    X = df_churn.drop('Exited', axis=1)
    y = df_churn['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_proba)

    # Full churn risk score
    df_churn['Churn_Risk_Score'] = model.predict_proba(X)[:, 1]
    df_churn['Advisory_Flag'] = np.where(df_churn['Churn_Risk_Score'] > 0.6, 'High Risk', 'Low/Medium Risk')

    return df_churn, report, auc_score

def preprocess_for_segment(df):
    df_segment = df[['CustomerId', 'Tenure', 'NumOfProducts', 'EstimatedSalary']].dropna()
    df_segment.rename(columns={'Tenure': 'Recency', 'NumOfProducts': 'Frequency', 'EstimatedSalary': 'Monetary'}, inplace=True)
    return df_segment

def segment_pipeline(df_segment):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df_segment[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_segment['Segment'] = kmeans.fit_predict(rfm_scaled)

    strategy_map = {
        0: 'High Value – Retain & Reward',
        1: 'Medium Value – Upsell',
        2: 'At Risk – Engage Quickly',
        3: 'New or Low Value – Educate'
    }
    df_segment['Advisory_Strategy'] = df_segment['Segment'].map(strategy_map)
    return df_segment

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Churn Prediction", "Customer Segmentation","Customer Emotion"])

# ---------------- Churn Tab ----------------
with tab1:
    df_churn = preprocess_for_churn(df)
    df_churn, report, auc_score = churn_model_pipeline(df_churn)

    st.subheader("Churn Prediction Summary")
    st.metric("ROC AUC", f"{auc_score:.2f}")
    st.write("Classification Report:")
    st.json(report)

    st.subheader("Churn Risk Preview")
    st.dataframe(df_churn[['Churn_Risk_Score', 'Advisory_Flag']].head(10))

    st.download_button("📥 Download Churn Results", 
                       data=df_churn.to_csv(index=False), 
                       file_name="churn_results.csv")

# ---------------- Segment Tab ----------------
with tab2:
    df_segment = preprocess_for_segment(df)
    df_segment = segment_pipeline(df_segment)

    st.subheader("Customer Segments")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_segment, x='Recency', y='Monetary', hue='Segment', palette='Set2', ax=ax)
    plt.title("Customer Clusters")
    st.pyplot(fig)

    st.dataframe(df_segment[['CustomerId', 'Segment', 'Advisory_Strategy']].head(10))

    st.download_button("📥 Download Segmentation Results", 
                       data=df_segment.to_csv(index=False), 
                       file_name="customer_segments.csv")

with tab3:
    st.subheader("Customer Emotion")