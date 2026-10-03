import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from predict import load_model, predict_price

# --- Configuration ---
st.set_page_config(page_title="Real Estate ML Analytics", page_icon="🏢", layout="wide")

# --- Caching Data & Model ---
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def load_data():
    return pd.read_csv('data/house_prices.csv')

@st.cache_data
def load_metrics():
    with open('models/model_metrics.json', 'r') as f:
        return json.load(f)

# --- Sidebar Navigation ---
st.sidebar.title("🏢 Estate Analytics")
st.sidebar.markdown("Professional House Price Prediction System")
page = st.sidebar.radio("Navigation", ["Home Dashboard", "Price Predictor", "Market Analytics", "Model Information"])

# --- Helper Formatting ---
def format_inr(amount):
    return f"₹ {amount:,.2f}"

# --- Page: Home Dashboard ---
if page == "Home Dashboard":
    st.title("House Price Prediction System")
    st.markdown("Welcome to the AI-powered real estate valuation platform.")
    
    try:
        df = load_data()
        metrics = load_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Properties in DB", f"{len(df):,}")
        col2.metric("Average Market Price", format_inr(df['Price_INR'].mean()))
        col3.metric("Supported Cities", df['Location'].nunique())
        col4.metric("Model Accuracy (R²)", f"{metrics['metrics']['R2']*100:.1f}%")
        
        st.divider()
        st.markdown("""
        ### Project Overview
        This professional dashboard utilizes Machine Learning to estimate property values based on historical data and property features. 
        
        **How to use this tool:**
        1. Navigate to the **Price Predictor** to get instant AI valuations.
        2. Use **Market Analytics** to explore regional real-estate trends.
        3. Check **Model Information** for transparency on the ML algorithms used.
        """)
        
        st.info("💡 Note: The dataset powering this demo is a synthetic statistical representation of the Indian housing market designed for analytical demonstration.")
        
    except FileNotFoundError:
        st.error("Data or model files missing. Please run `data/generate_data.py` and `train_model.py`.")

# --- Page: Price Predictor ---
elif page == "Price Predictor":
    st.title("Property Price Predictor")
    st.markdown("Enter property details below to receive a machine learning-based price estimate.")
    
    try:
        model = get_model()
        df = load_data()
        
        with st.form("prediction_form"):
            st.subheader("Property Specifications")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                location = st.selectbox("City / Location", sorted(df['Location'].unique()))
                prop_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
                area = st.number_input("Area (Sq. Ft.)", min_value=300, max_value=10000, value=1200, step=50)
                
            with col2:
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
                floors = st.number_input("Total Floors", min_value=1, max_value=50, value=1)
                
            with col3:
                age = st.number_input("Property Age (Years)", min_value=0, max_value=100, value=5)
                parking = st.radio("Parking Available", ["Yes", "No"])
                
            submit_button = st.form_submit_button("Generate Valuation Estimate")
            
        if submit_button:
            input_data = {
                "Location": location,
                "Property_Type": prop_type,
                "Area_sqft": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "Age_years": age,
                "Parking": parking
            }
            
            with st.spinner("Analyzing market data..."):
                prediction = predict_price(model, input_data)
                
            st.success("Analysis Complete!")
            st.markdown(f"""
            <div style="background-color:#F4F6F9; padding:20px; border-radius:10px; border-left: 5px solid #2E8B57;">
                <h3 style="color:#0A192F; margin:0;">Estimated Market Value</h3>
                <h1 style="color:#2E8B57; margin:0;">{format_inr(prediction)}</h1>
                <p style="margin-top:10px; font-size:14px; color:#555;"><i>*This is an AI-generated estimate and should not replace a professional appraisal.</i></p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error loading prediction engine: {e}")

# --- Page: Market Analytics ---
elif page == "Market Analytics":
    st.title("Market Analytics Dashboard")
    
    try:
        df = load_data()
        
        # Filters
        st.sidebar.subheader("Filter Analytics")
        city_filter = st.sidebar.multiselect("Select Cities", df['Location'].unique(), default=df['Location'].unique())
        filtered_df = df[df['Location'].isin(city_filter)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig1 = px.histogram(filtered_df, x="Price_INR", nbins=50, color_discrete_sequence=['#2E8B57'])
            fig1.update_layout(xaxis_title="Price (INR)", yaxis_title="Count")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Average Price by Location")
            avg_price = filtered_df.groupby('Location')['Price_INR'].mean().reset_index()
            fig2 = px.bar(avg_price, x='Location', y='Price_INR', color='Location', color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Area vs. Price Analysis")
        fig3 = px.scatter(filtered_df, x="Area_sqft", y="Price_INR", color="Property_Type", hover_data=['Location', 'Bedrooms'], opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)
            
    except Exception as e:
        st.error("Data not available for analytics.")

# --- Page: Model Information ---
elif page == "Model Information":
    st.title("Machine Learning Architecture")
    
    try:
        metrics = load_metrics()
        
        st.markdown("### Pipeline Overview")
        st.write("This system utilizes a robust Scikit-learn pipeline to prevent data leakage. The pipeline applies One-Hot Encoding to categorical variables (Location, Property Type, Parking) and Standard Scaling to numerical variables before feeding them into the regressor.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Selected Model:** `{metrics['selected_model']}`")
            st.markdown(f"**Training Data Size:** `{metrics['training_size']} records`")
            st.markdown("**Features Utilized:**")
            for f in metrics['features']:
                st.write(f"- {f}")
                
        with col2:
            st.markdown("### Performance Metrics")
            st.metric("R² Score (Accuracy)", f"{metrics['metrics']['R2']*100:.2f}%")
            st.metric("Root Mean Squared Error (RMSE)", format_inr(metrics['metrics']['RMSE']))
            st.metric("Mean Absolute Error (MAE)", format_inr(metrics['metrics']['MAE']))
            
        st.divider()
        st.markdown("#### System Limitations")
        st.warning("1. **Data Scope:** The current model is trained on a predefined synthetic statistical dataset. Real-world accuracy requires fine-tuning on regional municipal data.\n2. **Feature Limitations:** Does not account for macro-economic factors (interest rates) or localized micro-features (proximity to transit).")
        
    except Exception as e:
        st.error("Model metrics not found. Train the model first.")
