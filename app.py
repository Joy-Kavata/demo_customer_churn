import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page setup
st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('churn_random_forest_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Header
st.title("Telco Customer Churn Predictor Demo")
st.markdown("Adjust customer profile attributes on the left to evaluate real-time churn probabilities and risk levels.")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", value=65.00, step=5.0)
total_charges = st.sidebar.number_input("Total Charges ($)", value=780.00, step=50.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Prepare input data matching model encoding
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
    'InternetService_No': [1 if internet == "No" else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
    'PaymentMethod_Electronic check': [1 if payment == "Electronic check" else 0],
    'PaymentMethod_Mailed check': [1 if payment == "Mailed check" else 0]
})

# Scale numerical columns if needed
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
if all(col in input_data.columns for col in num_cols):
    input_data[num_cols] = scaler.transform(input_data[num_cols])

# Make prediction
if st.button("Calculate Churn Risk", type="primary"):
    input_df = input_data.copy()
    
    # Align feature columns with model training schema
    if hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

    # Predict probability
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_percentage = round(churn_prob * 100, 1)

    # Layout into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Churn Probability", value=f"{churn_percentage}%")

    with col2:
        if churn_percentage >= 50:
            st.metric(label="Risk Level", value="🔴 High Risk")
        elif churn_percentage >= 25:
            st.metric(label="Risk Level", value="🟡 Medium Risk")
        else:
            st.metric(label="Risk Level", value="🟢 Low Risk")

    with col3:
        annual_val = monthly_charges * 12
        st.metric(label="Est. Annual Value at Risk", value=f"${annual_val:,.2f}")

    # Visual progress bar
    st.write("### Churn Risk Assessment")
    st.progress(int(churn_percentage))

    # Actionable guidance
    if churn_percentage >= 50:
        st.error("⚠️ **Retention Action Required:** High churn probability detected. Recommend targeted discount incentives or contract upgrade offers immediately.")
    elif churn_percentage >= 25:
        st.warning("⚡ **Monitor Customer:** Moderate churn risk. Recommend automated engagement campaigns or service satisfaction check-ins.")
    else:
        st.success("✅ **Healthy Customer:** Low churn risk. Standard lifecycle communications recommended.")

    # Feature Importance visual
    st.markdown("---")
    st.write("### Top Feature Drivers")

    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importances, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Model Feature Weighting", 
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
