import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scalers
# Ensure these files are in the same directory as your script
try:
    model = pickle.load(open('final_model.pkl', 'rb'))
    scalers = pickle.load(open('scalers.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'final_model.pkl' and 'scalers.pkl' are in the correct directory.")
    st.stop()


# Numerical columns used in training (must be in the same order as when the model was trained)
numerical_columns = list(scalers.keys())

# --- Recommendation Function ---
def generate_recommendations(processed_data):
    """Generates personalized health recommendations based on user input."""
    recommendations = []
    
    # 1. BMI Recommendation
    if processed_data['bmi'] >= 30.0:
        recommendations.append(
            "**High BMI:** Your BMI is in the obese range. "
            "Focus on a balanced diet and regular physical activity to achieve a healthy weight, which can significantly lower your diabetes risk."
        )
    elif processed_data['bmi'] >= 25.0:
        recommendations.append(
            "**Elevated BMI:** Your BMI is in the overweight range. "
            "Consider incorporating more whole foods into your diet and increasing your daily physical activity."
        )

    # 2. Blood Glucose Recommendation
    # Assuming the input is Fasting Blood Glucose in mg/dL
    if processed_data['blood_glucose'] >= 126:
        recommendations.append(
            "**High Blood Glucose:** Your blood glucose level is in the diabetic range. "
            "It is crucial to consult a healthcare provider immediately for a formal diagnosis and management plan."
        )
    elif processed_data['blood_glucose'] >= 100:
        recommendations.append(
            "**Elevated Blood Glucose:** Your blood glucose level is in the prediabetic range. "
            "Prioritize a low-sugar diet, regular exercise, and consult with a doctor to prevent progression to type 2 diabetes."
        )

    # 3. Physical Activity Recommendation
    if processed_data['physical_activity'] < 30:
        recommendations.append(
            "**Low Physical Activity:** Aim for at least 30 minutes of moderate physical activity (like brisk walking, cycling, or swimming) most days of the week."
        )

    # 4. Diet Recommendation
    if processed_data['diet'] == 0:  # 0 corresponds to "unhealthy"
        recommendations.append(
            "**Diet Improvement:** A healthy diet is key. "
            "Focus on eating whole grains, lean proteins, fruits, and vegetables while reducing processed foods, sugary drinks, and saturated fats."
        )
        
    # 5. Stress Level Recommendation
    if processed_data['stress_level'] > 0:  # 1 for "medium", 2 for "high"
        recommendations.append(
            "**Stress Management:** Chronic stress can affect blood sugar. "
            "Consider stress-reduction techniques like mindfulness, yoga, meditation, or spending time in nature."
        )

    # 6. Sleep Hours Recommendation
    if processed_data['sleep_hours'] < 7:
        recommendations.append(
            "**Improve Sleep:** Lack of quality sleep can impact insulin resistance. "
            "Aim for 7-9 hours of restful sleep per night by maintaining a consistent sleep schedule and creating a relaxing bedtime routine."
        )
    
    # 7. Hydration Recommendation
    if processed_data['hydration_level'] == 0: # 0 corresponds to "no"
        recommendations.append(
            "**Stay Hydrated:** Proper hydration is important for overall health and can aid in blood sugar regulation. "
            "Make sure you are drinking enough water throughout the day."
        )
    
    return recommendations

# --- Streamlit App UI ---
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Input your data to estimate your risk of diabetes. This tool provides an estimate and is not a substitute for professional medical advice.")

# Create columns for a better layout
col1, col2 = st.columns(2)

with col1:
    user_input = {}
    user_input['weight'] = st.number_input("Enter your weight (kg)", min_value=0.0, format="%.2f")
    user_input['height'] = st.number_input("Enter your height (cm)", min_value=0.0, format="%.2f")
    user_input['blood_glucose'] = st.number_input("Enter your blood glucose level (mg/dL)", min_value=0.0, format="%.2f")
    user_input['physical_activity'] = st.number_input("Physical activity per day (minutes)", min_value=0.0, max_value=1440.0, step=1.0, format="%.2f")
    user_input['sleep_hours'] = st.number_input("Average sleep hours per night", min_value=0.0, max_value=24.0, step=1.0, format="%.1f")


with col2:
    user_input['diet'] = st.selectbox("How would you describe your diet?", options=["healthy", "unhealthy"])
    user_input['medication_adherence'] = st.selectbox("Medication adherence (if applicable)", options=["good", "poor"])
    user_input['stress_level'] = st.selectbox("Describe your typical stress level", options=["low", "medium", "high"])
    user_input['hydration_level'] = st.selectbox("Are you adequately hydrated?", options=["yes", "no"])

# Convert categorical inputs to numerical format for the model
user_input_processed = user_input.copy()
user_input_processed['diet'] = 1 if user_input['diet'] == "healthy" else 0
user_input_processed['medication_adherence'] = 1 if user_input['medication_adherence'] == "good" else 0
user_input_processed['stress_level'] = {"low": 0, "medium": 1, "high": 2}[user_input['stress_level']]
user_input_processed['hydration_level'] = 1 if user_input['hydration_level'] == "yes" else 0


if st.button("Predict My Risk", type="primary"):
    # Basic validation
    if user_input['height'] <= 0 or user_input['weight'] <= 0:
        st.error("Please enter valid weight and height values.")
    else:
        # Compute BMI and add it to the processed dictionary
        height_in_meters = user_input['height'] / 100
        bmi = user_input['weight'] / (height_in_meters ** 2)
        user_input_processed['bmi'] = bmi
        
        # Add BMI to the original user_input dictionary for the recommendation function
        user_input['bmi'] = bmi

        # Create a DataFrame from the processed input
        # Ensure the column order matches the training data
        try:
            input_df = pd.DataFrame([user_input_processed])
            
            # Reorder columns to match the order used during model training
            # This is crucial for the model and scalers to work correctly
            all_required_cols = numerical_columns + [col for col in input_df.columns if col not in numerical_columns]
            input_df_reordered = input_df[all_required_cols]

            # Apply scalers to numerical columns
            # Create a copy to avoid SettingWithCopyWarning
            input_df_scaled = input_df_reordered.copy()
            for col in numerical_columns:
                if col in input_df_scaled.columns:
                    input_df_scaled[[col]] = scalers[col].transform(input_df_scaled[[col]])

            # Predict risk
            risk_label = model.predict(input_df_scaled)[0]

            # Display prediction result
            st.subheader("Prediction Result")
            if "high" in risk_label.lower():
                st.error(f"🩺 **Predicted Diabetes Risk Level: {risk_label}**")
            else:
                st.success(f"🩺 **Predicted Diabetes Risk Level: {risk_label}**")

            # --- Display Recommendations ---
            st.subheader("Personalized Recommendations")
            recommendations = generate_recommendations(user_input_processed)
            
            if not recommendations:
                st.info("Great job! Your inputs suggest you are following healthy habits. Keep up the good work to maintain a low risk of diabetes.")
            else:
                for rec in recommendations:
                    st.warning(f"💡 {rec}")

            st.markdown("---")
            st.info("**Disclaimer:** This prediction is based on a machine learning model and should not be considered a medical diagnosis. Always consult with a qualified healthcare professional for any health concerns.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure your model's required features match the input fields.")