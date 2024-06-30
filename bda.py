import os
import sys
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the path to your Python executable
python_executable_path = sys.executable

# Set environment variables for PySpark
os.environ['PYSPARK_PYTHON'] = python_executable_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable_path

# Initialize Spark session
spark = SparkSession.builder.appName("Mobile PhonePricePrediction").getOrCreate()

# Load the data
data_path = "cellphone.csv"  # Ensure this path is correct
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Select relevant columns for training
feature_columns = ['sale', 'weight', 'resolution', 'ppi', 'cpu_core', 'cpu_freq', 'internal_mem', 'ram', 'rear_cam', 'front_cam', 'battery', 'thickness']
data = data.select(*feature_columns, 'price')

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=123)

# Initialize and train the linear regression model
lr = LinearRegression(featuresCol='features', labelCol='price')
lr_model = lr.fit(train_data)

# Evaluate the model on test data
test_results = lr_model.evaluate(test_data)
rmse = test_results.rootMeanSquaredError
r2 = test_results.r2

# Function to predict price based on input parameters
def predict_price(sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness):
    input_data = pd.DataFrame([[sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness]], 
                              columns=feature_columns)
    input_spark_df = spark.createDataFrame(input_data)
    input_spark_df = assembler.transform(input_spark_df)
    prediction = lr_model.transform(input_spark_df)
    predicted_price = prediction.select("prediction").collect()[0][0]
    return prediction.select("prediction").collect()[0][0]
    # new_predicted_price = -1 * predicted_price
    # return new_predicted_price
# Function to get confidence interval for prediction
def predict_price_with_confidence_interval(sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness):
    prediction = predict_price(sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness)
    std_error = np.std(test_results.residuals.select("residuals").collect())
    confidence_interval = 1.96 * std_error
    return prediction, confidence_interval

# Streamlit App
st.set_page_config(page_title="Mobile Phone Price Prediction", layout="wide", initial_sidebar_state="expanded")


# Initialize session state variables
if 'sale' not in st.session_state:
    st.session_state.sale = 10
if 'weight' not in st.session_state:
    st.session_state.weight = 135
if 'resolution' not in st.session_state:
    st.session_state.resolution = 5.2
if 'ppi' not in st.session_state:
    st.session_state.ppi = 424
if 'cpu_core' not in st.session_state:
    st.session_state.cpu_core = 8
if 'cpu_freq' not in st.session_state:
    st.session_state.cpu_freq = 1.35
if 'internal_mem' not in st.session_state:
    st.session_state.internal_mem = 16
if 'ram' not in st.session_state:
    st.session_state.ram = 3
if 'rear_cam' not in st.session_state:
    st.session_state.rear_cam = 13
if 'front_cam' not in st.session_state:
    st.session_state.front_cam = 8
if 'battery' not in st.session_state:
    st.session_state.battery = 2160
if 'thickness' not in st.session_state:
    st.session_state.thickness = 7.4


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Price", "Compare Prices", "Data Overview", "Model Performance", "Feature Correlations", "Advanced Analytics"])

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stRadio>div {
        flex-direction: row;
    }
    .stSidebar .css-1d391kg {
        background-color: #f0f2f6;
        color: #0c0d0e;
        font-size: 18px;
        border-right: 2px solid #f0f0f5;
    }
    .css-1aumxhk {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

if page == "Home":
    st.title("Mobile Phone Price Prediction")
    st.markdown("""
    Welcome to the **Mobile Phone Price Prediction** app! Use the sidebar to navigate through different sections.
    - **Predict Price**: Input Mobile Phone specifications to predict the price.
    - **Compare Prices**: Compare prices of multiple Mobile Phones.
    - **Data Overview**: View and explore the dataset and its distribution.
    - **Model Performance**: Check the performance metrics of the prediction model.
    - **Feature Correlations**: Explore correlations between different features.
    - **Advanced Analytics**: Gain advanced insights and analytics.
    """)

elif page == "Predict Price":
    st.title("Predict Mobile Phone Price")
    st.write("### Input the specifications of the Mobile Phone to predict its price")

    col1, col2, col3 = st.columns(3)
    with col1:
        sale = st.number_input("Sale", min_value=0, value=10, step=1, key="sale")
        weight = st.number_input("Weight (g)", min_value=0, value=135, step=1, key="weight")
        resolution = st.number_input("Resolution", min_value=0.0, value=5.2, step=0.1, key="resolution")
        ppi = st.number_input("PPI", min_value=0, value=424, step=1, key="ppi")
    with col2:
        cpu_core = st.number_input("CPU Cores", min_value=1, value=8, step=1, key="cpu_core")
        cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.0, value=1.35, step=0.1, key="cpu_freq")
        internal_mem = st.number_input("Internal Memory (GB)", min_value=0, value=16, step=1, key="internal_mem")
        ram = st.number_input("RAM (GB)", min_value=0, value=3, step=1, key="ram")
    with col3:
        rear_cam = st.number_input("Rear Camera (MP)", min_value=0, value=13, step=1, key="rear_cam")
        front_cam = st.number_input("Front Camera (MP)", min_value=0, value=8, step=1, key="front_cam")
        battery = st.number_input("Battery (mAh)", min_value=0, value=2160, step=1, key="battery")
        thickness = st.number_input("Thickness (mm)", min_value=0.0, value=7.4, step=0.1, key="thickness")

    if st.button("Predict Price"):
        predicted_price, confidence_interval = predict_price_with_confidence_interval(
            st.session_state.sale, st.session_state.weight, st.session_state.resolution, st.session_state.ppi, 
            st.session_state.cpu_core, st.session_state.cpu_freq, st.session_state.internal_mem, 
            st.session_state.ram, st.session_state.rear_cam, st.session_state.front_cam, 
            st.session_state.battery, st.session_state.thickness
        )
        st.success(f"The predicted price is: {predicted_price:.2f} ± {confidence_interval:.2f}")

elif page == "Compare Prices":
    st.title("Compare Mobile Phone Prices")
    st.write("### Input the specifications of the Mobile Phones to compare their prices")

    num_phones = st.slider("Number of phones to compare", 2, 5, 2)
    phones = []

    for i in range(num_phones):
        st.write(f"### Phone {i+1} Specifications")
        col1, col2, col3 = st.columns(3)
        with col1:
            sale = st.number_input(f"Sale {i+1}", min_value=0, value=10, step=1, key=f"sale_{i}")
            weight = st.number_input(f"Weight (g) {i+1}", min_value=0, value=135, step=1, key=f"weight_{i}")
            resolution = st.number_input(f"Resolution {i+1}", min_value=0.0, value=5.2, step=0.1, key=f"resolution_{i}")
            ppi = st.number_input(f"PPI {i+1}", min_value=0, value=424, step=1, key=f"ppi_{i}")
        with col2:
            cpu_core = st.number_input(f"CPU Cores {i+1}", min_value=1, value=8, step=1, key=f"cpu_core_{i}")
            cpu_freq = st.number_input(f"CPU Frequency (GHz) {i+1}", min_value=0.0, value=1.35, step=0.1, key=f"cpu_freq_{i}")
            internal_mem = st.number_input(f"Internal Memory (GB) {i+1}", min_value=0, value=16, step=1, key=f"internal_mem_{i}")
            ram = st.number_input(f"RAM (GB) {i+1}", min_value=0, value=3, step=1, key=f"ram_{i}")
        with col3:
            rear_cam = st.number_input(f"Rear Camera (MP) {i+1}", min_value=0, value=13, step=1, key=f"rear_cam_{i}")
            front_cam = st.number_input(f"Front Camera (MP) {i+1}", min_value=0, value=8, step=1, key=f"front_cam_{i}")
            battery = st.number_input(f"Battery (mAh) {i+1}", min_value=0, value=2610, step=1, key=f"battery_{i}")
            thickness = st.number_input(f"Thickness (mm) {i+1}", min_value=0.0, value=7.5, step=0.1, key=f"thickness_{i}")
        
        phones.append((sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness))

    if st.button("Compare Prices"):
        predictions = [predict_price(*phone) for phone in phones]
        for i, prediction in enumerate(predictions):
            st.success(f"The predicted price for Phone {i+1} is: {prediction:.2f}")

elif page == "Data Overview":
    st.title("Data Overview")
    st.write("### Dataset")

    # Remove the 'features' column and original feature columns from Spark DataFrame
    pandas_data = data.drop(*feature_columns).drop("features").toPandas()

    # Extract the 'features' column into separate columns
    features_array = np.array(data.select("features").rdd.map(lambda row: row.features.toArray()).collect())
    features_df = pd.DataFrame(features_array, columns=feature_columns)
    
    # Concatenate the Pandas DataFrame with the new features DataFrame
    complete_data = pd.concat([pandas_data, features_df], axis=1)
    
    st.write(complete_data)


elif page == "Model Performance":
    st.title("Model Performance")
    st.write("### Root Mean Squared Error (RMSE)")
    st.write(rmse)
    st.write("### R-squared (R2)")
    st.write(r2)

elif page == "Feature Correlations":
    st.title("Feature Correlations")
    pandas_data = data.drop(*feature_columns).drop("features").toPandas()

    # Extract the 'features' column into separate columns
    features_array = np.array(data.select("features").rdd.map(lambda row: row.features.toArray()).collect())
    features_df = pd.DataFrame(features_array, columns=feature_columns)
    
    # Concatenate the original data (without 'features') with the new features DataFrame
    complete_data = pd.concat([pandas_data, features_df], axis=1)
    
    corr_matrix = complete_data.corr()

    st.write("### Correlation Matrix")
    st.write(corr_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

elif page == "Advanced Analytics":
    st.title("Advanced Analytics and Insights")
    
    st.write("### Feature Importance")
    feature_importances = lr_model.coefficients
    importance_df = pd.DataFrame(list(zip(feature_columns, feature_importances)), columns=["Feature", "Importance"])
    st.write(importance_df.sort_values(by="Importance", ascending=False))
    
    st.write("### Hypothetical Feature Adjustment")
    st.write("Adjust a specific feature and see its impact on the predicted price.")
    feature_to_adjust = st.selectbox("Select Feature to Adjust", feature_columns)
    feature_value = st.slider(f"Adjust {feature_to_adjust}", min_value=0, max_value=1000, value=100)
    
    # Retrieve saved values from session state
    params = [
        st.session_state.sale, st.session_state.weight, st.session_state.resolution, st.session_state.ppi, 
        st.session_state.cpu_core, st.session_state.cpu_freq, st.session_state.internal_mem, 
        st.session_state.ram, st.session_state.rear_cam, st.session_state.front_cam, 
        st.session_state.battery, st.session_state.thickness
    ]
    feature_index = feature_columns.index(feature_to_adjust)
    params[feature_index] = feature_value
    adjusted_predicted_price = predict_price(*params)
    
    st.success(f"The predicted price with adjusted {feature_to_adjust} is: {adjusted_predicted_price:.2f}")

    st.write("### Price Sensitivity Analysis")
    st.write("Analyze how changes in each feature impact the predicted price.")
    feature_to_analyze = st.selectbox("Select Feature for Sensitivity Analysis", feature_columns)
    
    sensitivity_data = []
    for value in range(0, 101, 10):
        params = [
            st.session_state.sale, st.session_state.weight, st.session_state.resolution, st.session_state.ppi, 
            st.session_state.cpu_core, st.session_state.cpu_freq, st.session_state.internal_mem, 
            st.session_state.ram, st.session_state.rear_cam, st.session_state.front_cam, 
            st.session_state.battery, st.session_state.thickness
        ]
        feature_index = feature_columns.index(feature_to_analyze)
        params[feature_index] = value
        predicted_price = predict_price(*params)
        sensitivity_data.append((value, predicted_price))
    
    sensitivity_df = pd.DataFrame(sensitivity_data, columns=[feature_to_analyze, "Predicted Price"])
    fig, ax = plt.subplots()
    sns.lineplot(data=sensitivity_df, x=feature_to_analyze, y="Predicted Price", ax=ax)
    ax.set_title(f"Price Sensitivity Analysis for {feature_to_analyze}")
    ax.set_xlabel(feature_to_analyze)
    ax.set_ylabel("Predicted Price")
    st.pyplot(fig)
