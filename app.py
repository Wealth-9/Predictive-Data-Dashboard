import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Housing Predictor", page_icon="")
st.title("California Housing Price Predictor")
st.write("Adjust the sliders on the left to predict themedian house valluebasedonspecific features!")

@st.cache_data
def load_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['Price'] = california.target
    return df

df = load_data()
# @st.cache_resource prevents the app from retraining the model on every click

@st.cache_resource
def train_model():
    # 'X' is the data we use to predict, 'y' is what we want to predict (Price)
    X = df.drop('Price', axis=1)
    y = df['Price']

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X,y)
    return model

model = train_model()

def get_user_inputs():
    # create sliders based on the minimum, maximum, and average values in the dataset
    med_inc = st.sidebar.slider("Median Income (in $10k)", float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()))
    house_age = st.sidebar.slider("House Age (Years)", float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()))
    ave_rooms = st.sidebar.slider("Average Rooms", float(df['AveRooms'].min()), float(df['AveRooms'].max()), float(df['AveRooms'].mean()))
    

    data = {
        'MedInc': med_inc, 'HouseAge': house_age, 'AveRooms': ave_rooms,
        'AveBedrms': df['AveBedrms'].mean(), 'Population': df['Population'].mean(),
        'AveOccup': df['AveOccup'].mean(), 'Latitude': df['Latitude'].mean(),
        'Longitude': df['Longitude'].mean()
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

st.subheader("Predicted House Value")

prediction = model.predict(input_df)
st.metric(label="Estimated Value", value=f"${prediction[0] * 100000:,.2f}")
if st.checkbox("Show Raw Dataset"):
    st.write(df.head())