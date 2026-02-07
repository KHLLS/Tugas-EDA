import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config(page_title="Weather Prediction", layout="wide")

def load_data():
    df = pd.read_csv('dataset/for_trained_weatherAUS.csv')
    df_c = pd.read_csv('dataset/weather_cleaned.csv')
    df_raw = pd.read_csv('dataset/raw_weatherAUS.csv')
    return df, df_c, df_raw

def load_model():
    model = joblib.load('model/rain_prediction_model.pkl')
    packet = joblib.load('model/rain_prediction_model_accuracy.pkl')
    return model, packet

df, df_c,df_raw = load_data()
model, packet = load_model()

le = LabelEncoder()
df_c['RainTomorrow'] = le.fit_transform(df_c['RainTomorrow'])
df_c['RainToday'] = le.fit_transform(df_c['RainToday'])

model = packet['model_obj']
acc_score = packet['accuracy']

month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',
            5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 
            9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['MonthName'] = df['Month'].map(month_map)
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['MonthName'] = pd.Categorical(
    df['MonthName'], 
    categories=month_order, 
    ordered=True
)
list_kota = sorted(df_c['Location'].unique())

list_wind = ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 
             'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'SSW']

presentage = df_c["RainTomorrow"].map({1:'Rain',0:'Bright'})
prob = presentage.value_counts()

st.sidebar.info(f"Algorithm: Random Forest - Model Accuracy: {acc_score * 100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.title("Navigation Wheater")
menu = st.sidebar.radio("Menu", ["Dashboard", "Prediction"])

if menu == "Dashboard":
    st.title("Weather Dashboard")
    st.markdown("---")
    st.title("Summary Dataset")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total City", df_c['Location'].nunique())
    c2.metric("Total Rows", f"{len(df_raw):,}")
    c3.metric("Total Columns", len(df_raw.columns))
    st.markdown("---")
    st.write(f"""
        **Information About Dataset**
        - This dataset comprises about 10 years of daily weather observations 
        from numerous locations across Australia.
        - **Source & Acknowledgements** :
        Link Dataset: (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
        The observations were gathered from a multitude of weather stations. 
        You can access daily observations from (http://www.bom.gov.au/climate/data).
        For example, you can check the latest weather observations in Canberra here: Canberra Weather.
        - Definitions have been adapted from the Bureau of Meteorology's Climate Data Online.
        Data source: Climate Data and Climate Data Online.
        Copyright Commonwealth of Australia 2010, Bureau of Meteorology.
    """)
    st.markdown("---")

    st.subheader("Probability of Rain in Australia")
    st.plotly_chart(px.pie(prob, values=prob.values, names=prob.index,title='Probability of Rain',height=500,))
    st.info("""
            **Insight:**
            - Historical data shows Australia is a dry continent. Rain only occurs on ~22%
            """)
    st.markdown("---")

    st.subheader("Raw Correlation Between Columns")
    cols = df_c.select_dtypes(include=np.number).columns
    corr = df_c[cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f',height=600, color_continuous_scale='YlGnBu'), use_container_width=True)
    st.error("""
            **Warning:**
            - Columns with a correlation greater than or equal to 90 tend to
            be redundant and often make it difficult for the model to create patterns 
            because the data content is almost the same, so we can discard those columns.
            """)
    st.markdown("---")

    st.subheader("Cleaned Correlation Between Columns")
    cols = ['MinTemp','MaxTemp','Rainfall','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm',
            'Pressure3pm','Month','RainToday']
    corr = df_c[cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto='.2f',height=600, color_continuous_scale='YlGnBu'), use_container_width=True)
    st.info("""
            **Insigh:**
            - The humidity column shows that the higher the humidity, the greater the chance of rain.
            - The max temp column shows the opposite of the humidity column,
            the lower the temperature, the higher the chance of rain.
            """)
    st.markdown("---")

    st.subheader("Most Influential Column")
    imp = pd.DataFrame({'feature': model.feature_names_in_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    st.plotly_chart(px.bar(imp.head(10), x='importance', y='feature', orientation='h'), use_container_width=True)
    st.info(f"""
    **Insight:**
    - **{imp['feature'].values[0]}** shows the strongest correlation with rain tomorrow)
    - Higher afternoon humidity = higher rain probability
    - **{imp['feature'].values[1]}** is the 2nd most important factor
    """)
    st.markdown("---")

    st.subheader("Analysis by City")
    kota = st.selectbox("Select", list_kota)
    kota_id = list_kota.index(kota)
    df_kota = df[df['Location'] == kota_id]
    tab1, tab2 = st.tabs(['Rainfall','Temperature'])
    with tab1:
        col_a, col_b = st.columns(2)
        avg_r = df_kota['Rainfall'].mean()
        col_a.info(f"Avarage Rain in {kota}: {avg_r:.2f} mm")
        rainy_month = df_kota.groupby('MonthName')['Rainfall'].mean().sort_values(ascending=False).index[0]
        col_b.success(f"rainiest month: {rainy_month}")
        month_rain = df_kota.groupby('MonthName')['Rainfall'].mean().reset_index()
        st.plotly_chart(px.line(
            month_rain, 
            x='MonthName', 
            y='Rainfall', 
            title=f"Monthly Rain Trend - {kota}"
            ), use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        avg_t = df_kota['MaxTemp'].mean()
        col1.info(f"Avarage Temp in {kota}: {avg_t:.2f} C")
        hot_month = df_kota.groupby('MonthName')['MaxTemp'].mean().sort_values(ascending=False).index[0]
        col2.success(f"Hottest Temp in {kota}: {hot_month}")
        month_temp = df_kota.groupby('MonthName')['MaxTemp'].mean().reset_index()
        st.plotly_chart(px.line(
            month_temp, 
            x='MonthName', 
            y='MaxTemp', 
            title=f"Monthly Temp Trend - {kota}"
            ), use_container_width=True)

else:
    st.title("Rain Prediction for Tomorrow")
    st.info("Fill in the weather data below to predict rain tomorrow")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pk_kota = st.selectbox("City", list_kota)
        min_t = st.number_input("Min Temp", value=0.0)
        max_t = st.number_input("Max Temp", value=0.0)
    with c2:
        h9 = st.slider("Humidity 9am", 0, 100, 0)
        h3 = st.slider("Humidity 3pm", 0, 100, 0)
        p3 = st.number_input("Pressure 3pm", value=0.0)
    with c3:
        wind_dir = st.selectbox("Wind Gust Direction", list_wind)
        wind = st.number_input("Wind Gust Speed", value=0.0)
        bln = st.selectbox("Month", range(1, 13))
    with c4:
        wind_speed9 = st.number_input('Wind Speed 9 am',value=0.0)
        wind_speed3 = st.number_input('Wind Speed 3 pm',value=0.0)

    if st.button("Prediction", use_container_width=True):
        loc_id = list_kota.index(pk_kota)
        
        input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
        input_data['Location'] = loc_id
        input_data['MinTemp'] = min_t
        input_data['MaxTemp'] = max_t
        input_data['WindGustSpeed'] = np.log1p(wind)
        input_data['Humidity9am'] = h9
        input_data['Humidity3pm'] = h3
        input_data['Pressure3pm'] = p3
        input_data['Month'] = bln
        input_data['WindSpeed9am'] = wind_speed9
        input_data['WindSpeed3pm'] = wind_speed3

        wind_col = f"WindGustDir_{wind_dir}"
        if wind_col in input_data.columns:
            input_data[wind_col] = 1
        else:
            st.warning(f"Wind direction {wind_dir} not in training data")
        input_final = input_data[model.feature_names_in_]
        
        prediction = model.predict(input_final)[0]
        prob_rain = model.predict_proba(input_final)[0][1]
        
        if prob_rain >= 0.5:
            st.error(f"Result: Rain (Probability Rain: {prob_rain * 100:.1f}%)")
        else:
            st.success(f"Result: Bright (Probability Rain: {prob_rain * 100:.1f}%)")
    st.markdown("---")
    st.title("About The Model")
    st.write("""Random forests are a combination of tree predictors such that each tree 
             depends on the values of a random vector sampled independently and with the same distribution 
             for all trees in the forest. The generalization error for forests converges a.s. to a limit as 
             the number of trees in the forest becomes large.""")

