# Copyright (c) 2024 Copyright Holder All Rights Reserved.
# @author: diego garcia y jesus remon

import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pickle
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import json


# Load the Shapefile and the districts data
districts_shp = gpd.read_file("data/geo_chicago.shp")
district_data = pd.read_csv("data/dataframe.csv")

# Function that creates the color gradient
def get_color(value):
    r = int(max(0, min(255, 255 * (1 - value / 100))))
    g = int(max(0, min(255, 255 * (value / 100))))
    b = 0
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# Function that gets the number and crime scale of the district
def row_info(df, name):
    filtered_df = df[df['District Name'] == name]
    if not filtered_df.empty:
        number = filtered_df['District'].values[0]
        return number
    else:
        return None

# Function that gets the average crime_score of the district
def get_crime(number):
    filtered_data = district_data[district_data['District'] == number]
    crime_score = filtered_data['Crime Score'].mean()
    return crime_score

# Function that comparess between 2 dates
def compare_dates(date1, date2):
    date1 = datetime.strptime(date1, '%m/%d/%Y %I:%M:%S %p')
    date2 = datetime.strptime(date2, '%m/%d/%Y %I:%M:%S %p')
    if date1 >=  date2:
        return True
    return False

# Function to filter the data by dates
def filter_district_data(district_data, number, date2, date3=None):
    data = district_data[district_data['District'] == number] 
    if date3 is None:
        selected_data = pd.DataFrame(columns=data.columns)
        for index, row in data.iterrows():
            date1 = row['Date']
            if compare_dates(date1, date2):
                selected_data = pd.concat([selected_data, pd.DataFrame(row).T], ignore_index=True)
    else:
        selected_data = pd.DataFrame(columns=data.columns)
        for index, row in data.iterrows():
            date1 = row['Date']
            if compare_dates(date1, date2) and compare_dates(date3, date1):
                selected_data = pd.concat([selected_data, pd.DataFrame(row).T], ignore_index=True)
                
    return selected_data

###################################################################################################

def choose_district(district, window, data, value):
    data_reduced = data[data["District Name"] == district]

    data_reduced['Date'] = pd.to_datetime(data_reduced['Date'])
    data_reduced.set_index('Date', inplace=True)

    data_reduced = data_reduced[[value]].resample(window).mean()

    return data_reduced


def ARIMA_pred(dataframe, delta, value):
    df = dataframe

    model = ARIMA(df[value], order=(1, 1, 1))
    model_fit = model.fit()
    
    forecast = model_fit.get_forecast(steps=delta)
    forecast_summary = forecast.summary_frame()
    
    forecasted_values = forecast_summary['mean'].tolist()
    
    return model_fit

def ARIMA_pred2(dataframe, delta):
    df = dataframe

    print(df.head())

    # Fitting an ARIMA model to the differenced data
    model = ARIMA(df["response_time_minutes"], order=(1, 1, 1))
    model_fit = model.fit()
    
    forecast = model_fit.get_forecast(steps=delta)
    forecast_summary = forecast.summary_frame()
    
    forecasted_values = forecast_summary['mean'].tolist()
    
    return model_fit


def forecast(model_fit, delta = 3):
    
    forecast = model_fit.get_forecast(steps=delta)
    forecast_summary = forecast.summary_frame()
    
    forecasted_values = forecast_summary['mean'].tolist()
    
    return forecast_summary, forecasted_values


def save_ARIMA(model, path = 'arima_model.pkl'):
    """
    Function to store the ARIMA model into a pickle format.
    - model: The ARIMA model
    - path: The path where the ARIMA model will be saved
    """

    with open(path, 'wb') as pkl:
        pickle.dump(model, pkl)
        
    
def load_ARIMA(path):
    with open(path, 'rb') as pkl:
        loaded_model = pickle.load(pkl)
    
    return loaded_model

def plotting(df, forecast_summary, district, title, title_2, value, axis):
    """
    plotting
    - df: dataframe with the crime data
    - forescast_summary: dataframe with the forecaste prediction
    """

    # Create a figure with plotly.graph_objects
    fig = go.Figure()

    # Add historical crime score data
    fig.add_trace(go.Scatter(x=df.index, y=df[value], mode='lines+markers', name=title))

    # Add forecasted crime score data
    fig.add_trace(go.Scatter(x=forecast_summary.index, y=forecast_summary['mean'], mode='lines+markers', name=title_2, line=dict(dash='dash')))

    # Add confidence interval as a shaded area
    fig.add_trace(go.Scatter(x=forecast_summary.index.tolist() + forecast_summary.index[::-1].tolist(),
                             y=forecast_summary['mean_ci_upper'].tolist() + forecast_summary['mean_ci_lower'][::-1].tolist(),
                             fill='toself',
                             fillcolor='rgba(255, 0, 0, 0.1)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name='Confidence Interval 95%'))

    # Update layout
    fig.update_layout(title=f"{axis} <br><sup style='color:#d4d4d4;'>Distric: " + district + "</sup>",
                      xaxis_title='Date',
                      yaxis_title= axis,
                      legend_title='Legend',
                      template='plotly_white')

    # Show the figure
    return fig

def cluster_analysis(District, dataframe, numb_cl = 5):

    df = dataframe  

    # Preprocess the data: Group by district and calculate relevant metrics
    data_for_clustering = df.groupby('District Name').agg({
        'Crime Score': 'mean',  # Average crime score
        'Arrest': 'mean',       # Arrest rate
    }).reset_index()

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_for_clustering.drop('District Name', axis=1))

    # Perform KMeans clustering
    kmeans = KMeans(numb_cl, random_state=42)  # Adjust n_clusters based on your analysis
    data_for_clustering['Cluster'] = kmeans.fit_predict(scaled_features)

    cluster_of_input = data_for_clustering.loc[data_for_clustering['District Name'] == District, 'Cluster'].values[0]
    
    similar_districts = data_for_clustering[data_for_clustering['Cluster'] == cluster_of_input]['District Name'].tolist()
        
    return similar_districts, data_for_clustering

def internet_search_json(query, serper_key, search = "search", country = "us"):
    '''
    This function returns the internet search of the query, using Serper
    API and returns the response of the search results in JSON format.

    * query (str): Is the Question to do research from the user
    * serper_key (str): Is the API Key from serper to do ther search
    * search (str): Is what kind of search to perfrom, it can be "search" or "news"
    * country (str): Is the country to perform the search from

    Courtesy of Mr. Jesus Remon
    '''
    url = "https://google.serper.dev/" + search

    payload = json.dumps({
      "q": query,
      "gl": country
    })

    headers = {
      'X-API-KEY': serper_key,
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    return response_data



#### STREAMLIT

st.set_page_config("Arkham", layout="wide")

flow = False


with st.sidebar:
    st.subheader("Parameters for selection")

    # Dropdown menu for district selection
    with st.form("my_form"):
        district = st.selectbox('District', district_data['District Name'].unique())
        resample_win = st.selectbox('Window of Forecast', ["Monthly", "Weekly", "Daily"])

        delta_value = st.slider("Select forecast time", 1, 10)

        submitted = st.form_submit_button("Submit")
        if submitted:
            flow = True

# RANKING DISTRICTS
district_mean_scores = district_data.groupby('District Name')['Crime Score'].mean().reset_index()
sorted_districts = district_mean_scores.sort_values(by='Crime Score').reset_index(drop=True)
district_rank = sorted_districts[sorted_districts['District Name'] == district].index[0] + 1
average_crime_score = district_mean_scores['Crime Score'].mean()


if flow:


    number = row_info(district_data,district)
    crime_scale = get_crime(number)
    geometry = districts_shp[districts_shp['dist_num'] == str(number)]['geometry'].values[0]

    st.toast("Neural Network loading")

    st.markdown(f"## {district}")

    col1, col2, colend = st.column([1,1,10])

    col1.markdown(f"##### Rank: {district_rank}/23")
    col2.markdown(f"##### Crime Score: {round(crime_scale,2)}")


    addon = "th"

    if number == 1:
        addon = "st"
    elif number == 1:
        addon = "nd"
    
    st.page_link("https://home.chicagopolice.org/category/community-alerts/?tag=" + str(number) + addon + "-district", label="Community Alerts", icon="📁")
    st.page_link("https://home.chicagopolice.org/category/wanted/?tag="+ str(number) + addon +"-district", label="Community Fugitives", icon="📁")
    st.divider()

    data = district_data[district_data['District'] == number]

    arrest_rate_overall = (data['Arrest'].mean()) * 100
    
    criminals = data['Arrest'].value_counts().to_list()
    criminals = criminals[1]

    res_dict = {"Monthly":"ME", "Weekly":"W", "Daily":"D"}

    value_day = {"Monthly":"month", "Weekly":"week", "Daily":"day"}

    day_value = value_day[resample_win]

    resample_win = res_dict[resample_win]

    ## SARIMA MODEL

    df = choose_district(district, resample_win, district_data, "Crime Score")

    model_ARIMA = ARIMA_pred(df, delta_value, "Crime Score")

    forecast_df, prediction = forecast(model_ARIMA, delta_value)

    st.toast("Neural Network finished")

    # Create a folium map centered around Chicago
    mapa = folium.Map(location=[41.8781, -87.6298], zoom_start=10)


    # Add district to the map
    folium.GeoJson(
        geometry,
        name='geojson',
        style_function=lambda feature: {
            'fillColor': get_color(crime_scale), 
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.5,
        }
    ).add_to(mapa)


    colMapa, col1, col2 = st.columns([4,1,1])

    with colMapa:
        with st.spinner("Loading Map"):
            folium_static(mapa)

     # Getting the number of crimes for the metrics
    last_date = datetime.strptime(data['Date'].iloc[-1], '%m/%d/%Y %I:%M:%S %p')
    date1 = last_date - timedelta(days=2) # 48hr before
    date2 = last_date - timedelta(days=1) # 24hr before
    date3 = last_date - timedelta(days=7) # 7 days before
    date4 = last_date - timedelta(days=14) # 14 days before

    date1 = date1.strftime('%m/%d/%Y %I:%M:%S %p') 
    date2 = date2.strftime('%m/%d/%Y %I:%M:%S %p')
    date3 = date3.strftime('%m/%d/%Y %I:%M:%S %p')
    date4 = date4.strftime('%m/%d/%Y %I:%M:%S %p')

    data_last24 = filter_district_data(district_data, number, date2)
    data_prev24 = filter_district_data(district_data, number, date1, date2)
    data_last7 = filter_district_data(district_data, number, date3)
    data_prev7 = filter_district_data(district_data, number, date4, date3)

    # Calculate the delta 
    delta_24 = len(data_last24) - len(data_prev24)
    delta_7 = len(data_last7) - len(data_prev7)
    delta_crim = crime_scale - prediction[0]

    average_response_time_by_district = district_data.groupby('District Name')['response_time_minutes'].mean()

    district_time = average_response_time_by_district.loc[district]
    print(district_time)
    avg_time = np.mean(average_response_time_by_district)
    print(avg_time)
    delta_time = district_time - avg_time

    # Output the results
    with st.spinner("Loading Metrics"):
        col1.metric("Arrest Rate: (%)", round(arrest_rate_overall,2))
        col2.metric('Arrested:', criminals)
        col1.metric('24hr Change:', len(data_last24), delta=delta_24,delta_color="inverse")
        col2.metric('Last 7 days:', len(data_last7), delta=delta_7,delta_color="inverse")
        col2.metric(f'Crime score next {day_value}:', round(prediction[0],2), delta=round(delta_crim,2),delta_color="inverse")
        col1.metric("Response Time (min)", round(district_time,2), delta=round(delta_time,2), delta_color="inverse")

    # Count the occurrences of each group type
    group_counts = data['Crime Group'].value_counts().sort_index()
    # Moving the first row last
    first_row = group_counts.iloc[[0]]
    group_counts = group_counts.drop(group_counts.index[0])
    group_counts = pd.concat([group_counts, first_row])
    custom_order = group_counts.index.tolist()
    num_colors = len(group_counts)

    # Define a custom color scale from red to green
    custom_palette = px.colors.sequential.Inferno_r

    # Select a subset of colors from the color scale
    colors = custom_palette[0:int(num_colors)]

    # Create a pie chart using Plotly Express with custom ordering and colors
    fig = px.pie(names=group_counts.index, 
                values=group_counts.values, 
                title="Crime Group Distribution <br><sup style='color:#d4d4d4;'>Distric: " + district + "</sup>", 
                category_orders={"names": custom_order},
                color_discrete_sequence=custom_palette)

    col1, colmed, col2 = st.columns([4,1,4])

    # Show the pie chart
    col1.plotly_chart(fig)

    # Graph with forecast of the crime score
    col2.plotly_chart(plotting(df, forecast_df, district, 'Historical Crime Score', 'Forecasted Crime Score', 'Crime Score', "Crime Score"))


    ### Mostrar los distritos con correlación similar en crimen
    value_cluster, correlation_df = cluster_analysis(district, district_data, 5)

    ### Calcular la previsión de tiempo de respuesta
    df_res = choose_district(district, resample_win, district_data, "response_time_minutes")

    model_ARIMA_res = ARIMA_pred2(df_res, delta_value)

    forecast_df_res, prediction_res = forecast(model_ARIMA_res, delta_value)

    col1.plotly_chart(plotting(df_res, forecast_df_res, district, 'Historical Response Time', 'Forecasted Response Time', 'response_time_minutes', "Response Time"))

    

    value_cluster.remove(district)

    filtered_df = district_data[district_data['District Name'].isin(value_cluster)]

    filtered_df = filtered_df[["District Name", "Crime Score"]]

    data = filtered_df.groupby("District Name")["Crime Score"].mean()

    data = pd.DataFrame(data)



# Plotting
    
    fig = px.bar(data, x=data.index, y="Crime Score", title="Average Crime Score of similar districts <br><sup style='color:#d4d4d4;'>Distric: " + district + "</sup>")

    fig.update_layout(xaxis_title='Districts',
                      yaxis_title= 'Avg. Crime Score',
                      legend_title='Districts',
                      template='plotly_white')

    col2.plotly_chart(fig)

    bar_fig = px.bar(sorted_districts, x='District Name', y='Crime Score', 
                    title="District Crime Score Ranking <br><sup style='color:#d4d4d4;'>Distric: " + district + "</sup>",
                    labels={'Crime Score': 'Crime Score'},
                    color='Crime Score',
                    color_continuous_scale='Turbo',
                    orientation='v')

    bar_fig.add_hline(y=average_crime_score, line_dash="dot", line_color="red", annotation_text=f'Average: {average_crime_score:.2f}',
                    annotation_position="bottom right")

    col1.plotly_chart(bar_fig)

    #################

    data_reduced = district_data

    data_reduced['Date'] = pd.to_datetime(data_reduced['Date'])
    data_reduced.set_index('Date', inplace=True)

    data_reduced = data_reduced[data_reduced['District Name'].isin(value_cluster)]


    data_reduced = data_reduced.groupby('District Name')[['Crime Score']].resample('W').mean()

    
    # Reset the index of the DataFrame so 'Districts' and 'Dates' become columns
    df_reset = data_reduced.reset_index()    
    
    # Assuming df_reset now contains columns ['Districts', 'Dates', 'Crime Score']
    fig = px.density_heatmap(df_reset, x='Date', y='District Name', z='Crime Score', color_continuous_scale='Viridis')
    
    # Optional: Improve layout
    fig.update_layout(
        title="Crime Score Evolution in Similar Districts <br><sup style='color:#d4d4d4;'>Distric: " + district + "</sup>",
        xaxis_title='Date',
        yaxis_title='District',
        xaxis={'type': 'category'},  # Use this if you want discrete dates on the x-axis
    )

    col2.plotly_chart(fig)

    

    ### Grafica de previsión de crecimiento de carga policial
    # - Calcular una métrica para determinar la carga policial de 0 a 10
    # - Crear Gráfica con ARIMA sobre la previsión de crecimiento policial

    ### Escribir recomendaciones con IA sobre reducciones de crimen
    # - Caracteristicas sobre el barrio en cuestión
    # - Respuestas personalizadas para la zona

    ### Posibilidad de crear un PDF

    links = []
    
    for i in range(2):
      results = internet_search_json(f"Crime in {district}, Chicago", "9ade9c0d3878580c74d4990b313d1e269214bc5b", "news", "us")
      links.append(results)
    
    print(links)

    with st.expander("News Articles"):
        st.write(links[0]['news'][0]["title"])
        st.text(links[0]['news'][0]['snippet'])
        st.link_button("Article", links[0]['news'][0]['link'])
        st.divider()
        st.write(links[0]['news'][1]["title"])
        st.text(links[0]['news'][1]['snippet'])
        st.link_button("Article", links[0]['news'][1]['link'])
        st.divider()
        st.write(links[0]['news'][2]["title"])
        st.text(links[0]['news'][2]['snippet'])
        st.link_button("Article", links[0]['news'][2]['link'])






else:
    st.info("Select the data to simulate")
