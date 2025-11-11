from django.shortcuts import render

# Create your views here.
import requests # This libraray helps us to fetch data from AΡΙ
import pandas as pd #for handling and analysing datas
import numpy as np #for numerical operations
import pytz
import os
from sklearn.model_selection import train_test_split #to split data into training and testing sets
from sklearn.preprocessing import LabelEncoder #to convert catogerical data into numericals values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #models for classification and regression
from sklearn.metrics import mean_squared_error #to measure the accuracy of our predictions
from datetime import datetime, timedelta #to handle date and time
from pytz import timezone #to handle timezones

API_KEY = '6295fda2b480ed871fa2ac9a96456814' #replace with your actual API KEY
BASE_URL = 'https://api.openweathermap.org/data/2.5'# base url for making Api key requestsBASE_URL

#1 Fetch Current Weather Data
def get_current_weather(city):
  url = f"{BASE_URL}/weather?q={city}&appid={API_KEY}&units=metric" # construct the API request URL
  response = requests.get(url) # send the get request to API
  data = response.json()
  if response.status_code == 200: # Check if the API call was successful
    return {
        "city": data["name"],
        "current_temperature": round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "temp_min": round(data["main"]["temp_min"]),
        "temp_max": round(data["main"]["temp_max"]),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_Gust_speed': data['wind']['speed'],

        'clouds': data['clouds']['all'],
        'Visibility': data['visibility'],
    }
  else:
    print(f"Error: Could not get weather data for {city}. Status code: {response.status_code}")
    return None # Return None if the API call was not successful



#2 Read historical Data
def read_historical_data(filename):
  df = pd.read_csv(filename) # load csv file into dataframe
  df = df.dropna() # remove rows with missing values
  df = df.drop_duplicates()
  return df


#3 Prepare data for Training
def prepare_data(data):
  le = LabelEncoder() #create LabelEncoder instance
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  #define the feature variable and target variables

  X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] #feature variables
  y = data['RainTomorrow'] #target variable
  return X, y, le #return feture variable, target variable and the label encode

#4 Train Rain Prediction Model
def train_rain_model(x, y):
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train) #train the model

  y_pred = model.predict(X_test) #to make predictions on test set

  print("Mean Squared Error for Rain Model")

  print(mean_squared_error(y_test, y_pred))

  return model


#5Prepare regression data
def prepare_regression_data(data, feature):
  X, y = [], [] #initialize list for feature and target values

  for i in range(len(data)- 1):
    X.append(data[feature].iloc[i])

    y.append(data[feature].iloc[i+1])

  X = np.array(X).reshape(-1, 1)
  y = np.array(y)
  return X, y


#6 Train Regression Model
def train_regression_model(x, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(x, y)
  return model


#7 Predict Future
def predict_future(model, current_value):
  prediction = [current_value]

  for i in range(5):
    next_value = model.predict(np.array([[prediction[-1]]]))
    prediction.append(next_value[0])

  return prediction[1:]



#8 Weather analysis function


def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)

        # load historical data
        csv_path = os.path.join('C:\\Users\\hp\\OneDrive\\Desktop\\MachineLearningProject2\\weather.csv')
        historical_data = read_historical_data(csv_path)

        # prepare and train the rain prediction model
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        # map wind direction to campass points
        # There's a slight error here: wind_gust_dir is likely in degrees, 
        # so you probably want to use the raw value instead of modulo 360 if it's already between 0 and 360.
        wind_deg = current_weather['wind_gust_dir'] % 360 
        
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360) # Added wrap-around for N
        ]

        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)[0]
        
        # Add handling for the 348.75 to 360 range which should also be 'N'
        if wind_deg >= 348.75 and wind_deg < 360:
             compass_direction = "N"


        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            # Encode wind direction, fallback to -1 if not in classes
            'WindGustDir': le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1, 
            'WindGustSpeed': current_weather['wind_Gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temperature']
        }

        current_df = pd.DataFrame([current_data])

        # Predict rain
        rain_prediction = rain_model.predict(current_df)[0]

        # prepare regression model for temperature and humidity
        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        # predict future temperature and humidity
        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # prepare time for future predictions
        timezone = pytz.timezone("Asia/Karachi")
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        # The original code had (hours-i), assuming it meant (hours=i) or just (i)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # store each value separately
        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        # Pass data to tempelate
        context = {
            'location': city,
            'current_temp': current_weather['current_temperature'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'Feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],

            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %Y"),

            'wind': current_weather['wind_Gust_speed'],
            'pressure': current_weather['pressure'],

            'visibility': current_weather['Visibility'],

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            # Note: You were rounding the temps to different decimal places, I kept that logic but it might be a typo.
            'temp1': f"{round(temp1, 1)}",
            'temp2': f"{round(temp2, 2)}",
            'temp3': f"{round(temp3, 3)}",
            'temp4': f"{round(temp4, 4)}",
            'temp5': f"{round(temp5, 5)}",

            'hum1': f"{round(hum1, 1)}",
            'hum2': f"{round(hum2, 2)}",
            'hum3': f"{round(hum3, 3)}",
            'hum4': f"{round(hum4, 4)}",
            'hum5': f"{round(hum5, 5)}",
        }

        return render(request, 'weather.html', context)
    
    return render(request, 'weather.html')  


