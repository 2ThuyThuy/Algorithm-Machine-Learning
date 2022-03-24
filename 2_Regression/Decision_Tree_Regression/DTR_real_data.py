import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

col_names = ['date','avgtemp', 'mintemp', 'pp', 'snow', 'wind-dir', 'wind-speed', 'wind-gut', 'air-pressure', 'sunshine', 'dummy']
        
daily_weather_df = pd.read_csv('KCQT0.csv',sep=',',names=col_names,header=None)

#Delete irrelevant cols
del daily_weather_df['dummy']
del daily_weather_df['air-pressure']
del daily_weather_df['wind-speed']
del daily_weather_df['snow']
del daily_weather_df['wind-dir']
del daily_weather_df['date']
del daily_weather_df['mintemp']

#print(daily_weather_df.head())
#get temperatures of the last 365 days
daily_temp = daily_weather_df['avgtemp'].to_numpy()[-365:]
#print(f"the first 10 entries in daily_temp are {daily_temp[:10]}")

mean_temp = np.nanmean(daily_temp)  #find the meand ignore the NaN values
variance = np.nanvar(daily_temp)
#print(f"Mean tempreature is {mean_temp} celsius")
#print(f"Variance is {variance}")
#print(daily_weather_df.isnull().sum())