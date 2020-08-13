import json 
import pandas as pd
import os
import numpy as np
import datetime
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# get stress data
def calculate_date (row):
    return str(row['time'])[0:10]

def get_daily_stress(file):
    '''get daily stress of the given individual, calculate the mean stress level of each day'''
    
    # get user ID
    user = file[-2:]
    
    # convert the json file into dataframe
    with open('dataset/EMA/response/Stress'+'/' +str(file)+'.json') as f:
        data = json.load(f)
    
    stress_dict = {}
    stress_dict['stress_level'] = []
    stress_dict['location'] = []
    stress_dict['time'] = []
    
    for d in data:
        if ('null' not in d and 'level' in d):
            stress_dict['stress_level'].append(int(d['level']))
            stress_dict['location'].append(d['location'])
            stress_dict['time'].append(datetime.datetime.fromtimestamp(d['resp_time']).isoformat().replace('T',' '))
    
    stress_df = pd.DataFrame.from_dict(stress_dict)
    stress_df = stress_df.sort_values(by=['time'])
    if len(stress_df) > 0:
        stress_df['time'] = stress_df['time'].apply(pd.Timestamp)
        stress_df['date'] = stress_df.apply (lambda row: calculate_date(row),axis=1)
        stress_df = stress_df.drop(['location','time'], axis=1)
        stress_df = stress_df.groupby(['date']).mean()
        stress_df.reset_index(level=0, inplace=True)
        stress_df['user'] = user
        stress_df = stress_df.reindex(columns=['user', 'date', 'stress_level'])
        if not os.path.exists('daily_stress'):
            os.makedirs('daily_stress')
        stress_df.to_csv ('daily_stress/daily_stress_u' + user +'.csv', index = False)


# get sensing data
sensing_path = ['dataset/sensing/activity/activity_u',
                'dataset/sensing/audio/audio_u'
               ]
sensing_types = [['stationary', 'walking', 'running', 'activity_unknown'], 
                 ['silence', 'voice', 'noise', 'audio_unknown']
                ]
behavior_names = ['activity', 'audio']

# 0	Stationary
# 1	Walking
# 2	Running
# 3	Unknown

def get_daily_cumu_df(df, isNormalized):
    daily_df = df
    for i in range(len(daily_df) - 1):
        daily_df.iloc[i + 1] = daily_df.iloc[i + 1] + daily_df.iloc[i]
    if isNormalized:
        max_val = daily_df.iloc[len(daily_df) - 1]
        if max_val != 0:
            daily_df = daily_df.astype('float64')
            for i in range(len(daily_df)):
                daily_df[i] = np.float64(daily_df[i]).item() / np.float64(max_val).item()
    daily_df = daily_df.sort_index()
    return daily_df

def get_sensing_table(behavior_index, good_day_threshold):
    # only look at users with EMA data
    behavior_table_all = pd.DataFrame()
    filtered_count = 0
    all_data_points = 0
    for file in sorted(os.listdir('daily_stress')):
        if file[0] != '.':
            user_id = file[-6:-4]
            print()
            print(user_id)
            behavior_user = pd.read_csv(sensing_path[behavior_index] + user_id + '.csv')
            behavior_user.columns = ['timestamp', 'inference']
            behavior_user['time'] = pd.to_datetime(behavior_user.timestamp, unit='s')
            behavior_user = behavior_user.set_index('time')
            behavior_user = behavior_user.loc[~behavior_user.index.duplicated(keep='first')]
            behavior_user = behavior_user.sort_index()

            # get all days for this user
            dates_df = behavior_user.resample('D').count() 
            days = dates_df.index

            for day in days:
                print('day',day)
                df_user_day = pd.DataFrame(columns = sensing_types[behavior_index])
                behavior_user_day = behavior_user[day.strftime("%Y-%m-%d")]
                behavior_user_day = behavior_user_day.resample('S').pad()
                day_minutes = day + pd.to_timedelta(np.arange(1440), 'm')
                available_counts_0 = []
                available_counts_1 = []
                available_counts_2 = []
                available_counts_3 = []
                available_counts = [available_counts_0, available_counts_1, available_counts_2, available_counts_3]
                available_minutes = behavior_user_day.resample('T').count().index
                for minute in available_minutes:
                    behavior_user_minute = behavior_user_day[minute.strftime("%Y-%m-%d %H:%M")]
                    behavior_user_minute_grouped = behavior_user_minute.groupby('inference').count()
                    for i in range(len(sensing_types[behavior_index])):
                        if i in behavior_user_minute_grouped.index:
                            available_counts[i].append(behavior_user_minute_grouped.loc[i]['timestamp'])
                        else:
                            available_counts[i].append(0)
                df_user_day['time'] = available_minutes
                for i in range(len(sensing_types[behavior_index])): 
                    df_user_day[sensing_types[behavior_index][i]] = available_counts[i]
                df_user_day = df_user_day.set_index('time')
                df_user_day = df_user_day.reindex(day_minutes, fill_value=0)
                for i in range(len(sensing_types[behavior_index])):
                    df_user_day[sensing_types[behavior_index][i] + '_cumu'] = get_daily_cumu_df(df_user_day[sensing_types[behavior_index][i]].copy(), False)
                    df_user_day[sensing_types[behavior_index][i] + '_cumu_normal'] = get_daily_cumu_df(df_user_day[sensing_types[behavior_index][i]].copy(), True)

                # check if this day has more than 19 hours of sensing data
                total_sensing = 0
                for i in range(len(sensing_types[behavior_index])):
                    total_sensing += df_user_day.iloc[len(df_user_day)-1][sensing_types[behavior_index][i] + '_cumu']
                if total_sensing >= good_day_threshold*60*60:
                    df_user_day = df_user_day.reset_index()
                    df_user_day['time'] = df_user_day['index']
                    df_user_day = df_user_day.drop(['index'], axis=1)
                    df_user_day['user'] = user_id
                    behavior_table_all = pd.concat([behavior_table_all, df_user_day])
                else:
                    filtered_count +=1
                all_data_points+=1
            print('filter', filtered_count)
            print(all_data_points)
            print(datetime.datetime.now()) 
    behavior_table_all.to_csv(behavior_names[behavior_index] + '.csv', index = False)
    

# statistical method
def get_p_values(df):
    df1 = df.dropna()._get_numeric_data()
    coeffmat = np.zeros((df1.shape[1], df1.shape[1]))
    pvalmat = np.zeros((df1.shape[1], df1.shape[1]))

    for i in range(df1.shape[1]):    
        for j in range(df1.shape[1]):        
            corrtest = pearsonr(df1[df1.columns[i]], df1[df1.columns[j]])  

            coeffmat[i,j] = corrtest[0]
            pvalmat[i,j] = corrtest[1]

    dfcoeff = pd.DataFrame(coeffmat, columns=df1.columns, index=df1.columns)

    dfpvals = pd.DataFrame(pvalmat, columns=df1.columns, index=df1.columns)
    return dfpvals


# filtering
def get_filtered_ema_from_table(NUM_GOOD_DAYS_REQUIRED, current_ema):
    '''
    only keeps ema data with strictly more than "NUM_GOOD_DAYS_REQUIRED" days of behavioral data 
    for all activities
    '''
    table0 = current_ema

    table0_filtered = table0.loc[table0.stationary_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.walking_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.running_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.activity_unknown_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.silence_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.voice_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.noise_num_good_days >= NUM_GOOD_DAYS_REQUIRED]
    table0_filtered = table0_filtered.loc[table0_filtered.audio_unknown_num_good_days >= NUM_GOOD_DAYS_REQUIRED]

    new_ema = table0_filtered
    return new_ema
