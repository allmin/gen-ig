import pandas as pd
from utils import apstime

def preprocess(data, dummy = False, time_zone=None):
    th = apstime.TimeHelper(time_zone)
    if not dummy:
        data = data.dropna(subset=['date'])
        data['date'] = [int(th.convert_string_date_to_datetime(t).timestamp()) for t in data['date']]
    data['day_date'] = [th.convert_epoch_time_to_day_date(i) for i in data['date']]
    data['day_time'] = [th.convert_epoch_time_to_day_time(i) for i in data['date']]
    data['dayofweek'] = [th.convert_epoch_time_to_day_of_the_week(i) for i in data['date']]
    data['year'] = [th.convert_epoch_time_to_year(i) for i in data['date']]
    data['month'] = [th.convert_epoch_time_to_month(i) for i in data['date']]
    data['quarter'] = [((i-1)//3+1) for i in data['month']]
    data['week'] = data.date.apply(pd.to_datetime, unit='s')
    data['week'] = data['week'].dt.strftime('%U')
    data['available'] = True
    ### any other preprocessing comes below if not dummy###
    #if not dummy:
    return data