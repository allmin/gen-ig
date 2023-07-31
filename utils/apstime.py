import numpy as np
import datetime
from dateutil import tz
import time


class TimeHelper():
    def __init__(self,time_zone=None, format='%Y-%m-%d %H:%M:%S'):
        self.time_zone = time_zone
        self.format=format

    def convert_string_date_to_datetime(self, str_date):
        format=self.format
        dtime = datetime.datetime.strptime(str_date,format)
        dtime = dtime.replace(tzinfo=tz.gettz(self.time_zone))
        return dtime

    def convert_epoch_time_to_day_of_the_week(self,epoch):
        local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(self.time_zone))
        return local_time_object.weekday()

    def convert_epoch_time_to_year(self,epoch):
        local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(self.time_zone))
        return local_time_object.year

    def convert_epoch_time_to_month(self,epoch):
        local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(self.time_zone))
        return local_time_object.month

    def convert_epoch_time_to_day_date(self,epoch):
        local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(self.time_zone))
        td =  datetime.timedelta(hours=local_time_object.hour, minutes = local_time_object.minute, seconds = local_time_object.second)
        local_day_begin = local_time_object - td
        day_date_epoch = local_day_begin.timestamp()
        return day_date_epoch
    
    def convert_epoch_time_to_day_time(self,epoch):
        local_time_object = datetime.datetime.fromtimestamp(epoch,tz=tz.gettz(self.time_zone))
        day_time = local_time_object.hour + local_time_object.minute/60
        return day_time

