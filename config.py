import os

####DONOT CHANGE ANYTHING BELOW######
reserved_keywords = ["be", "measurements","measurement", "comparison"]
tense_dict = {"SP":"present",#Simple Present
                "PC":"present",#present Continuous
                "PP":"present",#Present Perfect
                "PPC":"present",#Present Perfect Continuous
                "SPa":"past",#Simple Past
                "PaC":"past",#Past Continuous
                "PaP":"past",#Past Perfect 
                "PaPC":"past",#Past Perfect Continuous
                "Future":"Future",#Future
                }
default_scope={"iphil":"user01","igt1":"allura01","rowettsleep":"user01"}

pre_defined_systems = {'iphil':{'data_file_format':'pickle', 
                                'timezone_dict':{'user01':'Europe/London', 'user02':'Europe/Amsterdam'},
                                'date_model':[3600, '2021-01-01', '2021-12-31']}, 
                       'diabetes':{'data_file_format':'pickle'},
                       'igt1':{'data_file_format':'xlsx',
                               'timezone_dict':{'allura01':'Europe/Amsterdam'},
                               'date_model':[3600, '2019-01-01', '2019-12-31']},
                       'rowettsleep':{'data_file_format':'xlsx',
                               'timezone_dict':{'user01':'Europe/Amsterdam'},
                               'date_model':[3600, '2020-01-01', '2021-12-31']},
                        'opinion_mining':{'data_file_format':'xlsx'}
                       }
                       
###########################################
#### FREE TO CHANGE BELOW #################

system_name = "igt1" # can be "igt", "opinion_mining" ,"igt1", "iphil" or "rowettsleep" or "diabetes" or 'test1' or 'focal_point'

if 'scope_{}'.format(system_name) in os.environ:
    scope_name = os.environ['scope_{}'.format(system_name)]
else: 
    if system_name in default_scope:
        scope_name = default_scope[system_name]
    else:
        scope_name = None

data_file_format = "xlsx" # the extensition of data file can be "pickle", "csv", or "xlsx"
unknown_phrase = """incomparable to""" # if not known, this has to be None
date_model = [3600, '2019-01-01', '2019-12-31'] # The scope duration of the data collection/ inisght generation (sampling interval, start date, finish date)
recommendation_count = 10
verbose = True
parallel = True


# is a system is apre-defined, the following lines override the last few lines to the predefined values.
if system_name in pre_defined_systems and "data_file_format" in pre_defined_systems[system_name]:
    data_file_format = pre_defined_systems[system_name]['data_file_format']
if system_name in pre_defined_systems and "date_model" in pre_defined_systems[system_name]:
    date_model = pre_defined_systems[system_name]['date_model']

