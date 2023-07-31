import pandas as pd
from config import system_name

entdata = pd.read_csv(f'systems/{system_name}/data_{system_name}.csv')
user_unique = list(entdata.user.unique())
context1_unique = list(entdata.context1.unique())
context2_unique = list(entdata.context2.unique())