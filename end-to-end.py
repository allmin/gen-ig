import os
import time
from config import system_name

from time import time

tic = time()
system_scope_dict = {'igt1':['allura01']}
env_variable = 'scope_{}'.format(system_name)
print("creating library...")    
os.system('python generate_insight_lib.py') #what to look
print("done.")
if system_name in system_scope_dict:
    scopes = system_scope_dict[system_name]
else:
    scopes = ['']
for scope in scopes:
    os.environ[env_variable]=scope
    print("scoring insights...")
    os.system('python score_insight_lib.py') #how to look
    print("scoring-done.")
    print("rewording insights...")
    os.system('python how_to_say.py') #how_to_say
    print("rewording-done.")
    print("recommending insights...")
    os.system("python recommend_insights.py") #what to say
    print("recommending insights-done.")
    print("recommending insights neurally...")
    os.system("python suggest_based_on_feedback.py") #what to say
    print("done.")
    print("picking insight pairs...")
    os.system("python pick_insight_pairs.py") #what to say
    print("done.")

toc = time()
print('elapsed_time:{}'.format((toc-tic) / 60))