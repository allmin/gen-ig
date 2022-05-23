import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('analysis.xlsx',engine='openpyxl')
plt.plot(df['total'],df['parallel'])
plt.plot(df['total'],df['serial'])
plt.legend(['Parallel','Serial'])
plt.xlabel('Total number of Insights')
plt.ylabel('Time in minutes')

plt.show()