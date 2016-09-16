dir ='F:/MachineLearningNanoDegree/MLP4Smartcab/smartcab/'
filename = dir + 'gridsearch_20160914-235951.log'

import pandas as pd
import seaborn as sns
sns.set()

df_long = pd.read_csv(filename)
#df = df_long.pivot("Alpha", "Gamma", "Last20RedLightViolations")
df = df_long.pivot("Alpha", "Gamma", "Last20PlannerNoncompliance")
#df = df_long.pivot("Alpha", "Gamma", "SuccessRate")
#cmap="YlGnBu"
sns.heatmap(df, annot=True, fmt=".2g", linewidths=.5)