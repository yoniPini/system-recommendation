# yonatan pinchas 315538074
import sys
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from scipy.sparse import data
import seaborn as sns
import pandas as pd
import numpy as np
from math import isnan, nan
df2 = pd.DataFrame({'col1': [1, 2],
                   'col2': [0.5, 0.75]},
                  index=['row1', 'row2'])

#a = pd.read_csv("data/movies_subset_small.csv", low_memory=False)
a = pd.read_csv("data/ratings-small.csv", low_memory=False)
movie_rate = [ [9,4718, 3.0], [9,7090, 3.0], [9,5418, 3.5], [9,741, 3.0], [9,924, 5.0],
    [9,110, 4.5], [9,3298, 4.0], [9,196, 4.0], [9,3033, 3.5], [9,1262, 5.0] ]

t = DataFrame(movie_rate, columns=a.columns)
a = a.append(t) 
df = a.pivot(index="userId", columns="movieId", values="rating")

print(df)





print(df)
temp = df.to_dict("index")
dict()
print(temp)
"""

temp = b.columns.copy()

c = DataFrame(b.values.T, index=b.columns, columns= b.index)

print(c, "\n")
print(type(c.index), "\n")

c = c.sort_values(by=["87640"], ascending=False)
print(c, "\n")
for e in c:
    print(e)



df_np = df.to_numpy()
#print(df_np, "\n")
df_np *= 0
df_np[np.isnan(df_np)] = 1
#print(df_np)

df2 = DataFrame(df_np, index=df.index, columns=df.columns)
print(df2)
# np.isnan(ratings_diff)
print(df)
print(df.mean(axis=1))

for index, row in a.iterrows():
    items = list(row.iteritems())
    df.at[items[0][1], items[1][1]] = items[2][1]

print(df)
            """