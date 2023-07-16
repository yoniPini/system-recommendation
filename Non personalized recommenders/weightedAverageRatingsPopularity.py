import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata1.csv', low_memory=False)

# Print the first three rows
#print(metadata.head(3))

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)
print("***")
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape)
print(metadata.shape)

####### weighted average ratings #######

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['weighted_average'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
#q_movies = q_movies.sort_values('popularity', ascending=False)

min_max_scaler = preprocessing.MinMaxScaler()
movies_scaled = min_max_scaler.fit_transform(q_movies[['weighted_average', 'popularity']])
movies_norm = pd.DataFrame(movies_scaled, columns=['weighted_average', 'popularity'])
movies_norm.head()

q_movies[['norm_weighted_average', 'norm_popularity']] = movies_norm

q_movies['score'] = q_movies['norm_weighted_average'] * 0.5 + q_movies['norm_popularity'] * 0.5
movies_scored = q_movies.sort_values(['score'], ascending=False)
#movies_scored[['title', 'norm_weighted_average', 'norm_popularity', 'score']].head(20)

scored = q_movies.sort_values('score', ascending=False)
#Print the top 10 movies
print(scored[['title', 'vote_count', 'vote_average', 'popularity', 'score']].head(10))

plt.figure(figsize=(16,6))

ax = sns.barplot(x=scored['score'].head(10), y=scored['title'].head(10), data=scored, palette='deep')

#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('best_WARPpp_movies.png')