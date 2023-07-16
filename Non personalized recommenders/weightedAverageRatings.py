import matplotlib.pyplot as plt
import seaborn as sns

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata1.csv', low_memory=False)

# Print the first three rows
print(metadata.head(3))

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)

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
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 10 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))


plt.figure(figsize=(16,6))

ax = sns.barplot(x=q_movies['score'].head(10), y=q_movies['title'].head(10), data=q_movies, palette='deep')

plt.xlim(7.5, 8.5)
plt.title('"Best" Movies by TMDB Votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('best_WAR_movies.png')

