import matplotlib.pyplot as plt
import seaborn as sns

# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata1.csv', low_memory=False)

metadata["popularity"] = pd.to_numeric(metadata["popularity"])
print(metadata.shape)

popular = metadata.sort_values('popularity', ascending=False)

#Print the top 10 movies
print(popular[['title', 'popularity']].head(10))

plt.figure(figsize=(16,6))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['title'].head(10), data=popular, palette='deep')

plt.title('"Most Popular" Movies by TMDB Votes', weight='bold')
plt.xlabel('Popularity Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('best_P_movies.png')
