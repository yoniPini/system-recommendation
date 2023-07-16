# yonatan pinchas 315538074
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []

        self.user_rated_movies = {}
        self.movies_id = {}
        self.movies_name = {}

    def update_user_rated_movies(self, ratings: DataFrame):
        if len(self.user_rated_movies) > 0:
            return
        self.user_rated_movies = ratings.to_dict("index")
    
    def update_movies_id(self, movies: DataFrame):
        if len(self.movies_id) > 0:
            return
        for id, title in zip(movies["movieId"], movies["title"]):
            self.movies_id[id] = title
            self.movies_name[title] = id

    def create_fake_user(self,ratings : DataFrame):
        movie_rate = [[283238,1, 5.0], [283238,2, 5.0], [283238,21, 4.5],
                    [283238,48, 3.0], [283238,153, 5.0], [283238,186, 1.5],
                    [283238,193, 1.0], [283238,215, 0.0], [283238,222, 0.5],
                    [283238,364, 4.0], [283238,367, 5.0], [283238,381, 1.5],
                    [283238,474, 4.0], [283238,491, 4.0], [283238,648, 4.5],

                    [283238,647, 4.0], [283238,653, 4.0], [283238,954, 1.5],
                    [283238,733, 4.0], [283238,736, 4.0], [283238,926, 1.5],
                    [283238,836, 4.0], [283238,910, 4.0], [283238,923, 1.5],
                    [283238,1240, 4.0], [283238,1254, 4.0], [283238,1041, 1.5],
                    [283238,1373, 4.0], [283238,1291, 4.0], [283238,1090, 0.5],
                    ]
        
        return ratings.append(DataFrame(movie_rate, columns=ratings.columns))

    def create_user_based_matrix(self, data):
        if len(data) > 1: self.update_movies_id(data[1])
        ratings = data[0]
        ratings = DataFrame(ratings) # TODO remove
        
        #for adding fake user
        ratings = self.create_fake_user(ratings)

        # reorder ratings as matrix dataFrame (kind of)
        ratings = ratings.pivot(index="userId", columns="movieId", values="rating")
        
        
        self.update_user_rated_movies(ratings)
        
        ratings_np = ratings.to_numpy()
        
        mean_user_rating = ratings.mean(axis=1).to_numpy().reshape(-1, 1)
        
        ratings_diff = ratings_np - mean_user_rating
        ratings_diff[np.isnan(ratings_diff)] = 0

        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

        pred_df = DataFrame(pred, index=ratings.index, columns=ratings.columns)
        self.user_based_matrix = list(pred_df.to_dict("index").items())
        
    def create_item_based_matrix(self, data):
        if len(data) > 1: self.update_movies_id(data[1])
        ratings = data[0]
        ratings = DataFrame(ratings) # TODO remove
        
        # reorder ratings as matrix dataFrame (kind of)
        ratings = ratings.pivot(index="userId", columns="movieId", values="rating")
        self.update_user_rated_movies(ratings)

        ratings_np = ratings.to_numpy()
        
        mean_user_rating = ratings.mean(axis=1).to_numpy().reshape(-1, 1)
        
        ratings_diff = ratings_np - mean_user_rating
        ratings_diff[np.isnan(ratings_diff)] = 0
        raitingItem = ratings_diff
        
        item_similarity = 1-pairwise_distances(raitingItem.T, metric='cosine')
        pred = mean_user_rating + raitingItem.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])

        pred_df = DataFrame(pred, index=ratings.index, columns=ratings.columns)
        self.item_based_metrix = list(pred_df.to_dict("index").items())

    def predict_movies(self, user_id, k, is_user_based=True):
        recommended = []
        user_id = int(user_id)

        # user_dic1 will hold a map between movies and their rating by the user.
        # if rating is nan then user didnt see the movie. 
        # user_dic2 will hold the rating prediction of any movie (even if rated) 
        user_dic1 = self.user_rated_movies[user_id]
        user_dic2 = {}
        if is_user_based:
            for id, id_dic in self.user_based_matrix:
                if id == user_id:
                    user_dic2 = id_dic
                    break
            
        else:
            for id, id_dic in self.item_based_metrix:
                if id == user_id:
                    user_dic2 = id_dic
                    break
            
        user_dic2 = {k: v for k, v in sorted(user_dic2.items(), key=lambda item: item[1], reverse=True)}
        for movie_id, rate in user_dic2.items():
            if np.isnan(user_dic1[movie_id]):
                recommended.append(self.movies_id[movie_id])
                k -= 1
                if not k: break
        
        return recommended
