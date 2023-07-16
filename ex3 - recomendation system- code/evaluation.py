# yonatan pinchas 315538074
# Import Pandas
from collaborative_filtering import collaborative_filtering
from pandas.core.frame import DataFrame

def into_dic(test_set: DataFrame):
    """
    return the test set as a dictionary between each user and each movie.
    the dictionary is orderd by values from max to min.
    """
    test_set = test_set.pivot(index="userId", columns="movieId", values="rating")
    test_set = test_set.fillna(-1)
    test_set_dic = test_set.to_dict("index")
    
    dic = {}
    for user_id, user_dic in test_set_dic.items():
        dic[user_id] = {k: v for k, v in sorted(user_dic.items(), key=lambda item: item[1], reverse=True)}

    return dic

def precision_10(test_set: DataFrame, cf: collaborative_filtering, is_user_based = True):
    test_set_dic = into_dic(test_set)
    recommended_users = {}

    for user_id, _ in test_set_dic.items():
        recommended_users[user_id] = cf.predict_movies(user_id, 10, is_user_based)

    pre_10 = 0.0
    count_users = 0
    for user_id, user_dic in test_set_dic.items():
        count_users += 1
        count_hit = 0
        for title in recommended_users[user_id]:
            movie_id = cf.movies_name[title]
            if movie_id in user_dic.keys() and user_dic[movie_id]  >= 4.0:
                count_hit += 1
        pre_10 += count_hit / 10

    print("Precision_k: " + str(pre_10 / count_users))

def ARHA(test_set,  cf: collaborative_filtering, is_user_based = True):
    test_set_dic = into_dic(test_set)
    recommended_users = {}

    for user_id, _ in test_set_dic.items():
        recommended_users[user_id] = cf.predict_movies(user_id, 10, is_user_based)
    
    count_users = 0
    sum_pos = 0.0
    for user_id, user_dic in test_set_dic.items():
        count_users += 1
        rec_list_user = recommended_users[user_id]
        for title in rec_list_user:
            movie_id = cf.movies_name[title]
            if movie_id in user_dic.keys() and user_dic[movie_id]  >= 4.0:
                sum_pos += 1 / (rec_list_user.index(title) + 1)

    print("ARHR: " + str(sum_pos / count_users))

def get_user_dic(user_id, cf: collaborative_filtering, is_user_based = True):
    if is_user_based:
        for id, dic in cf.user_based_matrix:
            if id == user_id: return dic
    for id, dic in cf.item_based_metrix:
        if id == user_id: return dic

def RSME(test_set, cf: collaborative_filtering, is_user_based = True):
    test_set_dic = into_dic(test_set)

    count_ratings = 0
    sum_squre_dist = 0.0
    for user_id, user_dic in test_set_dic.items():
        user_pred_dic = get_user_dic(user_id, cf, is_user_based)
        for movie_id, rate in user_dic.items():
            if rate >= 0.0:
                count_ratings += 1
                sum_squre_dist += (rate - user_pred_dic[movie_id]) ** 2
    
    root = (sum_squre_dist / count_ratings) ** 0.5
    print("RMSE: " + str(root))


