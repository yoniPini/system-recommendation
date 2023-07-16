# yonatan pinchas 315538074
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())
        
        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    df_rating = DataFrame(data[0]) # TODO remove
    userId = df_rating["userId"]
    movieId = df_rating["movieId"]

    print("userId unique", userId.nunique()) # how many unique values
    print("movieId unique", movieId.nunique()) # how many unique movies
    print("totall",userId.size) # how many in totall
    
    df= df_rating.groupby("movieId")["movieId"].count().sort_values()
    df2= df_rating.groupby("userId")["userId"].count().sort_values()
    print("min rank movie",df.min())
    print("max rank movie",df.max())
    print("min rank user",df2.min())
    print("max rank user",df2.max())


def plot_data(data, plot = True):
    df_rating = DataFrame(data[0]) # TODO remove
    df_rating.hist("rating")
    if plot:
        plt.show()
