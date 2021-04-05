from DataSource import DataSource
from ContentBase.utils.KNNAlgorithm import KNNAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

def LoadData():
    source = DataSource()
    print("Loading movie ratings...")
    data = source.loadMovieLensRating()
    print("Prepare movie information...")
    source.computeMovieInformation()
    print("Creating ranking for each movie ...")
    rankings = source.getPopularityRanksByRating()
    return (source, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(dataSource, data, rankings) = LoadData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(data, rankings)

contentKNN = KNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
# Random = NormalPredictor()
# evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate()

useTargetId = 85
totalMovieNeeded = 5
evaluator.GetRecomendationMovie(dataSource, useTargetId, totalMovieNeeded)


