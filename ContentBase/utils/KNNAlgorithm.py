from surprise import AlgoBase
from surprise import PredictionImpossible
from ContentBase.DataSource import DataSource
import math
import numpy as np
import heapq

class KNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        source = DataSource()
        genres = source.getGenres()
        years = source.getYears()

        print("Computing content-based similarity matrix...")
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRatingMovieId in range(self.trainset.n_items):
            if (thisRatingMovieId % 100 == 0):
                print(thisRatingMovieId, " of ", self.trainset.n_items)
            for otherRatingMovieId in range(thisRatingMovieId+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRatingMovieId))
                otherMovieID = int(self.trainset.to_raw_iid(otherRatingMovieId))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                self.similarities[thisRatingMovieId, otherRatingMovieId] = genreSimilarity * yearSimilarity
                self.similarities[otherRatingMovieId, thisRatingMovieId] = self.similarities[thisRatingMovieId, otherRatingMovieId]
                
        print("...done.")
                
        return self

    def estimate(self, u, i):
        print('estimating ...')
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('not found user and/or item to predict.')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal
        return predictedRating

    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim
