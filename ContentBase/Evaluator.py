from utils.EvaluationData import EvaluationData
from utils.EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset)

        # Print results
        print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
        for (name, metrics) in results.items():
            print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        
    def GetRecomendationMovie(self, ml, userId, k=10):
        
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(userId)
            predictions = algo.GetAlgorithm().test(testSet)
            recommendations = []
            
            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:k]:
                print(ml.getMovieName(ratings[0]), ratings[1])
