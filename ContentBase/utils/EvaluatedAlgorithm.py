from ContentBase.utils.MetricCalculator import MetricCalculator

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData):
        metrics = {}
        print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["MAE"] = MetricCalculator.MAE(predictions)
        metrics["RMSE"] = MetricCalculator.RMSE(predictions)
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
