"""
Glycemia detection calculates the confusion matrix. The metric returns two lists, the first containing the percentages
in each region, while the second returns the number of samples.
"""
from .base_metric import BaseMetric
import numpy as np


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Glycemia Detection')
        self.hypo_threshold = 70
        self.hyper_threshold = 180

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # tuples = []
        percentages = []
        n_conditions = 3

        for i in range(n_conditions):
            for j in range(n_conditions):
                true_count = np.sum(self.condition(y_true, j))
                relevant_indices = np.where(self.condition(y_true, j))
                relevant_predictions = y_pred[relevant_indices]
                pred_count = np.sum(self.condition(relevant_predictions, i))

                percentage = pred_count / true_count
                percentages += [percentage]
                # samples = pred_count
                # tuples.append((percentage, samples))

        matrix = np.array(percentages).reshape(3, 3).tolist()
        return matrix

    def condition(self, x, condition):
        if condition == 0:
            return x < self.hypo_threshold
        elif condition == 1:
            return (x >= self.hypo_threshold) & (x <= self.hyper_threshold)
        else:
            return x > self.hyper_threshold
