from metrics.base_metric import BaseMetric
import numpy as np

class RMSE(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.sqrt(np.mean(np.square(y_true - y_pred)))
