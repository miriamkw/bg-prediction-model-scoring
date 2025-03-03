from .base_metric import BaseMetric
from error_grids import zone_accuracy


class Metric(BaseMetric):
    def __init__(self):
        super().__init__('Parkes Error Grid')

    def _calculate_metric(self, y_true, y_pred, *args, **kwargs):
        accuracy_values = zone_accuracy(y_true, y_pred, 'parkes')
        formatted_values = ["{:.1f}%".format(value * 100) for value in accuracy_values]

        return formatted_values
