from torch import Tensor, zeros, bincount
from torch.nn import Module 

from typing import Optional





class Accuracy(Module): 


    def __init__(self): 
        super(Accuracy, self).__init__()
        return 
    

    def forward(
            self, 
            y_pred: Tensor, 
            y_true: Tensor,
            weights: Optional[Tensor] = None
    ): 
        N = y_pred.size(0)
        n = y_pred.argmax(1).eq(y_true).sum()
        accuracy = n / N
        return accuracy





class WeightedAccuracy(Module): 


    def __init__(self): 
        super(WeightedAccuracy, self).__init__()
        return 
    

    def forward(
            self, 
            y_pred: Tensor, 
            y_true: Tensor,
            weights: Optional[Tensor] = None
    ):  
        num_labels = weights.size(0)

        y_pred = y_pred.argmax(1)
        y_pred = y_pred[y_pred.eq(y_true)]

        y_pred_counts = bincount(y_pred, minlength = num_labels)
        y_true_counts = bincount(y_true, minlength = num_labels)
        is_not_null = y_true_counts > 0 

        accuracy = zeros(num_labels)
        accuracy[is_not_null] = y_pred_counts[is_not_null] / y_true_counts[is_not_null]
        accuracy = accuracy.dot(weights) / weights.sum()
        accuracy = accuracy

        return accuracy
