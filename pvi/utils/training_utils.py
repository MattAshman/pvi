import numpy as np


class EarlyStopping:
    """
    Returns False if the score doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        :param patience: How long to wait after last time score improved.
        :param verbose: Whether to print message informing of early stopping.
        :param delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

    def __call__(self, scores):
        """
        :param scores: A list of scores.
        :return: Whether to stop early.
        """
        if len(scores) > self.patience:
            prev_scores = np.array(scores[-self.patience:])
            ref_score = np.array(scores[-self.patience - 1]) + self.delta

            if np.all(prev_scores < ref_score):
                return True
            else:
                return False
        else:
            return False
