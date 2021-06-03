import numpy as np


class EarlyStopping:
    """
    Returns False if the score doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=5, score_name="elbo", verbose=False, delta=0,
                 stash_model=False):
        """
        :param patience: How long to wait after last time score improved.
        :param verbose: Whether to print message informing of early stopping.
        :param delta: Minimum change to qualify as an improvement.
        :param stash_model: Whether to update the best model with the score.
        """
        self.patience = patience
        self.score_name = score_name
        self.verbose = verbose
        self.delta = delta
        self.stash_model = stash_model
        self.best_model = None
        self.best_score = None

    def __call__(self, scores=None, model=None):
        """
        :param scores: A dict of scores.
        :param model: Current model producing latest score.
        :return: Whether to stop early.
        """
        if scores is None:
            self.best_score = None
            if self.stash_model:
                self.best_model = model

            return

        else:
            vals = scores[self.score_name]

            # Check whether best score has been beaten.
            new_val = vals[-1]
            if self.best_score is None or new_val > self.best_score:
                self.best_score = new_val
                if self.stash_model and model is not None:
                    self.best_model = model

            # Check whether to stop.
            if len(vals) > self.patience:
                prev_vals = np.array(vals[-self.patience:])
                ref_val = np.array(vals[-self.patience - 1]) + self.delta

                if np.all(prev_vals < ref_val):
                    return True
                else:
                    return False
            else:
                return False
