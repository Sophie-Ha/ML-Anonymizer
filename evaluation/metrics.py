import numpy as np


class MetricHelper:

    def __init__(self, predicted, target):
        """
        :param [[int]] predicted:
        :param [[int]] target:
        """
        self.predicted = predicted
        self.target = target
        self._conf_mat = None

    @property
    def conf_mat(self):
        if self._conf_mat is None:
            self._get_conf_mat()
        return self._conf_mat

    def _get_conf_mat(self):
        self._conf_mat = np.zeros((2, 2))  # dim: (true, pred)
        for p_list, t_list in zip(self.predicted, self.target):
            for p, t in zip(p_list, t_list):
                self.conf_mat[t, p] += 1

    def recall(self):
        s = self.conf_mat[1, 1] + self.conf_mat[1, 0]
        if s > 0:
            return self.conf_mat[1, 1]/s
        else:
            return 0

    def precision(self):
        s = self.conf_mat[1, 1] + self.conf_mat[0, 1]
        if s > 0:
            return self.conf_mat[1, 1] / s
        else:
            return 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        if p + r > 0:
            return (2 * p * r)/(p + r)
        else:
            return 0
