import Base
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
import numpy as np


class ScoresRecommender(BaseSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "ScoresRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, *otherRecs):
        super(ScoresRecommender, self).__init__(URM_train)
        self.URM_train = Base.Recommender_utils.check_matrix(URM_train.copy(), 'csr')
        self.RecList = []
        self.RecList.append(Recommender_1)
        self.RecList.append(Recommender_2)
        for rec in otherRecs:
            self.RecList.append(rec)
        self.recLen = len(self.RecList)

    def fit(self, *otherParam):
        self.alphaOnly = False
        self.params = []
        for param in otherParam:
            self.params.append(param)
        if self.recLen is not 2 and self.recLen is not len(self.params):
            print((self.recLen, len(self.params)))
            raise Exception('ERROR: parameters insufficient or wrong in number')
        if self.recLen is 2:
            if len(self.params) is 1:
                self.alphaOnly = True
                return
            elif len(self.params) is 2:
                pass
            else:
                raise Exception('ERROR: parameters insufficient or wrong in number')

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        partials = []

        for rec in self.RecList:
            partials.append(rec._compute_item_score(user_id_array, items_to_compute))

        item_weights = np.zeros_like(partials[0])

        if self.alphaOnly:
            item_weights = partials[0] * self.params[0] + partials[1] * (1 - self.params[0])
        else:
            for idx in range(self.recLen):
                item_weights = item_weights + partials[idx] * self.params[idx]

        return item_weights