import Base
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender


class ItemKNNScoresHybridRecommender(BaseSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, *otherRecs):
        super(ItemKNNScoresHybridRecommender, self).__init__(URM_train)
        self.URM_train = Base.Recommender_utils.check_matrix(URM_train.copy(), 'csr')
        self.RecList = []
        self.RecList.append(Recommender_1)
        self.RecList.append(Recommender_2)
        for rec in otherRecs:
            self.RecList.append(rec)
        self.recLen = len(self.RecList)

    def fit(self, alpha=0.5, **otherParam):
        self.params = otherParam
        self.params['alpha'] = alpha

        if self.recLen is not 2 and self.recLen is not len(self.params):
            raise Exception('ERROR: parameters insufficient')
        if self.recLen is 2 and len(self.params) is 0:
            self.alphaOnly = True


    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights = []
        for rec in self.RecList:
            item_weights.append(rec._compute_item_score(user_id_array))
        if self.alphaOnly:
            item_weights = item_weights[0] * self.params['alpha'] + item_weights[0]* (1 - self.params['alpha'])
        else:
            pass

        return item_weights