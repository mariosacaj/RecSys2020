
from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np


class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train):
        super(HybridRecommender, self).__init__(URM_train)
        self.topPop = TopPop(URM_train)
        self.CF = ItemKNNCFRecommender(URM_train)

    def fit(self):
        self.topPop.fit()
        self.CF.fit(shrink=50, topK=5)

    """

    :param user_id_array:       array containing the user indices whose recommendations need to be computed
    :param items_to_compute:    array containing the items whose scores are to be computed.
                                    If None, all items are computed, otherwise discarded items will have as score -np.inf
    :return:                    array (len(user_id_array), n_items) with the score.
    """
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        x = np.arange(0, self.n_users)
        cold_users_train = x[self._cold_user_mask]

        # Cold & Warm Users ID
        cold_user_id_array = np.intersect1d(cold_users_train, user_id_array)
        warm_user_id_array = np.setdiff1d(user_id_array, cold_user_id_array)

        item_weights_1 = self.topPop._compute_item_score(cold_user_id_array, items_to_compute)
        item_weights_2 = self.CF._compute_item_score(warm_user_id_array, items_to_compute)

        if items_to_compute is None:
            item_weights = np.empty((len(user_id_array), self.n_items))
        else:
            item_weights = np.empty((len(user_id_array), len(items_to_compute)))

        coldCounter: int = 0
        warmCounter: int = 0
        for idx, user_id in enumerate(user_id_array):
            if user_id in cold_user_id_array:
                item_weights[idx] = item_weights_1[coldCounter]
                coldCounter = coldCounter + 1
            else:
                item_weights[idx] = item_weights_2[warmCounter]
                warmCounter = warmCounter + 1
        return item_weights
