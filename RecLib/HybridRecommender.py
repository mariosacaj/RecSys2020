
from Base.BaseRecommender import BaseRecommender
from Base.NonPersonalizedRecommender import TopPop
from .ScoresRecommender import ScoresRecommender
import numpy as np



class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, *otherRecs):
        super(HybridRecommender, self).__init__(URM_train)
        self.topPop = TopPop(URM_train)
        self.ItemHybrid = ScoresRecommender(URM_train, Recommender_1, Recommender_2, otherRecs)

    def fit(self):
        self.topPop.fit()
        self.ItemHybrid.fit(shrink=50, topK=5)

    """

    :param user_id_array:       array containing the user indices whose recommendations need to be computed
    :param items_to_compute:    array containing the items whose scores are to be computed.
                                    If None, all items are computed, otherwise discarded items will have as score -np.inf
    :return:                    array (len(user_id_array), n_items) with the score.
    """
    # def _compute_item_score(self, user_id_array, items_to_compute=None):
    #     x = np.arange(0, self.n_users)
    #     cold_users_train = x[self._cold_user_mask]
    #
    #     # Cold & Warm Users ID
    #     cold_user_id_array = np.intersect1d(cold_users_train, user_id_array)
    #     warm_user_id_array = np.setdiff1d(user_id_array, cold_user_id_array)
    #
    #
    #     item_weights_1 = self.topPop._compute_item_score(cold_user_id_array, items_to_compute)
    #     item_weights_2 = self.CF._compute_item_score(warm_user_id_array, items_to_compute)
    #
    #
    #     if items_to_compute is None:
    #         item_weights = np.empty((len(user_id_array), self.n_items))
    #     else:
    #         item_weights = np.empty((len(user_id_array), len(items_to_compute)))
    #
    #     coldCounter: int = 0
    #     warmCounter: int = 0
    #     for idx, user_id in enumerate(user_id_array):
    #         if user_id in cold_user_id_array:
    #             item_weights[idx] = item_weights_1[coldCounter]
    #             coldCounter = coldCounter + 1
    #         elif user_id in warm_user_id_array:
    #             item_weights[idx] = item_weights_2[warmCounter]
    #             warmCounter = warmCounter + 1
    #         else:
    #             raise Exception("ERROR IN ITEM SCORE COMPUTATION")
    #     return item_weights
    
    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        x = np.arange(0, self.n_users)
        cold_users = x[self._cold_user_mask]
        rankList = []
        scoresList = []

        for user in user_id_array:
            if user in cold_users:
                recommended_items, scores = self.topPop.recommend(user, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag, remove_custom_items_flag, return_scores=True)
            else:
                recommended_items, scores = self.ItemHybrid.recommend(user, cutoff, remove_seen_flag, items_to_compute, remove_top_pop_flag, remove_custom_items_flag, return_scores=True)
            rankList.append(recommended_items)
            scoresList.append(scores)

        # Return single list for one user, instead of list of lists
        if single_user:
            rankList = rankList[0]

        if return_scores:
            return rankList, np.array(scoresList).reshape((len(user_id_array), -1))
        else:
            return rankList
