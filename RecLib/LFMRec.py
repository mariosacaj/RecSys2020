from lightfm import LightFM
from Base.BaseRecommender import BaseRecommender
import numpy as np


class LFM(BaseRecommender):

    def __init__(self, URM_train, user_features=None, item_features=None, sample_weight=None):
        super(LFM, self).__init__(URM_train, False)
        self.user_features = user_features
        self.item_features = item_features
        self.sample_weight = sample_weight

    def fit(self, epochs=1, no_components=10,
            k=5,
            n=10,
            learning_schedule="adagrad",
            loss="logistic",
            learning_rate=0.05,
            rho=0.95,
            epsilon=1e-6,
            item_alpha=0.0,
            user_alpha=0.0,
            max_sampled=10,
            random_state=None, num_threads=1,
            verbose=False):

        self.model = LightFM(no_components,
                             k,
                             n,
                             learning_schedule,
                             loss,
                             learning_rate,
                             rho,
                             epsilon,
                             item_alpha,
                             user_alpha,
                             max_sampled,
                             random_state)

        self.model.fit(self.URM_train,
                       self.user_features,
                       self.item_features,
                       self.sample_weight,
                       epochs,
                       num_threads,
                       verbose)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scoresList = []
        if items_to_compute is None:
            for user_id in user_id_array:
                scores = self.model.predict(user_id, np.arange(self.n_items), item_features=None, user_features=None,
                                            num_threads=1)
                scoresList.append(scores)
        else:
            for user_id in user_id_array:
                scores = self.model.predict(user_id, items_to_compute, item_features=None, user_features=None,
                                            num_threads=1)
                scoresList.append(scores)
        return np.stack(scoresList, axis=0)
