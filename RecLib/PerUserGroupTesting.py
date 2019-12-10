import scipy.sparse as sps
import numpy as np

from Base.Evaluation.Evaluator import EvaluatorHoldout


def prepare_user_base(URM_train, representativesPercentage = 0.05 ):
    URM_train = sps.csr_matrix(URM_train)
    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * representativesPercentage)
    sorted_users = np.argsort(profile_length)
    return profile_length, block_size, sorted_users

def _svd_rec(URM_train):
    from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    pureSVD = PureSVDRecommender(URM_train)
    pureSVD.fit()
    return pureSVD

def _topPop_rec(URM_train):
    from Base.NonPersonalizedRecommender import TopPop
    topPop = TopPop(URM_train)
    topPop.fit()
    return topPop

def compare(URM_train, URM_test, recommenderList, cutoff = 10, rangeLimit = 20, enableTop = True, enablePureSVD = True):
    profile_length, block_size, sorted_users = prepare_user_base(URM_train)

    MAPlist = [[]]

    if enablePureSVD:
        recommenderList.append(_svd_rec(URM_train))
    if enableTop:
        recommenderList.append(_topPop_rec(URM_train))

    MAP_itemKNN_per_group = []
    MAP_slim_per_group = []
    MAP_pureSVD_per_group = []
    MAP_topPop_per_group = []

    for group_id in range(0, rangeLimit):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        for idx, recommender in enumerate(recommenderList):
            results, _ = evaluator_test.evaluateRecommender(recommender)
            MAPlist[idx].append(results[cutoff]["MAP"])

    draw_charts(MAPlist, recommenderList)



def draw_charts(MAPlist, recommenderList):
    import matplotlib.pyplot as pyplot

    for idx, recommender in enumerate(recommenderList):
        pyplot.plot(MAPlist[idx], label=recommender.RECOMMENDER_NAME)
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()