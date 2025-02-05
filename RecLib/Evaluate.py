import scipy.sparse as sps
import numpy as np
from collections import defaultdict

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

    MAPlist = defaultdict(list)

    if enablePureSVD:
        recommenderList.append(_svd_rec(URM_train))
    if enableTop:
        recommenderList.append(_topPop_rec(URM_train))


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
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as pyplot

    for idx, recommender in enumerate(recommenderList):
        pyplot.plot(MAPlist[idx], label=recommender.RECOMMENDER_NAME)
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

def evaluateAgainstUsers(rec_to_eval, itrMax, URM_train, URM_test, cutoff=10):
    for itr in range(itrMax):
        to_compute_mask = np.ediff1d(URM_train.tocsr().indptr) == itr
        to_ignore_mask = np.invert(to_compute_mask)
        to_ignore = np.arange(URM_train.shape[0])[to_ignore_mask]
        if len(to_ignore) != URM_train.shape[0]:
            evalTest = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=to_ignore)
            results, _ = evalTest.evaluateRecommender(rec_to_eval)
            print('MAP at ' + str(itr) + ' interactions: ' + str(results[cutoff]["MAP"]))

def sortMap(MAP_LIST):
    from operator import itemgetter
    MAP_LIST = sorted(MAP_LIST, key=itemgetter(1))
    for row in MAP_LIST:
        print(row)
