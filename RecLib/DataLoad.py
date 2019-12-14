
# Visualize rows/columns stats
import csv
import numpy as np
from scipy.sparse import coo_matrix


def list_ID_stats(ID_list, label):
    ID_list = list(map(int, ID_list))
    list_length = len(ID_list)
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_val = len(set(ID_list))
    repetitions = list_length - unique_val
    delta = max_val - min_val
    missing_val = 0.
    if delta is not 0:
        missing_val = 1 - min(unique_val, delta)/delta

    print("{} data, ID: min {}, max {}, length {}, unique {}, repetitions {}, missig {:.2f} %".format(label, min_val, max_val, list_length, unique_val, repetitions, missing_val*100))


# This function loads CSV files to COOrdinate formatted sparse matrixes
def toCoo(filepath, rowsDesc, columnsDesc):
    rows = []
    columns = []
    data = []
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                rows.append(row[0])
                columns.append(row[1])
                data.append(row[2])
                line_count += 1
    print(filepath)
    list_ID_stats(rows, rowsDesc)
    list_ID_stats(columns, columnsDesc)
    print(
    )
    data = np.array(data).astype(np.float)
    rows = np.array(rows).astype(np.int)
    columns = np.array(columns).astype(np.int)
    return coo_matrix((data, (rows, columns)))

# This function loads CSV files to NParrays
def toNPArray(filepath):
    users = []
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                users.append(row[0])
                line_count += 1
    users = np.array(users).astype(np.int)
    return users

# Building the ICM
def ICMbuilder(URM, SubclassesMatrix, PriceMatrix, AssetMatrix):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    le.fit(AssetMatrix.data)
    assetList_icm = le.transform(AssetMatrix.data)


    le.fit(PriceMatrix.data)
    priceList_icm = le.transform(PriceMatrix.data)
    print(priceList_icm)
    print(str(assetList_icm.min()) + ' ' + str(assetList_icm.max()) + ' ' + str(len(set(assetList_icm))) + ' ' + str(
        len(assetList_icm)))
    print(str(priceList_icm.min()) + ' ' + str(priceList_icm.max()) + ' ' + str(len(set(priceList_icm))) + ' ' + str(
        len(priceList_icm)))

    n_items = URM.shape[1]
    n_assets = len(set(assetList_icm))
    ICMassets_shape = (n_items, n_assets)

    ones = np.ones(len(assetList_icm))
    ICM_assets = coo_matrix((ones, (AssetMatrix.row, assetList_icm)), shape=ICMassets_shape)
    ICM_assets = ICM_assets.tocsr()

    n_items = URM.shape[1]
    n_prices = len(set(priceList_icm))
    ICMprices_shape = (n_items, n_prices)

    ones = np.ones(len(priceList_icm))
    ICM_prices = coo_matrix((ones, (PriceMatrix.row, priceList_icm)), shape=ICMprices_shape)
    ICM_prices = ICM_prices.tocsr()

    # ICM_all=np.hstack((ICM_assets,ICM_prices))
    # ICM_all=np.hstack((ICM_all,ICM_subclass))
    ICM_all = np.concatenate((ICM_assets.todense(), ICM_prices.todense(), SubclassesMatrix.todense()), axis=1)

    return ICM_all


# Building the UCM
def UCMbuilder(URM, AgeMatrix, RegionMatrix):

    n_users = URM.shape[0]
    UCM_age = AgeMatrix
    UCM_region = RegionMatrix
    # zeros = np.zeros(UCM_age.shape[1])
    UCM_age.resize((n_users, UCM_age.shape[1]))
    UCM_region.resize((n_users, UCM_region.shape[1]))
    # np.append(UCM_age, zeros, axis=0)
    # print(UCM_age.shape + ' ' + zeros.shape())
    UCM = np.concatenate((UCM_age.todense(), UCM_region.todense()), axis=1)

    return UCM



