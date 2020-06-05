import pandas as pd
import distance_metric_normalized as distance_metric
from sklearn.preprocessing import LabelEncoder# instantiate
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from LSH import LSH


def run_knn(args):
    ds = download_files(args.dataset)
    X,y  = preprocessing(args.qbits,args.dataset,ds)
    nearest_neighbor(args.d_metric,args.qbits,X,y)


def nearest_neighbor(d_metric,qbits,X,y):
    i = 1
    print("Splitting Dataset to 80/20")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2234)

    distance_metric.qbits=qbits

    n_neighbor = 1
    if d_metric == "dot":
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.dotproductdist)
    elif d_metric == "cosine":
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.cosinedist)
    elif d_metric == "euclidean":
        distance_metric.minkowski_p=2
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.minkowski)
    elif d_metric == "manhattan":
        distance_metric.minkowski_p=1
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.minkowski)
    elif d_metric == "chebyshev":
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.chebyshev)
    elif d_metric == "mcam":
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.mcamdist)
    elif d_metric == "mcam_ideal":
        classifier = KNeighborsClassifier(n_neighbors=i, metric=distance_metric.mcam_ideal)
    elif d_metric == "lsh":
        classifier = LSH(len(X_train[0]),qbits)
    else:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbor, metric=d_metric)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))


def preprocessing(qbits,dset,ds):
    print("Preprocessing", qbits)
    categorical_feature_mask = ds.dtypes==object
    tag = False
    for i in range(len(categorical_feature_mask)):
        if categorical_feature_mask[i] == True:
            tag = True
            break
    if tag == True:
        categorical_cols = ds.columns[categorical_feature_mask].tolist()
        le = LabelEncoder()
        ds[categorical_cols] = ds[categorical_cols].apply(lambda col: le.fit_transform(col))
        #print(ds.iloc[:,:].values[0])
        ohe = OneHotEncoder(handle_unknown='ignore')
        dataset = pd.DataFrame(ohe.fit_transform(ds[categorical_cols]).toarray())
        dataset = dataset.join(ds)
    else:
        dataset = ds

    nDataset = len(dataset)

    print("Getting features")
    if dset == 'wine':
        X = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values
    else:
        X = dataset.iloc[0:int(nDataset), :-1].values
        y = dataset.iloc[0:int(nDataset), -1].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y

def download_files(dataset):
    print("Downloading dataset")
    if dataset == "cancer":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data" #ok
    elif dataset == "dermatology":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data" # ok
    elif dataset == "glass":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data" # ok
    elif dataset == "ionosphere":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data" #ok
    elif dataset == "iris":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" # ok
    elif dataset == "liver":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
    elif dataset == "soybean":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data"
    elif dataset == "adult":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # not tested
    elif dataset == "adult":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # not tested
    elif dataset == "heart":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # not tested
    else:
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    ds = pd.read_csv(url)

    return ds
