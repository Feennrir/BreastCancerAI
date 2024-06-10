from sklearn.neighbors import KNeighborsClassifier


def create_model_knn():
    """
    Creates and returns a K-Nearest Neighbors model.

    Returns:
        model (KNeighborsClassifier): The K-Nearest Neighbors model.
    """
    model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    return model