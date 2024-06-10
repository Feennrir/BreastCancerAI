from sklearn.svm import SVC


def create_model_svm():
    """
    Creates a Support Vector Machine (SVM) model with a linear kernel.

    Returns:
        model (SVC): The SVM model.
    """
    model = SVC(kernel='linear', random_state=0)
    return model
