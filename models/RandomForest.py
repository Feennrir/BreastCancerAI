from sklearn.ensemble import RandomForestClassifier


def create_model_random_forest():
    """
    Create and return a Random Forest classifier model.

    Returns:
        model (RandomForestClassifier): The Random Forest classifier model.
    """
    model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    return model