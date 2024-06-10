from sklearn.tree import DecisionTreeClassifier


def create_model_decision_tree():
    """
    Creates and returns a DecisionTreeClassifier model.

    Returns:
        model (DecisionTreeClassifier): The created DecisionTreeClassifier model.
    """
    model = DecisionTreeClassifier(random_state=0, max_depth=20)
    return model