from models.DecisionTree import create_model_decision_tree
from models.KNN import create_model_knn
from models.RandomForest import create_model_random_forest
from models.SVM import create_model_svm


def display_model_options():
    """
    Displays the available model options and prompts the user to choose a model.

    Returns:
        int: The index of the chosen model.
    """
    while True:
        print("Choose the model: ")
        print("1. Decision Tree")
        print("2. KNN")
        print("3. SVM")
        print("4. Random Forest")
        modelIndex = int(input("Enter the model index: "))
        if modelIndex in range(1, 5):
            break
        else:
            print("Invalid model index. Please try again.")
    return modelIndex


def chooseModel():
    """
    Function to choose a machine learning model based on user input.

    Returns:
        The selected machine learning model or an error message if the index is invalid.
    """
    index = display_model_options()
    switcher = {
        1: create_model_decision_tree(),
        2: create_model_knn(),
        3: create_model_svm(),
        4: create_model_random_forest()
    }

    return switcher.get(index, "Invalid model index. Please try again.")

