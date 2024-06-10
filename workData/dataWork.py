from sklearn.model_selection import train_test_split
from joblib import dump, load


def separate_data(data):
    """
    Separates the data into input features (X) and target variable (Y).

    Parameters:
    data (object): The data object containing both input features and target variable.

    Returns:
    tuple: A tuple containing the input features (X) and target variable (Y).
    """
    X = data.data
    Y = data.target
    return X, Y


def split_data(X, Y, test_size):
    """
    Split the data into training and testing sets.

    Parameters:
    X (array-like): The input features.
    Y (array-like): The target variable.
    test_size (float): The proportion of the data to include in the test set.

    Returns:
    X_train (array-like): The training set input features.
    X_test (array-like): The test set input features.
    Y_train (array-like): The training set target variable.
    Y_test (array-like): The test set target variable.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the performance of a machine learning model.

    Parameters:
    model (object): The trained machine learning model.
    X_test (array-like): The input features for testing.
    Y_test (array-like): The target values for testing.

    Returns:
    float: The score of the model's performance.
    """
    score = model.score(X_test, Y_test)
    return score


def model_file_exists(filename):
    """
    Check if a file exists.

    Parameters:
    filename (str): The name of the file to check.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    try:
        with open(filename, 'r') as file:
            return True
    except FileNotFoundError:
        return False


def load_model(filename):
    """
    Load a trained model from a file.

    Parameters:
    filename (str): The name of the file containing the trained model.

    Returns:
    object: The trained machine learning model.
    """
    model = load(filename)
    return model


def dump_model(model, X_train, Y_train):
    """
    Train a machine learning model and save it to a file.

    Parameters:
    model (object): The machine learning model to train.
    X_train (array-like): The input features for training.
    Y_train (array-like): The target values for training.

    Returns:
    None

    """
    filename = ''

    # If model is DecisionTreeClassifier
    if model.__class__.__name__ == 'DecisionTreeClassifier':
        print("Training Decision Tree model...")
        filename = 'modelTrainedDecisionTree.joblib'
    # If model is KNeighborsClassifier
    elif model.__class__.__name__ == 'KNeighborsClassifier':
        print("Training KNN model...")
        filename = 'modelTrainedKNN.joblib'
    # If model is SVC
    elif model.__class__.__name__ == 'SVC':
        print("Training SVM model...")
        filename = 'modelTrainedSVM.joblib'
    # If model is RandomForestClassifier
    elif model.__class__.__name__ == 'RandomForestClassifier':
        print("Training Random Forest model...")
        filename = 'modelTrainedRandomForest.joblib'
    else:
        print("Model not found")
        return

    trainedModel = model.fit(X_train, Y_train)
    dump(trainedModel, filename)
    return trainedModel
