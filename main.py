from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from userInput.chooseModel import chooseModel
from visualization.before_treatement import diagram_before_treatement
from workData.dataWork import separate_data, split_data, evaluate_model, load_model, model_file_exists, dump_model

model = chooseModel()

data = load_breast_cancer()
# Separate the data by features and target variable
X, Y = separate_data(data)

diagram_before_treatement(data, Y, X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.1)

# if the file modelTrained.joblib exists, load the model trained

knnFile = 'modelTrainedKNN.joblib'
svmFile = 'modelTrainedSVM.joblib'
decisionTreeFile = 'modelTrainedDecisionTree.joblib'
randomForestFile = 'modelTrainedRandomForest.joblib'

# If any of the model files exist, load the model from the file
if model_file_exists(knnFile) and model.__class__.__name__ == 'KNeighborsClassifier':
    print('KNN loaded from file')
    trainedModel = load_model('modelTrainedKNN.joblib')
elif model_file_exists(svmFile) and model.__class__.__name__ == 'SVC':
    print('SVM loaded from file')
    trainedModel = load_model('modelTrainedSVM.joblib')
elif model_file_exists(decisionTreeFile) and model.__class__.__name__ == 'DecisionTreeClassifier':
    print('Decision Tree loaded from file')
    trainedModel = load_model('modelTrainedDecisionTree.joblib')
elif model_file_exists(randomForestFile) and model.__class__.__name__ == 'RandomForestClassifier':
    print('Random Forest loaded from file')
    trainedModel = load_model('modelTrainedRandomForest.joblib')
# Else, dump the model to a file
else:
    trainedModel = dump_model(model, X_train, Y_train)

# Predict the target variable
Y_pred = trainedModel.predict(X_test)

# Evaluate the model
accuracy = evaluate_model(trainedModel, X_test, Y_test)
print("Accuracy:", str("{:.2f}".format(accuracy * 100) + "%"))

cm = confusion_matrix(Y_test, Y_pred)
# Calculate percentages for confusion matrix
cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Display confusion matrix with percentages
disp_percentages = ConfusionMatrixDisplay(confusion_matrix=cm_percentages, display_labels=data.target_names)
disp_percentages.plot(values_format='.2f', cmap='Blues')
plt.title('Confusion Matrix (Percentages)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# Input new values for prediction
new_values = [
    # 30 features in total
    [
        17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01315, 0.0198, 0.0023, 15.11, 19.26,
        99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ],
    [
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
        0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462,
        0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26,
        99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
    ],
    [
        13.08, 15.71, 85.63, 520, 0.1075, 0.127, 0.04568, 0.0311,
        0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097,
        0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49,
        96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183
    ]
]

# Convert the new values to a NumPy array
new_values = np.array(new_values)

# Make predictions on new values
new_predictions = trainedModel.predict(new_values)

# Get the indices of the "mean area" and "worst area" features
mean_area_index = data.feature_names.tolist().index("mean area")
worst_area_index = data.feature_names.tolist().index("worst area")

# Extract the values of "mean area" and "worst area" for each sample
mean_area_values = new_values[:, mean_area_index]
worst_area_values = new_values[:, worst_area_index]

# Plot the bar plot
width = 0.35
fig, ax = plt.subplots()
indices = np.arange(len(new_values))

# Plot "mean area" values
bar1 = ax.bar(indices - width/2, mean_area_values, width, label='Mean Area')

# Plot "worst area" values
bar2 = ax.bar(indices + width/2, worst_area_values, width, label='Worst Area')

# Add labels and title
ax.set_xlabel('Sample')
ax.set_ylabel('Area')
ax.set_title('Mean Area and Worst Area for New Values')
ax.set_xticks(indices)
ax.legend()

# Add text labels for prediction
for i, prediction in enumerate(new_predictions):
    ax.text(indices[i], max(mean_area_values[i], worst_area_values[i]),
            data.target_names[prediction], ha='center', va='bottom', color='black')

plt.show()