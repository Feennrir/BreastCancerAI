import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def diagram_before_treatement(data, Y, X):
    '''
    This function creates a box plot to visualize the distribution of feature values before any treatment.
    Args:
        data: The dataset.
        Y: The target variable.
        X: The input features.

    Returns: None
    '''
    # Create a DataFrame combining X and Y
    df = pd.DataFrame(data=X, columns=data.feature_names)
    df['diagnosis'] = data.target_names[Y]

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Melt the DataFrame to have a single column for feature values
    melted_df = pd.melt(df, id_vars=['diagnosis'], value_vars=data.feature_names, var_name='Feature',
                        value_name='Feature Value')


    sns.barplot(x='Feature', y='Feature Value', hue='diagnosis', data=melted_df, errorbar=None)

    # Set labels and title
    plt.xlabel('Feature')
    plt.ylabel('Feature Value')
    plt.title('Feature Values Before Treatment')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    # Display legend
    plt.legend(title='Diagnosis')

    # Show plot
    plt.tight_layout()
    plt.show()