# from django.shortcuts import render, HttpResponse
# from .models import TodoItem

# # Create your views here.
# def home(request):
#     return render(request, "home.html")

# def todos(request):
#     items = TodoItem.objects.all()
#     return render(request, "todos.html", {"todos": items })

from django.shortcuts import render
from .forms import UploadFileForm
from django.http import HttpResponse, HttpResponseBadRequest
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import normaltest
import numpy as np


def choose_scaler(df, threshold=1.5):
    """Automatically choose scaler based on data characteristics."""

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

    # Detect outliers based on the interquartile range (IQR)
    Q1 = df[numeric_features].quantile(0.25)
    Q3 = df[numeric_features].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_features] < (Q1 - threshold * IQR)) | (df[numeric_features] > (Q3 + threshold * IQR))).sum()

    # Check for normality (D'Agostino and Pearson's test)
    normal_p_values = df[numeric_features].apply(lambda x: normaltest(x).pvalue)
    normal_features = normal_p_values[normal_p_values > 0.05]  # Features likely normally distributed

    if len(normal_features) == len(numeric_features):
        # If all numeric features are normally distributed, use StandardScaler
        print("Data appears to be normally distributed, using StandardScaler.")
        return StandardScaler()
    elif any(outliers > 0):
        print("Outliers detected, using RobustScaler.")
        return RobustScaler()
    else:
        print("No significant outliers or normality, using MinMaxScaler.")
        return MinMaxScaler()

def remove_highly_correlated_features(df,numeric_features, correlation_threshold=0.9):
    """Remove features that have a high correlation with each other."""
    
    # Select only numeric features for correlation analysis
    #numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_features) == 0:
        print("No numeric features available for correlation analysis.")
        return df,numeric_features  # No changes to df since there are no numeric features

    corr_matrix = df[numeric_features].corr().abs()  # Get absolute value of the correlation matrix

    # Get the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
      # Drop safely
    # Drop columns from DataFrame
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # Update numeric_features list to exclude dropped columns
    updated_numeric_features = [col for col in numeric_features if col not in to_drop]

    return df, updated_numeric_features



def apply_pca(df, n_components=0.95):
    """Apply PCA to reduce dimensionality while retaining variance."""

    pca = PCA(n_components=n_components)

    # Transform data
    principal_components = pca.fit_transform(df)

    return pd.DataFrame(data=principal_components)

def encode_categorical(df, categorical_features, high_cardinality_threshold=10):
    """
    Automatically encode categorical features based on type and cardinality.

    :param df: Input DataFrame
    :param categorical_features: List of categorical columns
    :param model_type: Specify model type (e.g., 'tree-based' to avoid encoding for certain models)
    :param high_cardinality_threshold: Threshold to determine high cardinality for One-Hot Encoding
    :return: DataFrame with appropriate encoding applied
    """
    
    for column in categorical_features:
        unique_values = df[column].nunique()

        # Case 1: If the model is tree-based (like Random Forest, Decision Trees), skip encoding
        # if model_type == 'tree-based':
        #     print(f"Skipping encoding for {column} (tree-based model detected).")
        #     continue

        # Case 2: Binary features (2 unique values) can be label encoded or left as is
        if unique_values == 2:
            print(f"Binary feature detected: {column}, using Label Encoding.")
            df[column] = df[column].map({df[column].unique()[0]: 0, df[column].unique()[1]: 1})
            continue

        # Case 3: For low cardinality features, use One-Hot Encoding
        if unique_values <= high_cardinality_threshold:
            print(f"One-Hot Encoding for {column} (low cardinality).")
            df = pd.get_dummies(df, columns=[column], drop_first=True)
            continue

        # Case 4: For high cardinality features, use Target Encoding or Frequency Encoding
        if unique_values > high_cardinality_threshold:
            print(f"High cardinality feature detected: {column}, using Frequency Encoding.")
            # Frequency Encoding (replace categories by their frequency)
            freq_encoding = df[column].value_counts() / len(df)
            df[column] = df[column].map(freq_encoding)
    
    return df


def preprocess_data(df):
    """Automatically preprocess the dataset by choosing an appropriate scaler."""
    
    # Identify numeric and categorical features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Exclude 'ID' columns
    id_columns = [col for col in numeric_features if 'id' in col.lower()]
    numeric_features = numeric_features.difference(id_columns)
    
    # Fill missing values (impute)
    imputer = SimpleImputer(strategy='mean')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])

    for column in categorical_features:
        df[column] = df[column].fillna(df[column].mode()[0])
 
    # Encode categorical features
    #df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    df = encode_categorical(df, categorical_features)
    
    # Choose scaler automatically
    scaler = choose_scaler(df[numeric_features])

    # Scale numeric features
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Remove highly correlated features
    df, numeric_features = remove_highly_correlated_features(df,numeric_features)

    # Apply PCA for dimensionality reduction
    if len(numeric_features) > 12:  # Example condition to apply PCA
        df_numeric_pca = apply_pca(df[numeric_features])
        df_numeric_pca.columns = [f'PC{i + 1}' for i in range(df_numeric_pca.shape[1])]
        df = df.drop(columns=numeric_features).join(df_numeric_pca)


    # Drop ID columns
    df = df.drop(columns=id_columns)

    return df

def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
            data = pd.read_csv(uploaded_file)     

        # Preprocess the dataset
        preprocessed_data = preprocess_data(data)

        # Store preprocessed data in session as a CSV string
        request.session['preprocessed_csv_data'] = preprocessed_data.to_csv(index=False)

        # Display the first 10 rows of preprocessed data as HTML
        data_html = preprocessed_data.head(10).to_html(classes="table table-striped")

        return render(request, "myapp/display_data.html",{"data":data_html})
    else:
        form = UploadFileForm()

    return render(request, "myapp/upload.html", {"form": form})

def download_preprocessed_csv(request):
    # Retrieve the CSV data stored in the session
    csv_data = request.session.get('preprocessed_csv_data')  # Change to the correct key

    if csv_data:
        # Create an HTTP response with a CSV content type
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="preprocessed_data.csv"'

        # Write the CSV data to the response
        response.write(csv_data)
        return response

    return HttpResponse("No preprocessed data to download.", status=400)

from django.shortcuts import render, redirect
import os

# Define the form dynamically based on the earlier choice (tree-based or other)
def select_task_model(request):
    # model_type = request.session.get('model_type', None)  # Retrieve model_type from session
    # if not model_type:
    #     return redirect('upload_file')  # Redirect if model_type is not set

    # Define task and model options based on the model type
    #task_options = ['classification', 'regression'] if model_type == 'tree-based' else ['classification', 'regression', 'clustering']
    task_options = ['classification', 'regression', 'clustering']
    model_options = {
        # 'classification': ['decision_tree', 'random_forest', 'xgboost'] if model_type == 'tree-based' else ['decision_tree', 'random_forest', 'xgboost','logistic_regression', 'svm', 'knn', 'naive_bayes'],
        # 'regression': ['decision_tree', 'random_forest', 'xgboost'] if model_type == 'tree-based' else ['decision_tree', 'random_forest', 'xgboost','linear_regression', 'svm', 'knn'],
        # 'clustering': ['kmeans', 'dbscan']
        'classification':  ['decision_tree', 'random_forest', 'xgboost','logistic_regression', 'svm', 'knn', 'naive_bayes'],
        'regression': ['decision_tree', 'random_forest', 'xgboost','linear_regression', 'svm', 'knn'],
        'clustering': ['kmeans', 'dbscan'],
    }

    if request.method == "POST":
        # Get the selected task and model
        task_type = request.POST.get('task_type')
        model_type_selected = request.POST.get('model_type')

        # Get the preprocessed data
        csv_data = request.session.get('preprocessed_csv_data')
        if not csv_data:
            return redirect('upload_file')

        # Convert CSV string back to DataFrame
        import pandas as pd
        from io import StringIO
        preprocessed_data = pd.read_csv(StringIO(csv_data))

        # Use the root directory as the base
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


        notebooks_dir = os.path.join(BASE_DIR, 'Notebooks')
        os.makedirs(notebooks_dir, exist_ok=True)
        preprocessed_filepath = os.path.join(notebooks_dir, 'preprocessed_data.csv')
        preprocessed_data.to_csv(preprocessed_filepath, index=False)

        # Save the relative path for use in the Jupyter Notebook
        request.session['preprocessed_data_path'] = os.path.relpath(preprocessed_filepath, BASE_DIR)

        # Apply the ML model and generate insights
        model, insights,model_filepath = apply_ml_model(preprocessed_data, task_type, model_type_selected)
         # Save the model filename in the session for downloading
        request.session['model_filename'] = os.path.basename(model_filepath)



        # Store insights in the session for display on the results page
        request.session['results'] = {
            "task_type": task_type,
            "model_type_selected": model_type_selected,
            "insights": insights,
        }

        return redirect('results_page')
        

    # Render the page for selecting task and model
    return render(request, "myapp/select_task_model.html", {
        "task_options": task_options,
        "model_options": model_options,
        "results_displayed": False,
    })

def results_page(request):
    results = request.session.get('results', None)
    if not results:
        return redirect('select_task_model')  # Redirect if no results are available

    return render(request, "myapp/results.html", {
        "task_type": results["task_type"],
        "model_type_selected": results["model_type_selected"],
        "insights": results["insights"],
    })

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    silhouette_score,
)
import pandas as pd

import numpy as np

import joblib
import os

import papermill as pm
import os

def execute_notebook(algorithm):
    """Execute the corresponding Jupyter Notebook based on the algorithm."""
    # Get the root directory dynamically
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Correct paths for Notebooks and Saved Models directories
    notebooks_dir = os.path.join(BASE_DIR, 'Notebooks')
    saved_models_dir = os.path.join(BASE_DIR, 'savedModels')

    # Ensure the saved models directory exists
    os.makedirs(saved_models_dir, exist_ok=True)

    # Map algorithms to their respective notebooks
    notebook_map = {
        'linear_regression': 'linear_regression.ipynb',
        'kmeans': 'kmeans.ipynb',
        # Add more mappings as needed
    }

    # Determine paths for the input and output notebooks
    notebook_path = os.path.join(notebooks_dir, notebook_map[algorithm])
    output_notebook = os.path.join(notebooks_dir, f"output_{algorithm}.ipynb")

    # Path to the preprocessed dataset
    preprocessed_data_path = os.path.join(notebooks_dir, 'preprocessed_data.csv')

    # Execute the notebook
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        parameters={'data_path': preprocessed_data_path}
    )

    # Define the model filename for saving
    model_filename = f"{algorithm}_model.joblib"
    model_filepath = os.path.join(saved_models_dir, model_filename)

    return model_filepath

import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

# def apply_ml_model(df, task_type, model_type):
#     """Apply saved ML model based on the task type and model selected."""
#     insights = {}
#     model = None
#     model_filepath = None

#     if model_type in ['linear_regression', 'kmeans']:
#         BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
#         outputs_dir = os.path.join(BASE_DIR, 'Notebooks', 'outputs')
#         static_dir = os.path.join(BASE_DIR, 'myapp','static', 'myapp', 'outputs')
#         os.makedirs(static_dir, exist_ok=True)

#         # Ensure the outputs directory exists
#         os.makedirs(outputs_dir, exist_ok=True)

#         metrics_filepath = os.path.join(outputs_dir, f"{model_type}_metrics.json")
             
#         #model_filename = f"{model_type}_model.joblib"
#         notebooks_dir = os.path.join(BASE_DIR, 'Notebooks')

#         saved_models_dir = os.path.join(BASE_DIR, 'savedModels')
#         preprocessed_filepath = os.path.join(notebooks_dir, 'preprocessed_data.csv')
#         metrics_filepath = os.path.join(outputs_dir, "metrics.json")
#         graph_filepath = os.path.join(static_dir, "graph.png")
#         model_filepath = os.path.join(saved_models_dir, f"{model_type}_model.joblib")
#         df.to_csv(preprocessed_filepath, index=False)
#         #model_filepath = os.path.join(saved_models_dir, model_filename)
#         # Map task/model to corresponding notebook
#         notebook_map = {
#             'linear_regression': 'linear_regression.ipynb',
#             'kmeans': 'kmeans.ipynb',
#             # Add other notebooks here
#         }
#         # Ensure the notebook exists
#         if model_type not in notebook_map:
#             raise ValueError(f"No notebook available for model: {model_type}")
        
#         notebook_path = os.path.join(notebooks_dir, notebook_map[model_type])
#         output_notebook = os.path.join(notebooks_dir, f"output_{model_type}.ipynb")

#         # Path to save the trained model
#         model_filename = f"{model_type}_model.joblib"
#         model_filepath = os.path.join(saved_models_dir, model_filename)

#             # Execute the notebook with papermill
#         pm.execute_notebook(
#         notebook_path,
#         output_notebook,
#         parameters={
#             'data_path': preprocessed_filepath,
#             'model_save_path': model_filepath,
#             'metrics_save_path': metrics_filepath,   # Path to save the metrics JSON
#             'graph_save_path': graph_filepath,       # Path to save the graph
#         },
#         kernel_name='djangoenv'  # Specify the kernel name explicitly
#         )


#         # Read metrics from JSON
#         with open(metrics_filepath, 'r') as f:
#             metrics = json.load(f)

#         insights.update(metrics)
#          # Relative path to the graph
#         insights['graph_path'] = f"/static/myapp/outputs/{os.path.basename(graph_filepath)}"

#         return None, insights, model_filepath

#             # # Ensure the model file exists
#         # if not os.path.exists(model_filepath):
#         #     raise ValueError(f"Model {model_filename} not found!")
#         # # Load the model
#         # model = joblib.load(model_filepath)

#          # Assuming the task type is either classification, regression, or clustering
#         # X = df.iloc[:, :-1]  # Features (all columns except the last one)
#         # y = df.iloc[:, -1]   # Target (last column)

#         # if task_type == 'classification' or task_type == 'regression':
#         #     # For Linear Regression or other regression models
#         #     if model_type == 'linear_regression':
#         #         predictions = model.predict(X)
#         #         mse = mean_squared_error(y, predictions)
#         #         r2 = r2_score(y, predictions)
#         #         insights['mse'] = mse
#         #         insights['r2'] = r2
#         #         insights['predictions'] = predictions.tolist()
       
#         #          # Plot the actual vs predicted values for Linear Regression
#         #         plt.figure(figsize=(8, 6))
#         #         plt.scatter(y, predictions, color='blue', label='Actual vs Predicted')
#         #         plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
#         #         plt.title('Linear Regression: Actual vs Predicted')
#         #         plt.xlabel('Actual Values')
#         #         plt.ylabel('Predicted Values')
#         #         plt.legend()
#         #         plt.show()
#         #     else:
#         #         pass
#         # elif task_type == 'clustering':
#         #     # For KMeans or other clustering models
#         #     if model_type == 'kmeans':
#         #         predictions = model.predict(X)
#         #         silhouette_avg = silhouette_score(X, predictions)
#         #         insights['silhouette_score'] = silhouette_avg
#         #         insights['predictions'] = predictions.tolist()

#         #         # Plot the clusters and cluster centers for KMeans
#         #         pca = PCA(n_components=2)
#         #         X_reduced = pca.fit_transform(X)
#         #         plt.figure(figsize=(8, 6))
#         #         for cluster in range(model.n_clusters):
#         #             plt.scatter(X_reduced[predictions == cluster, 0], X_reduced[predictions == cluster, 1], label=f'Cluster {cluster}')
#         #         plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
#         #         plt.title('KMeans Clustering: 2D Visualization')
#         #         plt.xlabel('PCA Component 1')
#         #         plt.ylabel('PCA Component 2')
#         #         plt.legend()
#         #         plt.show()
#         #     else:
#         #         pass
#         # Return dummy insights for now (can be replaced with actual notebook outputs)
#         #insights = {"note": "Insights will be fetched from the executed notebook."}
#         #return None, insights, model_filepath
#     else:
#         """Apply selected ML model and return insights."""
#         X = df.iloc[:, :-1]  # Features (all columns except the last one)
#         y = df.iloc[:, -1]   # Target (last column)
#         insights = {}
#         model = None  # Initialize model to avoid UnboundLocalError

#         if task_type in ['classification', 'regression']:
#             # Split data into train and test sets
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Classification Models
#             if task_type == 'classification':
#                 if model_type == 'logistic_regression':
#                     model = LogisticRegression()
#                 elif model_type == 'decision_tree':
#                     model = DecisionTreeClassifier()
#                 elif model_type == 'random_forest':
#                     model = RandomForestClassifier()
#                 elif model_type == 'svm':
#                     model = SVC()
#                 elif model_type == 'knn':
#                     model = KNeighborsClassifier()
#                 elif model_type == 'naive_bayes':
#                     model = GaussianNB()
#                 else:
#                     raise ValueError(f"Unsupported model type for classification: {model_type}")

#                 # Train and predict
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_test)

#                 # Insights
#                 insights['accuracy'] = accuracy_score(y_test, predictions)
#                 insights['classification_report'] = classification_report(y_test, predictions, output_dict=True)
#                 insights['confusion_matrix'] = np.array(confusion_matrix(y_test, predictions)).tolist()

#             # Regression Models
#             elif task_type == 'regression':
#                 if model_type == 'linear_regression':
#                     model = LinearRegression()
#                 elif model_type == 'decision_tree':
#                     model = DecisionTreeRegressor()
#                 elif model_type == 'random_forest':
#                     model = RandomForestRegressor()
#                 elif model_type == 'svm':
#                     model = SVR()
#                 else:
#                     raise ValueError(f"Unsupported model type for regression: {model_type}")

#                 # Train and predict
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_test)

#                 # Insights
#                 insights['mean_squared_error'] = mean_squared_error(y_test, predictions)

#         elif task_type == 'clustering':
#             # Clustering Models
#             if model_type == 'kmeans':
#                 model = KMeans(n_clusters=3)  # Example: 3 clusters
#                 model.fit(X)
#                 predictions = model.labels_
#                 insights['silhouette_score'] = silhouette_score(X, predictions)
#                 insights['cluster_centers'] = model.cluster_centers_.tolist()  # Convert to list for serialization

#             elif model_type == 'dbscan':
#                 model = DBSCAN()
#                 model.fit(X)
#                 predictions = model.labels_
#                 insights['silhouette_score'] = silhouette_score(X, predictions)
#             else:
#                 raise ValueError(f"Unsupported model type for clustering: {model_type}")
        
#         else:
#             raise ValueError(f"Unsupported task type: {task_type}")

#     # Save the trained model
#     model_filename = f"{model_type}_model.joblib"
#     saved_models_dir = os.path.join('savedModels')
#     os.makedirs(saved_models_dir, exist_ok=True)
#     model_filepath = os.path.join(saved_models_dir, model_filename)
#     joblib.dump(model, model_filepath)

#     # Ensure all values in insights are JSON-serializable
#     insights['predictions'] = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

#     return model, insights, model_filepath

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, silhouette_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier

# Main function
# preprocess.py

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler

def preprocess_data_for_clustering(df):
    """
    Preprocess the data for clustering tasks:
    - Removes non-numeric columns
    - Scales the features for clustering
    """
    # Remove non-numeric columns
    numeric_data = df.select_dtypes(include=['number'])
    
    # Check for missing values
    if numeric_data.isnull().sum().sum() > 0:
        numeric_data = numeric_data.fillna(numeric_data.mean())  # Fill missing values with column mean

    # Scale the features
    scaler = StandardScaler()  # You can also use MinMaxScaler() if preferred
    scaled_data = scaler.fit_transform(numeric_data)

    return pd.DataFrame(scaled_data, columns=numeric_data.columns)

def select_features_and_target(data):
    """
    Automatically identifies features and the target column for ML tasks.
    Ensures the target column is categorical for classification tasks.
    
    Args:
        data (pd.DataFrame): Input dataframe with features and target variable.
        
    Returns:
        features (pd.DataFrame): Feature dataframe.
        target (pd.Series): Target variable (as a categorical dtype for classification).
    """
    # Identify potential target columns (non-numerical or categorical columns)
    potential_targets = data.select_dtypes(exclude=['number']).columns.tolist()

    if len(potential_targets) == 0:
        # If no non-numeric columns, assume the last column is the target
        target_col = data.columns[-1]
    else:
        # Use the first non-numeric column as the target
        target_col = potential_targets[0]

    # Convert target to categorical if it's not already
    target = data[target_col]
    if not pd.api.types.is_categorical_dtype(target):
        target = target.astype('category')

    # Drop the target column from the feature set
    features = data.drop(columns=[target_col])

    return features, target

def preprocess_continous_data(df, task_type):
    """
     Preprocess data for ML models. Dynamically selects target column and handles
    continuous target variables for classification tasks.
    
    Args:
        df (pd.DataFrame): Input dataframe with features and target variable.
        task_type (str): The type of ML task - 'classification', 'regression', or 'clustering'.
        
    Returns:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series or None): Target variable for classification/regression tasks; None for clustering.
    """
    
    # # Separate features (X) and target (y)
    # X = df.iloc[:, :-1]  # All columns except the last one as features
    # y = df.iloc[:, -1] if task_type in ['classification', 'regression'] else None
    X, y = select_features_and_target(df) 

    # Handle continuous target variables for classification
    if task_type == 'classification' and pd.api.types.is_numeric_dtype(y):
        # Use KBinsDiscretizer to convert continuous data into discrete bins
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  # 3 bins by default
        y = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
        y = pd.Series(y, name=df.columns[-1])  # Reassign as a pandas Series

    # Return preprocessed data
    return X, y

def apply_ml_model(df, task_type, model_type):
    """Apply selected ML model and return insights."""
    insights = {}
    model = None  # Initialize model to avoid UnboundLocalError

    # Split data into features (X) and target (y)
    if task_type=='clustering':
        X = df.iloc[:, :-1]  # Features (all columns except the last one)
        y = df.iloc[:, -1] if task_type in ['classification', 'regression'] else None
    else:
        X, y = preprocess_continous_data(df, task_type)

    if task_type in ['classification', 'regression']:
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Classification Task
        if task_type == 'classification':
            model, insights = train_classification_model(model_type, X_train, X_test, y_train, y_test)

        # Regression Task
        elif task_type == 'regression':
            model, insights = train_regression_model(model_type, X_train, X_test, y_train, y_test)

    elif task_type == 'clustering':
        # Clustering Task
        model, insights = train_clustering_model(model_type, X)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Save the trained model
    model_filepath = save_model(model, model_type)

    return model, insights, model_filepath

# Helper function: Train classification model
def train_classification_model(model_type, X_train, X_test, y_train, y_test):
    """Train a classification model and return insights."""
    model_mapping = {
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(),
        'knn': KNeighborsClassifier(),
        'naive_bayes': GaussianNB(),
        'xgboost': XGBClassifier()
    }
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type for classification: {model_type}")

    model = model_mapping[model_type]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Insights
    insights = {
        'accuracy': accuracy_score(y_test, predictions),
        'classification_report': classification_report(y_test, predictions, output_dict=True),
        'confusion_matrix': np.array(confusion_matrix(y_test, predictions)).tolist()
    }
    # Plot Confusion Matrix
    plot_confusion_matrix(confusion_matrix(y_test, predictions))
    #insights['confusion_matrix_graph'] = 'static/confusion_matrix.png'  # Save the graph path
    #insights['confusion_matrix_graph'] = plot_confusion_matrix(confusion_matrix(y_test, predictions))
    insights['confusion_matrix_graph'] = 'myapp/outputs/confusion_matrix.png'


    return model, insights

# Helper function: Train regression model
def train_regression_model(model_type, X_train, X_test, y_train, y_test):
    """Train a regression model and return insights."""
    model_mapping = {
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor(),
        'svm': SVR(),
        'xgboost': XGBClassifier(),
        'knn': KNeighborsClassifier()
    }
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type for regression: {model_type}")

    model = model_mapping[model_type]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error and R²
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # Insights
    insights = {
        'mse': mse,
        'r2': r2
    }
    # Plot Predicted vs Actual
    plot_regression_results(y_test, predictions)
    insights['regression_graph'] = 'myapp/outputs/regression_plot.png'  # Save the graph path


    return model, insights

# Helper function: Train clustering model
# def find_optimal_clusters(X):
#     """Use the elbow method to determine the optimal number of clusters."""
#     distortions = []
#     max_clusters = 10
#     for k in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(X)
#         distortions.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center

#     # Plot elbow curve
#     #elbow_graph_path = plot_elbow_curve(range(1, max_clusters + 1), distortions)

#     # Identify optimal clusters (this can be refined with a threshold-based method)
#     optimal_clusters = np.argmin(np.gradient(distortions)) + 1
#     return optimal_clusters

def train_clustering_model(model_type, X):
    """Train a clustering model and return insights."""
    
    model_mapping = {
        'kmeans': KMeans(n_clusters=3, random_state=42),  # Initial n_clusters = 3
        'dbscan': DBSCAN(eps=0.5, min_samples=5)  # Default parameters
    }
    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model type for clustering: {model_type}")

    model = model_mapping[model_type]
    model.fit(X)
    predictions = model.labels_
    # Count unique clusters
    unique_labels = np.unique(predictions)

    # Handle single-cluster case
    if len(unique_labels) <= 1:
        insights = {
            'error': "Clustering resulted in a single cluster. Silhouette Score cannot be computed."
        }
        return model, insights
    # Insights
    insights = {
        'silhouette_score': silhouette_score(X, predictions),
    }
    if model_type == 'kmeans':
        insights['cluster_centers'] = model.cluster_centers_.tolist()
      # Plot Clusters
    plot_clusters(X, predictions)
    insights['clustering_graph'] = 'myapp/outputs/clustering_plot.png'  # Save the graph path

    return model, insights

# Helper function: Save model
def save_model(model, model_type):
    """Save the model to disk and return the file path."""
    model_filename = f"{model_type}_model.joblib"
    saved_models_dir = os.path.join('savedModels')
    os.makedirs(saved_models_dir, exist_ok=True)
    model_filepath = os.path.join(saved_models_dir, model_filename)
    joblib.dump(model, model_filepath)
    return model_filepath

import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Define a base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to the directory of this script
STATIC_OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'myapp', 'outputs')

# Ensure the output directory exists
os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)

# Helper function to plot confusion matrix
def plot_confusion_matrix(cm):
    """Plot confusion matrix and save the figure."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    output_path = os.path.join(STATIC_OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    return output_path  # Return the path for insights


# Helper function to plot regression results
def plot_regression_results(y_test, predictions):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, predictions, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    output_path = os.path.join(STATIC_OUTPUT_DIR, 'regression_plot.png')
    plt.savefig(output_path)
    plt.close()
    return output_path  # Return the path for insights

# Helper function to plot clusters
def plot_clusters(X, predictions):
    """Plot clusters for KMeans or DBSCAN."""
    plt.figure(figsize=(4, 4))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=predictions, cmap='viridis')
    plt.title('Cluster Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    output_path = os.path.join(STATIC_OUTPUT_DIR, 'clustering_plot.png')
    plt.savefig(output_path)
    plt.close()
    return output_path  # Return the path for insights


from django.http import FileResponse, Http404
import os

def download_model(request):
    """Allow users to download the trained model."""
    model_filename = request.session.get('model_filename')
    if not model_filename:
        raise Http404("Model not found.")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_filepath  = os.path.join(BASE_DIR, 'savedModels',model_filename)

    #model_filepath = os.path.join('uploadcsv', 'savedModels', model_filename)
    try:
        return FileResponse(open(model_filepath, 'rb'), content_type='application/octet-stream')
    except FileNotFoundError:
        raise Http404("Model file not found.")
