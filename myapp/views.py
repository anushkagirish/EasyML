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

def encode_categorical(df, categorical_features, model_type=None, high_cardinality_threshold=10):
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
        if model_type == 'tree-based':
            print(f"Skipping encoding for {column} (tree-based model detected).")
            continue

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


def preprocess_data(df,model_type):
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
    df = encode_categorical(df, categorical_features, model_type)
    

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
            model_type = form.cleaned_data['model_type']
            data = pd.read_csv(uploaded_file)     

        # Preprocess the dataset
        preprocessed_data = preprocess_data(data, model_type=model_type)

        # Store preprocessed data in session as a CSV string
        request.session['preprocessed_csv_data'] = preprocessed_data.to_csv(index=False)

        # Display the first 10 rows of preprocessed data as HTML
        data_html = preprocessed_data.head(10).to_html(classes="table table-striped")

        return render(request, "myapp/display_data.html",{"data":data_html,"model": model_type})
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

