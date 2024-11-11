from django import forms
from django.core.exceptions import ValidationError
MODEL_CHOICES = [
        ('tree-based', 'Tree-based (e.g., Random Forest, Decision Tree)'),
        ('other', 'Linear Models (e.g., Logistic Regression, SVM)'),
    ]

# Define available model choices
# MODEL_CHOICES = [
#     ('logistic_regression', 'Logistic Regression'),
#     ('decision_tree', 'Decision Tree'),
#     ('random_forest', 'Random Forest'),
#     ('svm', 'Support Vector Machine'),
#     ('knn', 'K-Nearest Neighbors'),
#     ('naive_bayes', 'Naive Bayes'),
#     ('gradient_boosting', 'Gradient Boosting'),
# ]

class UploadFileForm(forms.Form):
    file = forms.FileField(label="Select CSV File")   
    model_type = forms.ChoiceField(choices=MODEL_CHOICES, label='Model Type')


    def clean_file(self):
        uploaded_file = self.cleaned_data['file']
        if not uploaded_file.name.endswith('.csv'):
            raise ValidationError("Only CSV files are allowed.")
        return uploaded_file