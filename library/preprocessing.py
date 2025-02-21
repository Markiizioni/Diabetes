import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import mannwhitneyu

def encode_categories(df, columns_to_encode, target_col='readmitted_binary', min_samples=10):
    """
    Encode all categorical variables using simple target mean encoding.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing columns to encode
    columns_to_encode : list
        List of all columns to encode
    target_col : str
        Name of the binary target column (default: 'readmitted_binary')
    min_samples : int
        Minimum samples for smoothing (default: 10)
        
    Returns:
    --------
    df_encoded : pandas DataFrame
        DataFrame with encoded columns
    """
    # Copy input data
    df_encoded = df.copy()
    
    # Input validation
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Calculate global mean for smoothing
    global_mean = df[target_col].mean()
    
    # Process each column
    for col in columns_to_encode:
        # Calculate category statistics
        category_stats = df.groupby(col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing based on sample size
        weight = category_stats['count'] / (category_stats['count'] + min_samples)
        smoothed_means = weight * category_stats['mean'] + (1 - weight) * global_mean
        
        # Map encoded values
        df_encoded[f"{col}_encoded"] = df_encoded[col].map(smoothed_means)
        
        # Remove original column
        df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded


def categorize_icd9(code):
    """Convert ICD-9 codes into broad disease categories"""
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    
    if code.startswith('V'): return "V-Code (Health Services)"
    if code.startswith('E'): return "E-Code (External Injury)"

    try:
        code = int(float(code))
        if 1 <= code <= 139: return "Infectious Disease"
        elif 140 <= code <= 239: return "Neoplasms"
        elif 240 <= code <= 279: return "Endocrine (e.g., Diabetes)"
        elif 280 <= code <= 289: return "Blood Diseases"
        elif 290 <= code <= 319: return "Mental Disorders"
        elif 320 <= code <= 389: return "Nervous System"
        elif 390 <= code <= 459: return "Circulatory System"
        elif 460 <= code <= 519: return "Respiratory System"
        elif 520 <= code <= 579: return "Digestive System"
        elif 580 <= code <= 629: return "Genitourinary System"
        elif 630 <= code <= 679: return "Pregnancy-related"
        elif 680 <= code <= 709: return "Skin Diseases"
        elif 710 <= code <= 739: return "Musculoskeletal"
        elif 740 <= code <= 759: return "Congenital Disorders"
        elif 760 <= code <= 779: return "Perinatal Conditions"
        elif 780 <= code <= 799: return "Symptoms & Ill-defined"
        elif 800 <= code <= 999: return "Injury & Poisoning"
    except ValueError:
        return "Unknown"
    return "Other"

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['weight', 'payer_code', 'medical_specialty']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # Replace missing values with NaN
        X = X.replace('?', np.nan)
        
        # Remove rows where gender is 'Unknown/Invalid'
        X = X[X['gender'] != 'Unknown/Invalid']
        
        # Drop specified columns
        X = X.drop(columns=self.columns_to_drop)
        
        # Drop rows with missing values in specific columns
        X = X.dropna(subset=['diag_1', 'diag_2', 'diag_3', 'race'])
        
        # Fill NaN values for specific columns
        X['max_glu_serum'] = X['max_glu_serum'].fillna(0)
        X['A1Cresult'] = X['A1Cresult'].fillna(0)
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drug_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        self.POLYPHARMACY_THRESHOLD = 10
        self.interaction_terms = [
            ('num_medications', 'time_in_hospital'),
            ('num_medications', 'num_procedures'),
            ('time_in_hospital', 'num_lab_procedures'),
            ('num_medications', 'num_lab_procedures'),
            ('num_medications', 'number_diagnoses'),
            ('number_diagnoses', 'time_in_hospital'),
            ('num_medications', 'num_visits')
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # Create readmitted_binary and drop readmitted
        if 'readmitted_binary' not in X.columns and 'readmitted' in X.columns:
            X['readmitted_binary'] = X['readmitted'].map({'NO': 0, '>30': 1, '<30': 1})
            X = X.drop(columns=['readmitted'])
        
        # Process diagnoses
        X['diag_1_category'] = X['diag_1'].apply(categorize_icd9)
        X['diag_2_category'] = X['diag_2'].apply(categorize_icd9)
        X['diag_3_category'] = X['diag_3'].apply(categorize_icd9)
        X = X.drop(columns=['diag_1', 'diag_2', 'diag_3'])
        
        # Basic features
        X['total_medication_changes'] = X[self.drug_cols].apply(
            lambda row: sum(row.isin(['Up', 'Down'])), axis=1
        )
        X['polypharmacy_indicator'] = (X['num_medications'] > self.POLYPHARMACY_THRESHOLD).astype(int)
        
        # Insulin features
        X['insulin_used'] = X['insulin'].apply(lambda x: 0 if x == 'No' else 1)
        X['insulin_dependence_ratio'] = X['insulin_used'] / (X['num_medications'] + 1)
        
        # Stability index
        X['stability_index'] = (X['num_medications'] + X['num_lab_procedures']) / (X['time_in_hospital'] + 1)
        
        # Number of visits
        X['num_visits'] = X['number_outpatient'] + X['number_emergency'] + X['number_inpatient']
        
        # Convert relevant columns to numeric
        numeric_cols = ['num_medications', 'time_in_hospital', 'num_procedures', 
                       'num_lab_procedures', 'number_diagnoses', 'num_visits']
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Generate interaction features
        for col1, col2 in self.interaction_terms:
            if col1 in X.columns and col2 in X.columns:
                name = f"{col1}|{col2}"
                X[f"{name}_mult"] = X[col1] * X[col2]
                X[f"{name}_sum"] = X[col1] + X[col2]
                X[f"{name}_ratio"] = X[col1] / (X[col2] + 1e-5)
        
        # Polynomial features
        X['num_medications_squared'] = X['num_medications'] ** 2
        X['time_in_hospital_squared'] = X['time_in_hospital'] ** 2
        X['number_diagnoses_squared'] = X['number_diagnoses'] ** 2
        
        # Healthcare utilization ratios
        X['num_medications_time_in_hospital_ratio'] = X['num_medications'] / (X['time_in_hospital'] + 1e-5)
        X['num_medications_num_lab_procedures_ratio'] = X['num_medications'] / (X['num_lab_procedures'] + 1e-5)
        X['number_diagnoses_time_in_hospital_ratio'] = X['number_diagnoses'] / (X['time_in_hospital'] + 1e-5)
        
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformations = {}
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include='number').columns
        
        for col in numeric_cols:
            skew = X[col].skew()
            kurt = X[col].kurtosis()
            
            if abs(skew) > 2 and abs(kurt) > 2:
                zeros_ratio = (X[col] == 0).sum() / len(X)
                self.transformations[col] = 'log1p' if zeros_ratio > 0.02 else 'log'
                
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col, transform_type in self.transformations.items():
            if transform_type == 'log':
                X[col] = np.log(X[col].replace(0, np.nan))
            else:  # log1p
                X[col] = np.log1p(X[col])
                
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        return encode_categories(X, categorical_columns)

class DataCollapser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # Sort by patient ID and encounter
        X_sorted = X.sort_values(by=['patient_nbr', 'encounter_id'])
        
        # Keep first encounter for each patient
        collapsed_data = X_sorted.drop_duplicates(subset=['patient_nbr'], keep='first')
        
        # Calculate total encounters
        encounter_counts = X['patient_nbr'].value_counts().reset_index()
        encounter_counts.columns = ['patient_nbr', 'total_encounters']
        
        # Merge back and calculate average length of stay
        collapsed_data = collapsed_data.merge(encounter_counts, on='patient_nbr', how='left')
        collapsed_data['avg_los_per_visit'] = collapsed_data['time_in_hospital'] / (collapsed_data['total_encounters'] + 1)
        
        return collapsed_data

def get_preprocessing_pipeline():
    """Returns the preprocessing pipeline with all transformers."""
    return Pipeline([
        ('cleaner', DataCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('categorical_encoder', CategoricalEncoder()),
        ('log_transformer', LogTransformer()),
        ('data_collapser', DataCollapser())
    ])