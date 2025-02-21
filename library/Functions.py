# utils.py

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def categorize_icd9(code):
    """ Convert ICD-9 codes into broad disease categories """
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    
    if code.startswith('V'):
        return "V-Code (Health Services)"
    if code.startswith('E'):
        return "E-Code (External Injury)"

    try:
        code = int(float(code))
        if 1 <= code <= 139:
            return "Infectious Disease"
        elif 140 <= code <= 239:
            return "Neoplasms"
        elif 240 <= code <= 279:
            return "Endocrine (e.g., Diabetes)"
        elif 280 <= code <= 289:
            return "Blood Diseases"
        elif 290 <= code <= 319:
            return "Mental Disorders"
        elif 320 <= code <= 389:
            return "Nervous System"
        elif 390 <= code <= 459:
            return "Circulatory System"
        elif 460 <= code <= 519:
            return "Respiratory System"
        elif 520 <= code <= 579:
            return "Digestive System"
        elif 580 <= code <= 629:
            return "Genitourinary System"
        elif 630 <= code <= 679:
            return "Pregnancy-related"
        elif 680 <= code <= 709:
            return "Skin Diseases"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif 740 <= code <= 759:
            return "Congenital Disorders"
        elif 760 <= code <= 779:
            return "Perinatal Conditions"
        elif 780 <= code <= 799:
            return "Symptoms & Ill-defined"
        elif 800 <= code <= 999:
            return "Injury & Poisoning"
    except ValueError:
        return "Unknown"
    
    return "Other"


def split_by_patient(df, test_size=0.2, random_state=42):
    """
    Split the dataset ensuring all encounters for a patient stay together.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['patient_nbr']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Total encounters: {len(df)}")
    print(f"Unique patients: {df['patient_nbr'].nunique()}")
    print("\nTrain set:")
    print(f"Encounters: {len(train_df)}")
    print(f"Patients: {train_df['patient_nbr'].nunique()}")
    print("\nTest set:")
    print(f"Encounters: {len(test_df)}")
    print(f"Patients: {test_df['patient_nbr'].nunique()}")
    
    overlap = set(train_df['patient_nbr']).intersection(set(test_df['patient_nbr']))
    if overlap:
        print("\nWARNING: Patients in both sets!")
    else:
        print("\nVerified: No patient overlap")
        
    return train_df, test_df


def encode_categories(df, columns_to_encode, target_col='readmitted_binary', min_samples=10):
    """
    Encode categorical variables using simple target mean encoding.
    """
    df_encoded = df.copy()
    global_mean = df[target_col].mean()
    
    for col in columns_to_encode:
        category_stats = df.groupby(col)[target_col].agg(['mean', 'count'])
        weight = category_stats['count'] / (category_stats['count'] + min_samples)
        smoothed_means = weight * category_stats['mean'] + (1 - weight) * global_mean
        
        df_encoded[f"{col}_encoded"] = df_encoded[col].map(smoothed_means)
        df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded