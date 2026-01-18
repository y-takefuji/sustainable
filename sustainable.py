import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import shap
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('Data_for_UCI_named.csv')
print(data.shape)
print(data['stabf'].value_counts())
# Encode 'stabf' to binary (0=unstable, 1=stable)
le = LabelEncoder()
data['stabf'] = le.fit_transform(data['stabf'])

# Define X and y
X = data.drop('stabf', axis=1)
y = data['stabf']

# Analyze target distribution
print("\n=== Target Distribution Analysis ===")
target_counts = y.value_counts()
print(f"Target value counts:\n{target_counts}")
print(f"Target distribution percentage:\n{target_counts / len(y) * 100}")

# Define a function for feature selection using Random Forest
def select_features_rf(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance.head(min(n_features, len(feature_importance)))['feature'].tolist()

# Define a function for feature selection using XGBoost
def select_features_xgb(X, y, n_features=5):
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance.head(min(n_features, len(feature_importance)))['feature'].tolist()

# Define a function for feature selection using Feature Agglomeration
def select_features_fa(X, y=None, n_features=5):  # Added y parameter with default None
    # Check if there are enough features to create clusters
    if X.shape[1] <= 1:
        return X.columns.tolist()[:min(n_features, X.shape[1])]  # Return all remaining features
        
    # Group features into clusters
    n_clusters = min(X.shape[1] - 1, 2)  # Ensure at least 2 clusters but not more than features-1
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    
    # Calculate importance of each feature (variance)
    variances = np.var(X, axis=0)
    
    # Map each feature to its cluster
    cluster_map = pd.DataFrame({
        'feature': X.columns,
        'cluster': agglo.labels_,
        'variance': variances
    })
    
    # Sort features by variance (importance)
    cluster_map = cluster_map.sort_values('variance', ascending=False)
    
    # Select top n_features across all clusters
    return cluster_map.head(min(n_features, len(cluster_map)))['feature'].tolist()

# Define a function for feature selection using HVGS
def select_features_hvgs(X, y=None, n_features=5):  # Added y parameter with default None
    variances = np.var(X, axis=0)
    feature_variance = pd.DataFrame({'feature': X.columns, 'variance': variances})
    feature_variance = feature_variance.sort_values('variance', ascending=False)
    return feature_variance.head(min(n_features, len(feature_variance)))['feature'].tolist()

# Define a function for feature selection using Spearman's correlation
def select_features_spearman(X, y, n_features=5):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append((col, abs(corr)))  # Use absolute value for ranking
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in correlations[:min(n_features, len(correlations))]]

# Define a function for feature selection using RF-SHAP
def select_features_rf_shap(X, y, n_features=5):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Create explainer and get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Debug info
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values list length: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            print(f"SHAP values[{i}] shape: {sv.shape}")
    else:
        print(f"SHAP values shape: {shap_values.shape}")
    
    # Handle the specific (n_samples, n_features, n_classes) shape
    if not isinstance(shap_values, list) and len(shap_values.shape) == 3 and shap_values.shape[0] == len(X) and shap_values.shape[1] == len(X.columns):
        print("Handling (n_samples, n_features, n_classes) shape")
        # Take the mean absolute SHAP values across all samples and classes
        shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    elif isinstance(shap_values, list):
        if len(shap_values) >= 2:  # Binary classification typically gives 2 arrays
            # Use the positive class (usually index 1)
            shap_importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            # If only one array, use it
            shap_importance = np.abs(shap_values[0]).mean(axis=0)
    else:
        # If shap_values is not a list, check its shape
        if len(shap_values.shape) == 3:  # Other 3D shapes
            shap_importance = np.abs(shap_values).mean(axis=(0, 1))
        elif len(shap_values.shape) == 2:  # (n_samples, n_features)
            shap_importance = np.abs(shap_values).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    
    print(f"Resulting shap_importance shape: {shap_importance.shape}, length: {len(shap_importance)}")
    print(f"Number of features: {len(X.columns)}")
    
    # Make sure we have the right number of features
    if len(shap_importance) != len(X.columns):
        raise ValueError(f"SHAP importance length ({len(shap_importance)}) doesn't match number of features ({len(X.columns)})")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns, 
        'importance': shap_importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance.head(min(n_features, len(feature_importance)))['feature'].tolist()

# Define a function for feature selection using XGB-SHAP
def select_features_xgb_shap(X, y, n_features=5):
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    
    # Create explainer and get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Debug info
    print(f"XGB-SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"XGB-SHAP values list length: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            print(f"XGB-SHAP values[{i}] shape: {sv.shape}")
    else:
        print(f"XGB-SHAP values shape: {shap_values.shape}")
    
    # Handle the specific (n_samples, n_features, n_classes) shape
    if not isinstance(shap_values, list) and len(shap_values.shape) == 3 and shap_values.shape[0] == len(X) and shap_values.shape[1] == len(X.columns):
        print("Handling (n_samples, n_features, n_classes) shape")
        # Take the mean absolute SHAP values across all samples and classes
        shap_importance = np.abs(shap_values).mean(axis=(0, 2))
    elif isinstance(shap_values, list):
        if len(shap_values) >= 2:  # Binary classification typically gives 2 arrays
            # Use the positive class (usually index 1)
            shap_importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            # If only one array, use it
            shap_importance = np.abs(shap_values[0]).mean(axis=0)
    else:
        # If shap_values is not a list, check its shape
        if len(shap_values.shape) == 3:  # Other 3D shapes
            shap_importance = np.abs(shap_values).mean(axis=(0, 1))
        elif len(shap_values.shape) == 2:  # (n_samples, n_features)
            shap_importance = np.abs(shap_values).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    
    print(f"Resulting shap_importance shape: {shap_importance.shape}, length: {len(shap_importance)}")
    print(f"Number of features: {len(X.columns)}")
    
    # Make sure we have the right number of features
    if len(shap_importance) != len(X.columns):
        raise ValueError(f"SHAP importance length ({len(shap_importance)}) doesn't match number of features ({len(X.columns)})")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns, 
        'importance': shap_importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    return feature_importance.head(min(n_features, len(feature_importance)))['feature'].tolist()

# Define a function for cross-validation
def perform_cv(X, y, features, model_type='rf', cv=5):
    if not features:  # Check if features list is empty
        return 0.0
        
    X_subset = X[features]
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(random_state=42)
    
    cv_scores = cross_val_score(model, X_subset, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), scoring='accuracy')
    return np.mean(cv_scores)

# Initialize results dictionary
results = {
    'method': [],
    'CV5': [],
    'CV4': [],
    'top5_features': [],
    'top4_features': []
}

# Process each algorithm
for method, func, model_type in [
    ('Random Forest', select_features_rf, 'rf'),
    ('XGBoost', select_features_xgb, 'xgb'),
    ('Feature Agglomeration', select_features_fa, 'rf'),
    ('HVGS', select_features_hvgs, 'rf'),
    ('Spearman', select_features_spearman, 'rf'),
    ('RF-SHAP', select_features_rf_shap, 'rf'),
    ('XGB-SHAP', select_features_xgb_shap, 'xgb')
]:
    try:
        print(f"\n=== Processing method: {method} ===")
        # All methods use the same function signature now
        top5_features = func(X, y, n_features=5)
        print(f"Top 5 features: {top5_features}")
        
        # Calculate CV score with top 5 features
        cv5_score = perform_cv(X, y, top5_features, model_type=model_type)
        print(f"CV5 score: {cv5_score:.4f}")
        
        # Check if we have at least one feature to remove
        if top5_features:
            # Create reduced dataset by removing the highest ranked feature
            X_reduced = X.drop(top5_features[0], axis=1)
            
            # Select top 4 features from reduced dataset
            top4_features = func(X_reduced, y, n_features=4)
            print(f"Top 4 features: {top4_features}")
        else:
            top4_features = []
        
        # Calculate CV score with top 4 features
        cv4_score = perform_cv(X, y, top4_features, model_type=model_type)
        print(f"CV4 score: {cv4_score:.4f}")
        
        # Store results
        results['method'].append(method)
        results['CV5'].append(cv5_score)
        results['CV4'].append(cv4_score)
        results['top5_features'].append(', '.join(top5_features))
        results['top4_features'].append(', '.join(top4_features))
        print(f"Successfully processed method: {method}")
    except Exception as e:
        print(f"Error processing {method}: {e}")
        # Add error entries to results
        results['method'].append(method)
        results['CV5'].append(np.nan)
        results['CV4'].append(np.nan)
        results['top5_features'].append(f"ERROR: {str(e)}")
        results['top4_features'].append(f"ERROR: {str(e)}")

# Create and save results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv('result.csv', index=False)
print("\n=== Final Results ===")
print(results_df)
