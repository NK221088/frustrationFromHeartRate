from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score,
                           balanced_accuracy_score, cohen_kappa_score)
import numpy as np

from dataInspection import data

random_state = 29  # Set a random state for reproducibility

groups = list(data["Individual"])

frustrationLevels = {"low": [0,1,2,3], "medium": [4,5,6], "high": [7,8,9,10]}

y = data["Frustrated"].to_numpy()
yNew = []
for frustration in y:
    if frustration in frustrationLevels["low"]:
        yNew.append(0)
    elif frustration in frustrationLevels["medium"]:
        yNew.append(1)
    else:
        yNew.append(2)
y = np.array(yNew)
X = data.drop(columns=["Individual", "Phase", "Puzzler", "Frustrated", "Round", "Cohort"], axis=1).to_numpy()

# Initialize Leave-One-Group-Out cross-validator and models

logo = LeaveOneGroupOut()
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Store comprehensive metrics
results = {
    'logreg': {
        'accuracies': [], 'balanced_accuracies': [], 'kappa_scores': [],
        'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': [], 'classification_reports': [],
        'predictions': [], 'true_labels': []
    },
    'rf': {
        'accuracies': [], 'balanced_accuracies': [], 'kappa_scores': [],
        'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': [], 'classification_reports': [],
        'predictions': [], 'true_labels': []
    }
}

# Helper functions to train the models
def train_logreg(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model

# Helper function to evaluate models

def evaluate_model(model, X_test, y_test, modelName):
    """Comprehensive evaluation of a model"""
    
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = model.score(X_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, 
                             target_names=["Low", "Medium", "High"], 
                             labels=[0, 1, 2],  # Add this line
                             output_dict=True)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred.copy(),
        'true_labels': y_test.copy()
    }

# Create mapping between result keys and evaluate_model return keys
metric_mapping = {
    'accuracies': 'accuracy',
    'balanced_accuracies': 'balanced_accuracy', 
    'kappa_scores': 'kappa',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}
    
for train_idx, test_idx in logo.split(X, y, groups):
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    
    # Train model
    model1 = train_logreg(X_train, y_train)
    model2 = train_rf(X_train, y_train)
    
    # Evaluate model
    logreg_results = evaluate_model(model1, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(model2, X_test, y_test, "Random Forest")
    
    # Store results using the mapping
    for metric_key, result_key in metric_mapping.items():
        results['logreg'][metric_key].append(logreg_results[result_key])
        results['rf'][metric_key].append(rf_results[result_key])
    
    results['logreg']['confusion_matrices'].append(logreg_results['confusion_matrix'])
    results['rf']['confusion_matrices'].append(rf_results['confusion_matrix'])
    
    results['logreg']['classification_reports'].append(logreg_results['classification_report'])
    results['rf']['classification_reports'].append(rf_results['classification_report'])
    
    results['logreg']['predictions'].append(logreg_results['predictions'])
    results['rf']['predictions'].append(rf_results['predictions'])
    
    results['logreg']['true_labels'].append(logreg_results['true_labels'])
    results['rf']['true_labels'].append(rf_results['true_labels'])
    
# Calculate summary statistics
def calculate_summary_stats(metric_list):
    return {
        'mean': np.mean(metric_list),
        'std': np.std(metric_list),
        'min': np.min(metric_list),
        'max': np.max(metric_list),
        'median': np.median(metric_list)
    }

print("\n" + "="*60)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*60)

for model_name in ['logreg', 'rf']:
    print(f"\n{model_name.upper()} RESULTS:")
    print("-" * 30)
    
    for metric in ['accuracies', 'balanced_accuracies', 'kappa_scores', 'precision', 'recall', 'f1']:
        stats = calculate_summary_stats(results[model_name][metric])
        metric_display = metric.replace('_', ' ').title()
        print(f"{metric_display:20s}: {stats['mean']:.3f} ± {stats['std']:.3f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")

# Overall confusion matrix (aggregated across all folds)
print("\n" + "="*60)
print("AGGREGATED CONFUSION MATRICES")
print("="*60)

for model_name in ['logreg', 'rf']:
    # Initialize a 3x3 matrix for all classes
    overall_cm = np.zeros((3, 3), dtype=int)
    
    # Add each confusion matrix, handling different shapes
    for idx, cm in enumerate(results[model_name]['confusion_matrices']):
        # Get unique classes present in this fold
        present_classes = sorted(set(np.concatenate([
            results[model_name]['true_labels'][idx],
            results[model_name]['predictions'][idx]
        ])))
        
        # Map the confusion matrix values to the correct positions
        for i, true_class in enumerate(present_classes):
            for j, pred_class in enumerate(present_classes):
                overall_cm[true_class, pred_class] += cm[i, j]
    
    print(f"\n{model_name.upper()} - Overall Confusion Matrix:")
    print("    Predicted: Low  Med  High")
    for i, true_label in enumerate(['Low ', 'Med ', 'High']):
        print(f"True {true_label}: {overall_cm[i]}")

# Class Distribution Analysis
print("\n" + "="*60)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*60)

def analyze_class_distribution():
    """Analyze class distribution and its impact on performance"""
    all_true = np.concatenate(results['logreg']['true_labels'])
    unique, counts = np.unique(all_true, return_counts=True)
    
    # Map class indices to names
    class_names = ['Low', 'Medium', 'High']
    class_dist = {class_names[i]: counts[list(unique).index(i)] if i in unique else 0 
                  for i in range(3)}
    
    print("Overall Class Distribution:")
    total_samples = sum(class_dist.values())
    for class_name, count in class_dist.items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name:6s}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Check class distribution per fold
    print("\nPer-fold Class Distribution:")
    for fold_idx, true_labels in enumerate(results['logreg']['true_labels']):
        unique_fold, counts_fold = np.unique(true_labels, return_counts=True)
        fold_dist = {class_names[i]: counts_fold[list(unique_fold).index(i)] if i in unique_fold else 0 
                     for i in range(3)}
        print(f"  Fold {fold_idx+1:2d}: {fold_dist}")
    
    return class_dist

class_distribution = analyze_class_distribution()

# Class-wise performance summary
print("\n" + "="*60)
print("CLASS-WISE PERFORMANCE SUMMARY")
print("="*60)

for model_name in ['logreg', 'rf']:
    print(f"\n{model_name.upper()}:")
    
    # Aggregate classification reports
    all_reports = results[model_name]['classification_reports']
    
    for class_name in ['Low', 'Medium', 'High']:
        precisions = [report[class_name]['precision'] for report in all_reports if class_name in report]
        recalls = [report[class_name]['recall'] for report in all_reports if class_name in report]
        f1s = [report[class_name]['f1-score'] for report in all_reports if class_name in report]
        
        if precisions:  # Only print if we have data for this class
            print(f"  {class_name:6s} - Precision: {np.mean(precisions):.3f}±{np.std(precisions):.3f}, "
                  f"Recall: {np.mean(recalls):.3f}±{np.std(recalls):.3f}, "
                  f"F1: {np.mean(f1s):.3f}±{np.std(f1s):.3f} "
                  f"(appears in {len(precisions)}/{len(all_reports)} folds)")
        else:
            print(f"  {class_name:6s} - No predictions for this class across any fold")

# Statistical significance test between models
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import matthews_corrcoef

print("\n" + "="*60)
print("STATISTICAL COMPARISON")
print("="*60)

# McNemar's Test - Most appropriate for paired predictions
def compare_models_mcnemar(pred1, pred2, y_true):
    """McNemar's test for comparing paired predictions"""
    all_pred1 = np.concatenate(pred1)
    all_pred2 = np.concatenate(pred2) 
    all_true = np.concatenate(y_true)
    
    correct1 = (all_pred1 == all_true)
    correct2 = (all_pred2 == all_true)
    
    # Create 2x2 contingency table
    both_correct = np.sum(correct1 & correct2)
    logreg_only = np.sum(correct1 & ~correct2)
    rf_only = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    table = np.array([[both_correct, logreg_only],
                      [rf_only, both_wrong]])
    
    try:
        result = mcnemar(table, exact=True)
        return result, table
    except:
        return None, table

mcnemar_result, mcnemar_table = compare_models_mcnemar(
    results['logreg']['predictions'], 
    results['rf']['predictions'], 
    results['logreg']['true_labels']
)

print("McNemar's Test (LogReg vs RandomForest):")
print("Contingency Table:")
print("                 RF Correct  RF Wrong")
print(f"LogReg Correct:     {mcnemar_table[0,0]:3d}       {mcnemar_table[0,1]:3d}")
print(f"LogReg Wrong:       {mcnemar_table[1,0]:3d}       {mcnemar_table[1,1]:3d}")

if mcnemar_result is not None:
    print(f"McNemar's statistic: {mcnemar_result.statistic:.3f}")
    print(f"p-value: {mcnemar_result.pvalue:.3f}")
    if mcnemar_result.pvalue < 0.05:
        print("Significant difference between models (p < 0.05)")
    else:
        print("No significant difference between models (p >= 0.05)")
else:
    print("McNemar's test could not be computed (insufficient data)")

print("\nPaired t-tests comparing LogReg vs RandomForest:")
for metric in ['accuracies', 'balanced_accuracies', 'f1']:
    logreg_scores = results['logreg'][metric]
    rf_scores = results['rf'][metric]
    
    t_stat, p_value = stats.ttest_rel(logreg_scores, rf_scores)
    metric_display = metric.replace('_', ' ').title()
    print(f"{metric_display:20s}: t={t_stat:.3f}, p={p_value:.3f}")

# Effect sizes (Cohen's d)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

print("\nEffect sizes (Cohen's d) - LogReg vs RandomForest:")
for metric in ['accuracies', 'balanced_accuracies', 'f1']:
    d = cohens_d(results['logreg'][metric], results['rf'][metric])
    metric_display = metric.replace('_', ' ').title()
    print(f"{metric_display:20s}: d={d:.3f}")

# Matthews Correlation Coefficient - Better for imbalanced classes
print("\n" + "="*60)
print("MATTHEWS CORRELATION COEFFICIENT")
print("="*60)

mcc_scores = {'logreg': [], 'rf': []}
for model_name in ['logreg', 'rf']:
    for pred, true in zip(results[model_name]['predictions'], results[model_name]['true_labels']):
        mcc = matthews_corrcoef(true, pred)
        mcc_scores[model_name].append(mcc)

for model_name in ['logreg', 'rf']:
    mcc_stats = calculate_summary_stats(mcc_scores[model_name])
    print(f"{model_name.upper()} MCC: {mcc_stats['mean']:.3f} ± {mcc_stats['std']:.3f} "
          f"(min: {mcc_stats['min']:.3f}, max: {mcc_stats['max']:.3f})")

# MCC comparison
mcc_t_stat, mcc_p_value = stats.ttest_rel(mcc_scores['logreg'], mcc_scores['rf'])
mcc_d = cohens_d(mcc_scores['logreg'], mcc_scores['rf'])
print(f"\nMCC Comparison (LogReg vs RF): t={mcc_t_stat:.3f}, p={mcc_p_value:.3f}, d={mcc_d:.3f}")

# Model Robustness Analysis
print("\n" + "="*60)
print("MODEL ROBUSTNESS ANALYSIS")
print("="*60)

def analyze_robustness():
    """Analyze performance variability across individuals (folds)"""
    fold_accuracies = {'logreg': [], 'rf': []}
    fold_f1_scores = {'logreg': [], 'rf': []}
    
    for model_name in ['logreg', 'rf']:
        for pred, true in zip(results[model_name]['predictions'], results[model_name]['true_labels']):
            fold_acc = np.mean(pred == true)
            fold_f1 = calculate_summary_stats([results[model_name]['f1'][
                results[model_name]['predictions'].index(pred)]])['mean']
            fold_accuracies[model_name].append(fold_acc)
            fold_f1_scores[model_name].append(fold_f1)
    
    return fold_accuracies, fold_f1_scores

fold_accuracies, fold_f1_scores = analyze_robustness()

print("Cross-Validation Stability (lower std = more robust):")
print(f"LogReg Accuracy CV std: {np.std(fold_accuracies['logreg']):.3f}")
print(f"RF Accuracy CV std:     {np.std(fold_accuracies['rf']):.3f}")
print(f"LogReg F1 CV std:       {np.std(results['logreg']['f1']):.3f}")
print(f"RF F1 CV std:           {np.std(results['rf']['f1']):.3f}")

# Determine which model is more robust
if np.std(fold_accuracies['logreg']) < np.std(fold_accuracies['rf']):
    print("\nLogistic Regression shows more consistent performance across individuals")
else:
    print("\nRandom Forest shows more consistent performance across individuals")

print("\n" + "="*60)
print("CROSS-VALIDATION STRATEGY JUSTIFICATION")
print("="*60)

print("Leave-One-Group-Out Cross-Validation Analysis:")
print(f"- Total number of individuals (groups): {len(set(groups))}")
print(f"- Total number of samples: {len(y)}")
print(f"- Samples per individual: {len(y) / len(set(groups)):.1f} (average)")

print("\nWhy LOGO is appropriate for this dataset:")
print("1. Tests generalization to completely NEW INDIVIDUALS")
print("2. Prevents data leakage from repeated measures (rounds/phases)")
print("3. Simulates real-world deployment where model sees new patients")
print("4. Accounts for individual differences in physiological responses")

print(f"\nData structure justifying LOGO:")
print("- Multiple rounds (1-4) per individual")
print("- Multiple phases (1-3) per round") 
print("- This creates strong dependencies within individuals")
print("- Standard k-fold would leak information between train/test splits")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)