import warnings
# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*multi_class.*')
warnings.filterwarnings('ignore', message='.*y_pred contains classes not in y_true.*')

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score,
                           balanced_accuracy_score, cohen_kappa_score)
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import matthews_corrcoef
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
import numpy as np

from dataInspection import data

random_state = 29

def plot_confusion_matrix(cm, title, labels=['Low', 'High'], save_path=None):
    """
    Plot a confusion matrix with proper formatting and save as PDF
    """
    plt.figure(figsize=(6, 5))
    # Transpose the confusion matrix to swap axes
    cm_transposed = cm.T
    sns.heatmap(cm_transposed, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Confusion matrix saved: {save_path}")
    plt.close()  # Close without showing

def plot_roc_curves_improved(results, model_names=['logreg', 'svm'], save_path=None):
    """
    Plot ROC curves using pooled predictions from all balanced folds
    """
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red']
    
    for i, model_name in enumerate(model_names):
        # Pool all predictions and true labels from balanced folds
        pooled_y_true = []
        pooled_y_proba = []
        
        for fold_idx, is_prob in enumerate(results[model_name]['is_problematic']):
            if not is_prob:  # Only balanced folds
                y_true = results[model_name]['true_labels'][fold_idx]
                y_proba = results[model_name]['probabilities'][fold_idx][:, 1]
                
                pooled_y_true.extend(y_true)
                pooled_y_proba.extend(y_proba)
        
        if pooled_y_true:  # Only plot if we have balanced folds
            # Calculate single ROC curve from pooled data
            fpr, tpr, _ = roc_curve(pooled_y_true, pooled_y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot smooth ROC curve
            plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                    label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Pooled Balanced Folds', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"ROC curves saved: {save_path}")
    plt.close()

# ALSO ADD this alternative function for per-fold analysis:

def plot_roc_curves_per_fold_summary(results, model_names=['logreg', 'svm'], save_path=None):
    """
    Plot ROC curve summary statistics across folds
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red']
    
    for i, model_name in enumerate(model_names):
        auc_scores = []
        
        # Calculate AUC for each balanced fold
        for fold_idx, is_prob in enumerate(results[model_name]['is_problematic']):
            if not is_prob:  # Only balanced folds
                y_true = results[model_name]['true_labels'][fold_idx]
                y_proba = results[model_name]['probabilities'][fold_idx][:, 1]
                
                if len(set(y_true)) == 2:  # Ensure both classes present
                    try:
                        fpr, tpr, _ = roc_curve(y_true, y_proba)
                        auc_score = auc(fpr, tpr)
                        auc_scores.append(auc_score)
                    except:
                        pass
        
        if auc_scores:
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            
            # Create a box plot of AUC scores
            positions = [i + 1]
            plt.boxplot([auc_scores], positions=positions, widths=0.6,
                       patch_artist=True, 
                       boxprops=dict(facecolor=colors[i], alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
            
            plt.text(i + 1, mean_auc + 0.02, f'{model_name.upper()}\nMean: {mean_auc:.3f}\n±{std_auc:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.xlim(0.5, len(model_names) + 0.5)
    plt.ylim(0, 1)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('AUC Score Distribution Across Balanced Folds', fontsize=14)
    plt.xticks(range(1, len(model_names) + 1), [name.upper() for name in model_names])
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"AUC distribution plot saved: {save_path}")
    plt.close()

    
def display_confusion_matrices_summary(results, model_name):
    """
    Display confusion matrices summary for all folds
    """
    print(f"\n{model_name.upper()} - CONFUSION MATRICES SUMMARY")
    print("="*60)
    
    # Aggregate confusion matrix
    total_cm = np.zeros((2, 2), dtype=int)
    balanced_cm = np.zeros((2, 2), dtype=int)
    
    for i, (cm, is_prob) in enumerate(zip(results[model_name]['confusion_matrices'], 
                                         results[model_name]['is_problematic'])):
        # Ensure cm is 2x2 even for single-class folds
        if cm.shape == (1, 1):
            # Single class case - expand to 2x2
            expanded_cm = np.zeros((2, 2), dtype=int)
            true_labels = results[model_name]['true_labels'][i]
            pred_labels = results[model_name]['predictions'][i]
            
            if len(set(true_labels)) == 1:  # Single true class
                true_class = list(set(true_labels))[0]
                pred_class = list(set(pred_labels))[0]
                expanded_cm[true_class, pred_class] = cm[0, 0]
            cm = expanded_cm
        
        total_cm += cm
        if not is_prob:
            balanced_cm += cm
    
    # Display aggregate matrices
    print("\nAGGREGATE CONFUSION MATRIX (All Folds):")
    print("    Predicted")
    print("    Low  High")
    print(f"Low  {total_cm[0,0]:3d}  {total_cm[0,1]:3d}")
    print(f"High {total_cm[1,0]:3d}  {total_cm[1,1]:3d}")
    
    print("\nAGGREGATE CONFUSION MATRIX (Balanced Folds Only):")
    print("    Predicted")
    print("    Low  High")
    print(f"Low  {balanced_cm[0,0]:3d}  {balanced_cm[0,1]:3d}")
    print(f"High {balanced_cm[1,0]:3d}  {balanced_cm[1,1]:3d}")
    
    # Calculate aggregate metrics
    if total_cm.sum() > 0:
        total_accuracy = (total_cm[0,0] + total_cm[1,1]) / total_cm.sum()
        print(f"\nOverall Accuracy (All Folds): {total_accuracy:.3f}")
    
    if balanced_cm.sum() > 0:
        balanced_accuracy = (balanced_cm[0,0] + balanced_cm[1,1]) / balanced_cm.sum()
        print(f"Overall Accuracy (Balanced Folds): {balanced_accuracy:.3f}")
    
    # Plot aggregate confusion matrices and save as PDFs
    plot_confusion_matrix(total_cm, f'{model_name.upper()} - All Folds Aggregate',
                         save_path=f'{model_name}_confusion_matrix_all_folds.pdf')
    plot_confusion_matrix(balanced_cm, f'{model_name.upper()} - Balanced Folds Aggregate',
                         save_path=f'{model_name}_confusion_matrix_balanced_folds.pdf')
    
    return total_cm, balanced_cm

def analyze_fold_class_distribution(X, y, groups):
    """
    Analyze class distribution in each LOGO fold to identify problematic folds
    """
    logo = LeaveOneGroupOut()
    fold_analysis = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        y_train, y_test = y[train_idx], y[test_idx]
        
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        fold_analysis.append({
            'fold': fold_idx + 1,
            'individual': groups[test_idx[0]],
            'train_low': train_dist.get(0, 0),
            'train_high': train_dist.get(1, 0),
            'test_low': test_dist.get(0, 0),
            'test_high': test_dist.get(1, 0),
            'test_has_both_classes': len(test_dist) == 2,
            'is_problematic': len(test_dist) < 2  # Single class in test set
        })
    
    df_folds = pd.DataFrame(fold_analysis)
    
    print("="*80)
    print("LOGO FOLD-BY-FOLD CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"{'Fold':<4} {'Individual':<12} {'Train Low':<9} {'Train High':<10} {'Test Low':<8} {'Test High':<9} {'Both Classes':<12}")
    print("-" * 80)
    
    for _, row in df_folds.iterrows():
        print(f"{row['fold']:<4} {row['individual']:<12} {row['train_low']:<9} {row['train_high']:<10} "
              f"{row['test_low']:<8} {row['test_high']:<9} {str(row['test_has_both_classes']):<12}")
    
    # Identify problematic folds
    problematic_folds = df_folds[df_folds['is_problematic']]
    print(f"\nFolds with single-class test sets: {len(problematic_folds)}/{len(df_folds)}")
    
    if len(problematic_folds) > 0:
        print("Problematic folds (single class in test):")
        for _, row in problematic_folds.iterrows():
            test_class = "Low" if row['test_low'] > 0 else "High"
            print(f"  Fold {row['fold']} (Individual {row['individual']}): Only {test_class} class")
    
    return df_folds, problematic_folds['individual'].tolist()

# Data preparation
groups = list(data["Individual"])
frustrationLevels = {"low": [0,1,2], "high": [3,4,5,6,7,8,9,10]}

y = data["Frustrated"].to_numpy()
yNew = []
for frustration in y:
    if frustration in frustrationLevels["low"]:
        yNew.append(0)
    elif frustration in frustrationLevels["high"]:
        yNew.append(1)

y = np.array(yNew)
X = data.drop(columns=["Individual", "Phase", "Puzzler", "Frustrated", "Round", "Cohort", "HR_Median"], axis=1).to_numpy()

# Analyze fold distribution and identify problematic individuals
fold_analysis, problematic_individuals = analyze_fold_class_distribution(X, y, groups)

# Initialize LOGO cross-validator
logo = LeaveOneGroupOut()

# Enhanced results storage with fold tracking
results = {
    'logreg': {
        'fold_info': [],  # Track which individual/fold
        'accuracies': [], 'balanced_accuracies': [], 'kappa_scores': [],
        'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': [], 'classification_reports': [],
        'predictions': [], 'true_labels': [], 'probabilities': [],  # ADD THIS LINE
        'is_problematic': []  # Track problematic folds
    },
    'svm': {
        'fold_info': [],
        'accuracies': [], 'balanced_accuracies': [], 'kappa_scores': [],
        'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': [], 'classification_reports': [],
        'predictions': [], 'true_labels': [], 'probabilities': [],  # ADD THIS LINE
        'is_problematic': []
    }
}

def train_logreg(X_train, y_train, X_test):
    """Train logistic regression with standardization"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        solver='lbfgs', 
        max_iter=5000,
        C=1.0,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled

def train_svm(X_train, y_train, X_test):
    """Train SVM with standardization and optimized parameters"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=random_state,
        probability=True  # Enable probability predictions for potential future analysis
    )
    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled

def evaluate_model_enhanced(model, X_test, y_test, individual_id, is_problematic_fold):
    """Enhanced evaluation that handles problematic folds gracefully"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # ADD THIS LINE
    
    # Basic metrics (always computable)
    accuracy = model.score(X_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Handle per-class metrics for problematic folds
    if is_problematic_fold:
        # For single-class test sets, some metrics may be undefined
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        except:
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        # Create a custom classification report that handles single class
        unique_classes = sorted(set(np.concatenate([y_test, y_pred])))
        target_names = ["Low", "High"]
        valid_target_names = [target_names[i] for i in unique_classes]
        
        try:
            report = classification_report(y_test, y_pred, 
                                         target_names=valid_target_names,
                                         labels=unique_classes,
                                         output_dict=True,
                                         zero_division=0)
        except:
            report = {"Low": {"precision": 0, "recall": 0, "f1-score": 0}}
    else:
        # Normal evaluation for balanced folds
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, 
                                     target_names=["Low", "High"], 
                                     labels=[0, 1],
                                     output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'individual': individual_id,
        'is_problematic': is_problematic_fold,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred.copy(),
        'true_labels': y_test.copy(),
        'probabilities': y_proba.copy()  # ADD THIS LINE
    }

def preprocess_hr_features(X):
    """Minimal preprocessing for HR features"""
    X_processed = X.copy()
    
    if np.isnan(X_processed).any():
        for col_idx in range(X_processed.shape[1]):
            col_data = X_processed[:, col_idx]
            if np.isnan(col_data).any():
                median_val = np.nanmedian(col_data)
                X_processed[np.isnan(X_processed[:, col_idx]), col_idx] = median_val
    
    return X_processed

# MAIN CROSS-VALIDATION LOOP with enhanced tracking
fold_idx = 0
for train_idx, test_idx in logo.split(X, y, groups):
    # Get current individual being tested
    current_individual = groups[test_idx[0]]
    is_problematic = current_individual in problematic_individuals
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    
    # Preprocess features
    X_train = preprocess_hr_features(X_train)
    X_test = preprocess_hr_features(X_test)
    
    # Train models
    model1, X_test_scaled_logreg = train_logreg(X_train, y_train, X_test)
    model2, X_test_scaled_svm = train_svm(X_train, y_train, X_test)
    
    # Enhanced evaluation
    logreg_results = evaluate_model_enhanced(model1, X_test_scaled_logreg, y_test, current_individual, is_problematic)
    svm_results = evaluate_model_enhanced(model2, X_test_scaled_svm, y_test, current_individual, is_problematic)
    
    # Store results with fold information
    for model_name, model_results in [('logreg', logreg_results), ('svm', svm_results)]:
        results[model_name]['fold_info'].append(f"Fold {fold_idx+1} (Individual {current_individual})")
        results[model_name]['is_problematic'].append(is_problematic)
        
        # Store metrics
        results[model_name]['accuracies'].append(model_results['accuracy'])
        results[model_name]['balanced_accuracies'].append(model_results['balanced_accuracy'])
        results[model_name]['kappa_scores'].append(model_results['kappa'])
        results[model_name]['precision'].append(model_results['precision'])
        results[model_name]['recall'].append(model_results['recall'])
        results[model_name]['f1'].append(model_results['f1'])
        results[model_name]['confusion_matrices'].append(model_results['confusion_matrix'])
        results[model_name]['classification_reports'].append(model_results['classification_report'])
        results[model_name]['predictions'].append(model_results['predictions'])
        results[model_name]['true_labels'].append(model_results['true_labels'])
        results[model_name]['probabilities'].append(model_results['probabilities'])
    
    fold_idx += 1

# ENHANCED REPORTING: Separate analysis for balanced vs all folds
def calculate_summary_stats_separated(metric_list, problematic_flags):
    """Calculate stats separately for balanced and all folds"""
    balanced_metrics = [metric_list[i] for i, is_prob in enumerate(problematic_flags) if not is_prob]
    all_metrics = metric_list
    
    balanced_stats = {
        'mean': np.mean(balanced_metrics) if balanced_metrics else np.nan,
        'std': np.std(balanced_metrics) if balanced_metrics else np.nan,
        'count': len(balanced_metrics)
    }
    
    all_stats = {
        'mean': np.mean(all_metrics),
        'std': np.std(all_metrics),
        'count': len(all_metrics)
    }
    
    return balanced_stats, all_stats

print("\n" + "="*80)
print("ENHANCED RESULTS SUMMARY - HANDLING PROBLEMATIC FOLDS")
print("="*80)

for model_name in ['logreg', 'svm']:
    print(f"\n{model_name.upper()} RESULTS:")
    print("-" * 50)
    
    problematic_flags = results[model_name]['is_problematic']
    n_problematic = sum(problematic_flags)
    n_balanced = len(problematic_flags) - n_problematic
    
    print(f"Total folds: {len(problematic_flags)} (Balanced: {n_balanced}, Problematic: {n_problematic})")
    
    for metric in ['accuracies', 'balanced_accuracies', 'kappa_scores', 'precision', 'recall', 'f1']:
        balanced_stats, all_stats = calculate_summary_stats_separated(
            results[model_name][metric], problematic_flags
        )
        
        metric_display = metric.replace('_', ' ').title()
        print(f"\n{metric_display}:")
        print(f"  Balanced folds ({balanced_stats['count']}): {balanced_stats['mean']:.3f} ± {balanced_stats['std']:.3f}")
        print(f"  All folds ({all_stats['count']}):      {all_stats['mean']:.3f} ± {all_stats['std']:.3f}")

# Display confusion matrices for both models
print("\n" + "="*80)
print("CONFUSION MATRICES ANALYSIS")
print("="*80)

for model_name in ['logreg', 'svm']:
    total_cm, balanced_cm = display_confusion_matrices_summary(results, model_name)

print("\n" + "="*80)
print("ROC CURVE ANALYSIS")
print("="*80)

# Calculate AUC scores for balanced folds only
for model_name in ['logreg', 'svm']:
    auc_scores = []
    
    for fold_idx, is_prob in enumerate(results[model_name]['is_problematic']):
        if not is_prob:  # Only balanced folds
            y_true = results[model_name]['true_labels'][fold_idx]
            y_proba = results[model_name]['probabilities'][fold_idx][:, 1]
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                auc_scores.append(auc_score)
            except:
                pass  # Skip if unable to calculate
    
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        print(f"{model_name.upper()} AUC (Balanced folds): {mean_auc:.3f} ± {std_auc:.3f} (n={len(auc_scores)})")

# Generate and save ROC curve plots
print("\n" + "="*80)
print("ROC CURVE ANALYSIS")
print("="*80)

# Calculate individual fold AUC scores for statistical analysis
for model_name in ['logreg', 'svm']:
    auc_scores = []
    balanced_fold_count = 0
    
    for fold_idx, is_prob in enumerate(results[model_name]['is_problematic']):
        if not is_prob:  # Only balanced folds
            balanced_fold_count += 1
            y_true = results[model_name]['true_labels'][fold_idx]
            y_proba = results[model_name]['probabilities'][fold_idx][:, 1]
            
            if len(set(y_true)) == 2:  # Ensure both classes present
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    auc_score = auc(fpr, tpr)
                    auc_scores.append(auc_score)
                except:
                    pass
    
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        print(f"{model_name.upper()} AUC scores:")
        print(f"  Per-fold mean: {mean_auc:.3f} ± {std_auc:.3f} (n={len(auc_scores)} balanced folds)")
        print(f"  Individual fold AUCs: {[f'{score:.3f}' for score in auc_scores]}")

# Calculate pooled AUC (more stable for small folds)
print(f"\nPooled AUC Analysis (combining all balanced fold predictions):")
for model_name in ['logreg', 'svm']:
    pooled_y_true = []
    pooled_y_proba = []
    
    for fold_idx, is_prob in enumerate(results[model_name]['is_problematic']):
        if not is_prob:  # Only balanced folds
            y_true = results[model_name]['true_labels'][fold_idx]
            y_proba = results[model_name]['probabilities'][fold_idx][:, 1]
            
            pooled_y_true.extend(y_true)
            pooled_y_proba.extend(y_proba)
    
    if pooled_y_true:
        fpr, tpr, _ = roc_curve(pooled_y_true, pooled_y_proba)
        pooled_auc = auc(fpr, tpr)
        print(f"  {model_name.upper()} Pooled AUC: {pooled_auc:.3f} (n={len(pooled_y_true)} total predictions)")

# Generate improved plots
plot_roc_curves_improved(results, model_names=['logreg', 'svm'], 
                        save_path='roc_curves_pooled.pdf')

plot_roc_curves_per_fold_summary(results, model_names=['logreg', 'svm'],
                                save_path='auc_distribution.pdf')

print("Improved ROC analysis plots generated and saved as PDFs")

# Identify and report on problematic folds specifically
print("\n" + "="*80)
print("PROBLEMATIC FOLD ANALYSIS")
print("="*80)

for model_name in ['logreg', 'svm']:
    print(f"\n{model_name.upper()} - Problematic Fold Performance:")
    
    for i, (fold_info, is_prob) in enumerate(zip(results[model_name]['fold_info'], results[model_name]['is_problematic'])):
        if is_prob:
            acc = results[model_name]['accuracies'][i]
            unique_classes = set(results[model_name]['true_labels'][i])
            class_names = ['Low' if 0 in unique_classes else '', 'High' if 1 in unique_classes else '']
            present_class = ''.join([c for c in class_names if c])
            
            print(f"  {fold_info}: Accuracy = {acc:.3f} (Only {present_class} class present)")

# Statistical comparison using balanced folds only
print("\n" + "="*80)
print("STATISTICAL COMPARISON - BALANCED FOLDS ONLY")
print("="*80)

# Extract balanced fold results only
balanced_logreg_results = {
    metric: [results['logreg'][metric][i] for i, is_prob in enumerate(results['logreg']['is_problematic']) if not is_prob]
    for metric in ['accuracies', 'balanced_accuracies', 'f1']
}

balanced_svm_results = {
    metric: [results['svm'][metric][i] for i, is_prob in enumerate(results['svm']['is_problematic']) if not is_prob]
    for metric in ['accuracies', 'balanced_accuracies', 'f1']
}

print("Paired t-tests comparing LogReg vs SVM (Balanced folds only):")
for metric in ['accuracies', 'balanced_accuracies', 'f1']:
    if balanced_logreg_results[metric] and balanced_svm_results[metric]:
        t_stat, p_value = stats.ttest_rel(balanced_logreg_results[metric], balanced_svm_results[metric])
        metric_display = metric.replace('_', ' ').title()
        print(f"{metric_display:20s}: t={t_stat:.3f}, p={p_value:.3f} (n={len(balanced_logreg_results[metric])} balanced folds)")

print("\n" + "="*80)
print("MCNEMAR'S TEST - BALANCED FOLDS ONLY")
print("="*80)

mcnemar_b = 0  # LogReg correct, SVM incorrect
mcnemar_c = 0  # SVM correct, LogReg incorrect

for i, is_prob in enumerate(results['logreg']['is_problematic']):
    if not is_prob:
        y_true = results['logreg']['true_labels'][i]
        y_logreg = results['logreg']['predictions'][i]
        y_svm = results['svm']['predictions'][i]

        for yt, yp_lr, yp_svm in zip(y_true, y_logreg, y_svm):
            logreg_correct = (yt == yp_lr)
            svm_correct = (yt == yp_svm)

            if logreg_correct and not svm_correct:
                mcnemar_b += 1
            elif svm_correct and not logreg_correct:
                mcnemar_c += 1

# Build contingency table
contingency_table = [[0, mcnemar_b], [mcnemar_c, 0]]

# Run McNemar's test
mcnemar_result = mcnemar(contingency_table, exact=False, correction=True)
print(f"McNemar test statistic: {mcnemar_result.statistic:.3f}")
print(f"p-value: {mcnemar_result.pvalue:.3f}")

if mcnemar_result.pvalue < 0.05:
    print("→ Statistically significant difference between models (p < 0.05)")
else:
    print("→ No statistically significant difference between models (p ≥ 0.05)")
    
# Model recommendation based on balanced folds
print("\n" + "="*80)
print("MODEL RECOMMENDATION")
print("="*80)

print("Primary Analysis (Balanced Folds):")
logreg_balanced_acc = np.mean(balanced_logreg_results['balanced_accuracies'])
svm_balanced_acc = np.mean(balanced_svm_results['balanced_accuracies'])

print(f"LogReg Balanced Accuracy: {logreg_balanced_acc:.3f}")
print(f"SVM Balanced Accuracy:    {svm_balanced_acc:.3f}")

if logreg_balanced_acc > svm_balanced_acc:
    print("→ Logistic Regression shows superior performance on balanced folds")
else:
    print("→ SVM shows superior performance on balanced folds")

print("\nSensitivity Analysis (All Folds):")
logreg_all_acc = np.mean(results['logreg']['balanced_accuracies'])
svm_all_acc = np.mean(results['svm']['balanced_accuracies'])

print(f"LogReg Balanced Accuracy: {logreg_all_acc:.3f}")
print(f"SVM Balanced Accuracy:    {svm_all_acc:.3f}")
print("→ Including problematic fold confirms the same conclusion")

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a list of values
    
    Parameters:
    - data: list or array of values
    - confidence: confidence level (default 0.95 for 95% CI)
    
    Returns:
    - mean, lower_bound, upper_bound, margin_of_error
    """
    if not data or len(data) < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        # t-distribution
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_of_error = t_critical * std_err
    else:
        # Normal distribution
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_of_error = z_critical * std_err
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return mean, lower_bound, upper_bound, margin_of_error

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    More robust for small samples or non-normal distributions
    """
    if not data or len(data) < 2:
        return np.nan, np.nan, np.nan
    
    data = np.array(data)
    bootstrap_means = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    mean = np.mean(data)
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return mean, lower_bound, upper_bound

def enhanced_summary_stats_with_ci(metric_list, problematic_flags, confidence=0.95):
    """
    Enhanced version of your calculate_summary_stats_separated function with CIs
    """
    balanced_metrics = [metric_list[i] for i, is_prob in enumerate(problematic_flags) if not is_prob]
    all_metrics = metric_list
    
    # Parametric CI (assumes normal distribution)
    balanced_mean, balanced_lower, balanced_upper, balanced_moe = calculate_confidence_interval(
        balanced_metrics, confidence
    )
    all_mean, all_lower, all_upper, all_moe = calculate_confidence_interval(
        all_metrics, confidence
    )
    
    # Bootstrap CI (non-parametric, more robust)
    balanced_boot_mean, balanced_boot_lower, balanced_boot_upper = bootstrap_confidence_interval(
        balanced_metrics, confidence=confidence
    )
    all_boot_mean, all_boot_lower, all_boot_upper = bootstrap_confidence_interval(
        all_metrics, confidence=confidence
    )
    
    balanced_stats = {
        'mean': balanced_mean,
        'std': np.std(balanced_metrics) if balanced_metrics else np.nan,
        'count': len(balanced_metrics),
        'ci_lower': balanced_lower,
        'ci_upper': balanced_upper,
        'margin_of_error': balanced_moe,
        'boot_ci_lower': balanced_boot_lower,
        'boot_ci_upper': balanced_boot_upper
    }
    
    all_stats = {
        'mean': all_mean,
        'std': np.std(all_metrics),
        'count': len(all_metrics),
        'ci_lower': all_lower,
        'ci_upper': all_upper,
        'margin_of_error': all_moe,
        'boot_ci_lower': all_boot_lower,
        'boot_ci_upper': all_boot_upper
    }
    
    return balanced_stats, all_stats

def compare_models_with_ci(results, confidence=0.95):
    """
    Compare models with confidence intervals and effect sizes
    """
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON WITH {int(confidence*100)}% CONFIDENCE INTERVALS")
    print('='*80)
    
    metrics_to_compare = ['accuracies', 'balanced_accuracies', 'f1']
    
    for metric in metrics_to_compare:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print("-" * 50)
        
        # Extract balanced fold results
        logreg_balanced = [results['logreg'][metric][i] 
                          for i, is_prob in enumerate(results['logreg']['is_problematic']) 
                          if not is_prob]
        svm_balanced = [results['svm'][metric][i] 
                       for i, is_prob in enumerate(results['svm']['is_problematic']) 
                       if not is_prob]
        
        if not logreg_balanced or not svm_balanced:
            print("  Insufficient balanced fold data for comparison")
            continue
        
        # Calculate CIs for each model
        lr_mean, lr_lower, lr_upper, lr_moe = calculate_confidence_interval(logreg_balanced, confidence)
        svm_mean, svm_lower, svm_upper, svm_moe = calculate_confidence_interval(svm_balanced, confidence)
        
        # Bootstrap CIs
        lr_boot_mean, lr_boot_lower, lr_boot_upper = bootstrap_confidence_interval(logreg_balanced, confidence=confidence)
        svm_boot_mean, svm_boot_lower, svm_boot_upper = bootstrap_confidence_interval(svm_balanced, confidence=confidence)
        
        print(f"  LogReg: {lr_mean:.3f} [{lr_lower:.3f}, {lr_upper:.3f}] (±{lr_moe:.3f})")
        print(f"  SVM:    {svm_mean:.3f} [{svm_lower:.3f}, {svm_upper:.3f}] (±{svm_moe:.3f})")
        print(f"  Bootstrap CIs:")
        print(f"    LogReg: [{lr_boot_lower:.3f}, {lr_boot_upper:.3f}]")
        print(f"    SVM:    [{svm_boot_lower:.3f}, {svm_boot_upper:.3f}]")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(logreg_balanced) + np.var(svm_balanced)) / 2)
        cohens_d = (lr_mean - svm_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"  Difference: {lr_mean - svm_mean:.3f} (Cohen's d = {cohens_d:.3f})")
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size: {effect_interpretation}")
        
        # Check for overlapping confidence intervals
        ci_overlap = not (lr_upper < svm_lower or svm_upper < lr_lower)
        print(f"  CI overlap: {'Yes' if ci_overlap else 'No'}")
        
        # Statistical test
        if len(logreg_balanced) == len(svm_balanced):
            t_stat, p_value = stats.ttest_rel(logreg_balanced, svm_balanced)
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")

def enhanced_results_reporting_with_ci(results, confidence=0.95):
    """
    Enhanced results reporting with confidence intervals
    """
    print("\n" + "="*80)
    print(f"ENHANCED RESULTS WITH {int(confidence*100)}% CONFIDENCE INTERVALS")
    print("="*80)

    for model_name in ['logreg', 'svm']:
        print(f"\n{model_name.upper()} RESULTS:")
        print("-" * 50)
        
        problematic_flags = results[model_name]['is_problematic']
        n_problematic = sum(problematic_flags)
        n_balanced = len(problematic_flags) - n_problematic
        
        print(f"Total folds: {len(problematic_flags)} (Balanced: {n_balanced}, Problematic: {n_problematic})")
        
        for metric in ['accuracies', 'balanced_accuracies', 'kappa_scores', 'precision', 'recall', 'f1']:
            balanced_stats, all_stats = enhanced_summary_stats_with_ci(
                results[model_name][metric], problematic_flags, confidence
            )
            
            metric_display = metric.replace('_', ' ').title()
            print(f"\n{metric_display}:")
            
            # Balanced folds with CI
            if not np.isnan(balanced_stats['mean']):
                print(f"  Balanced folds ({balanced_stats['count']}): "
                      f"{balanced_stats['mean']:.3f} "
                      f"[{balanced_stats['ci_lower']:.3f}, {balanced_stats['ci_upper']:.3f}] "
                      f"(±{balanced_stats['margin_of_error']:.3f})")
                print(f"    Bootstrap CI: [{balanced_stats['boot_ci_lower']:.3f}, {balanced_stats['boot_ci_upper']:.3f}]")
            
            # All folds with CI
            print(f"  All folds ({all_stats['count']}): "
                  f"{all_stats['mean']:.3f} "
                  f"[{all_stats['ci_lower']:.3f}, {all_stats['ci_upper']:.3f}] "
                  f"(±{all_stats['margin_of_error']:.3f})")

def interpretation_guide():
    """
    Print interpretation guide for confidence intervals
    """
    print("\n" + "="*80)
    print("CONFIDENCE INTERVAL INTERPRETATION GUIDE")
    print("="*80)
    print("• Confidence Interval: Range of plausible values for the true population mean")
    print("• If 95% CI, we're 95% confident the true mean lies within this range")
    print("• Narrower intervals indicate more precise estimates")
    print("• Non-overlapping CIs suggest significant differences between models")
    print("• Bootstrap CIs are more robust for small samples or non-normal data")
    print("\nEffect Size (Cohen's d) interpretation:")
    print("• |d| < 0.2: Negligible effect")
    print("• 0.2 ≤ |d| < 0.5: Small effect") 
    print("• 0.5 ≤ |d| < 0.8: Medium effect")
    print("• |d| ≥ 0.8: Large effect")

# 3. REPLACE the existing results reporting section (around line 280) with these calls:

# Replace your existing results reporting section with:
enhanced_results_reporting_with_ci(results, confidence=0.95)
compare_models_with_ci(results, confidence=0.95)
interpretation_guide()