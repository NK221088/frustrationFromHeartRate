import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

###########################################
# Loading Data
###########################################

data = pd.read_csv("HR_data.csv")
data = data.drop(data.columns[0], axis=1) # Remove the row index

###########################################
# Summary statistics and Correlation Matrix for all data
###########################################

summaryColumns = list(data.columns)
toBeRemoved = ["Individual", "Phase", "Round", "Cohort", "Puzzler"]
summaryColumns = [col for col in summaryColumns if col not in toBeRemoved]

print(data[summaryColumns].describe())

# Calculate correlation matrix
correlation_matrix = data[summaryColumns].corr()

# Set up the matplotlib figure with better size and DPI
plt.figure(figsize=(12, 8), dpi=1200)

# Create a mask for the upper triangle (optional - shows only lower triangle)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

axis_corr = sns.heatmap(
    correlation_matrix,
    mask=mask,  # Remove this line if you want the full matrix
    annot=True,  # Show correlation values
    fmt='.2f',   # Format numbers to 2 decimal places
    vmin=-1, 
    vmax=1, 
    center=0,
    cmap='RdBu_r',  # Red-Blue reversed (blue=positive, red=negative)
    square=True,
    linewidths=0.5,  # Add gridlines
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}  # Improve colorbar
)

plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlationMatrix.pdf', dpi=1200, bbox_inches='tight')

###########################################
# Summary statistics and Correlation Matrix for phases
###########################################

phaseOnedf = data[data["Phase"] == "phase1"]
phaseTwodf = data[data["Phase"] == "phase2"]
phaseThreedf = data[data["Phase"] == "phase3"]

print(phaseOnedf[summaryColumns].describe())
print(phaseTwodf[summaryColumns].describe())
print(phaseThreedf[summaryColumns].describe())

# Calculate correlation matrices
correlation_matrix_phase_one = phaseOnedf[summaryColumns].corr()
correlation_matrix_phase_two = phaseTwodf[summaryColumns].corr()
correlation_matrix_phase_three = phaseThreedf[summaryColumns].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix_phase_one, dtype=bool))

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Create heatmaps without individual colorbars
hm1 = sns.heatmap(correlation_matrix_phase_one, mask=mask, annot=True, fmt='.2f',
                  vmin=-1, vmax=1, center=0, cmap='RdBu_r', square=True,
                  linewidths=0.5, cbar=False, ax=axes[0])
hm2 = sns.heatmap(correlation_matrix_phase_two, mask=mask, annot=True, fmt='.2f',
                  vmin=-1, vmax=1, center=0, cmap='RdBu_r', square=True,
                  linewidths=0.5, cbar=False, ax=axes[1])
hm3 = sns.heatmap(correlation_matrix_phase_three, mask=mask, annot=True, fmt='.2f',
                  vmin=-1, vmax=1, center=0, cmap='RdBu_r', square=True,
                  linewidths=0.5, cbar=False, ax=axes[2])

# Set titles and labels
titles = ['Phase 1', 'Phase 2', 'Phase 3']
for i, ax in enumerate(axes):
    ax.set_title(f'{titles[i]} Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    if i == 0:
        ax.set_ylabel('Features', fontsize=12)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])  # remove y-axis tick labels for axes[1] and axes[2]
    ax.tick_params(axis='x', rotation=45)

# Adjust layout and leave space for the colorbar
plt.tight_layout(rect=[0, 0, 0.93, 1])  # reserve space on the right (7% margin)

# Add a single shared vertical colorbar
cbar = fig.colorbar(hm3.collections[0], ax=axes.ravel().tolist(),
                    orientation='vertical', fraction=0.02, pad=0.02,
                    label='Correlation Coefficient')

# Save and show
plt.savefig('correlationMatrixPhases.pdf', dpi=1200, bbox_inches='tight')

###########################################
# Feature Distributions - All features as histograms in one PDF
###########################################

# Create histograms for all features and save to a single PDF
with PdfPages('feature_distributions.pdf') as pdf:
    # Calculate number of rows and columns for subplots
    n_features = len(summaryColumns)
    n_cols = 3  # 3 columns per page
    n_rows = 4  # 4 rows per page
    plots_per_page = n_cols * n_rows
    
    # Split features into pages
    for page_start in range(0, n_features, plots_per_page):
        page_end = min(page_start + plots_per_page, n_features)
        page_features = summaryColumns[page_start:page_end]
        
        # Calculate actual rows needed for this page
        actual_plots = len(page_features)
        actual_rows = (actual_plots + n_cols - 1) // n_cols
        
        # Create figure for this page
        fig, axes = plt.subplots(nrows=actual_rows, ncols=n_cols, 
                                figsize=(15, 4*actual_rows), dpi=300)
        
        # Handle case where we have only one row
        if actual_rows == 1:
            axes = axes.reshape(1, -1) if actual_plots > 1 else [axes]
        
        # Flatten axes array for easier indexing
        axes_flat = axes.flatten() if actual_plots > 1 else axes
        
        # Create histograms
        for i, feature in enumerate(page_features):
            ax = axes_flat[i] if actual_plots > 1 else axes_flat
            
            # Create histogram
            data[feature].hist(bins=20, edgecolor='black', alpha=0.7, ax=ax)
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Add basic statistics as text
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        # Hide unused subplots
        if actual_plots > 1:
            for i in range(actual_plots, len(axes_flat)):
                axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory

print(f"Feature distributions saved to 'feature_distributions.pdf'")

###########################################
# Looking at distribution for the frustration output
###########################################

data['Frustrated'].hist(bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Frustration Output', fontsize=16, fontweight='bold')
plt.xlabel('Frustration Level', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('frustration_distribution.pdf', dpi=1200, bbox_inches='tight')