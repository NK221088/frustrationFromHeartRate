from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle

from dataInspection import data

random_state = 29  # Set a random state for reproducibility

groups = list(data["Individual"])
y = data["Frustrated"].to_numpy()
X = data.drop(columns=["Individual", "Phase", "Puzzler", "Frustrated", "Round", "Cohort"], axis=1).to_numpy()

logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, groups):
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    
    # Train model
    
print("e")