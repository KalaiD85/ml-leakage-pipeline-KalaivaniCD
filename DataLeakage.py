#Task 1: Reproduce and Identify Leakage 
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
scalar = StandardScaler()
# Incorrect scaling: scaling before splitting
x_scaled = scalar.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# issue is data leakage:
#By scaling the entire dataset before splitting, the test set statistics are used in transforming the training set.
#information from the test set has “leaked” into the training process.
#As a result, the reported test accuracy is biased — it looks better but may fail in a real-world scenario.

#Task 2: Fix the Workflow Using a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train,X_test,y_train,y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()),
                    ('Reg', LogisticRegression())])

# Perform 5-fold cross-validation on training set
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Report mean accuracy and standard deviation
mean_accuracy = cv_scores.mean().round(2)
std_accuracy = cv_scores.std().round(2)

print(f"5-Fold CV Mean Accuracy: {mean_accuracy}")
print(f"5-Fold CV Standard Deviation: {std_accuracy}")

#Task 3: Experiment with Decision Tree Depth 
from sklearn.tree import DecisionTreeClassifier

for depth in [1, 5, 20]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    print(f"Depth {depth:2d}  | Train: {train_acc:.2f}  Test: {test_acc:.2f}")

#The DecisionTreeClassifier results show that depth=5 provides the best balance: 
#it achieves high test accuracy without overfitting, 
#unlike depth=20 which memorizes the training set. 
#Depth=1 is too simple and underfits.

