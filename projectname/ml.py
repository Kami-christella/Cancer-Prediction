import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Use raw string to specify the file path
# file_path = r"C:\Users\Kami\Desktop\AUCA STUDIES\Big Data\Exam preparation\two\project1\Data.xlsx"
file_path=r"C:\Users\Kami\Desktop\AUCA STUDIES\Big Data\Final Exam\projectname\data.xlsx"
# Load the dataset
try:
    dataset = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
    print(dataset.describe())
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# 1. Remove duplicates
dataset.drop_duplicates(inplace=True)

# 2. Handle null values
radius_mean = dataset["radius_mean"].mean()
texture_mean= dataset["texture_mean"].mean()
perimeter_mean = dataset["perimeter_mean"].mean()
area_mean= dataset["area_mean"].mean()
smoothness_mean = dataset["smoothness_mean"].mean()

compactness_mean= dataset["compactness_mean"].mean()
concavity_mean = dataset["concavity_mean"].mean()
concave_points_mean= dataset["concave_points_mean"].mean()
symmetry_mean = dataset["symmetry_mean"].mean()
fractal_dimension_mean= dataset["fractal_dimension_mean"].mean()

dataset.fillna({"radius_mean": radius_mean}, inplace=True)
dataset.fillna({"texture_mean": texture_mean}, inplace=True)
dataset.fillna({"perimeter_mean": perimeter_mean}, inplace=True)
dataset.fillna({"area_mean": area_mean}, inplace=True)
dataset.fillna({"smoothness_mean": smoothness_mean}, inplace=True)

dataset.fillna({"compactness_mean": compactness_mean}, inplace=True)
dataset.fillna({"concavity_mean": concavity_mean}, inplace=True)
dataset.fillna({"concave_points_mean": concave_points_mean}, inplace=True)
dataset.fillna({"symmetry_mean": symmetry_mean}, inplace=True)
dataset.fillna({"fractal_dimension_mean": fractal_dimension_mean}, inplace=True)

# 3. Handle wrong data formats (if any) - specific examples needed
# 4. Correct wrong data values


# Prepare the data for machine learning
X = dataset.drop(columns=['id','diagnosis', 'radius_se', 'texture_se', 'perimeter_se', 'area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'])
y = dataset['diagnosis']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize models
Decision_tree_model = DecisionTreeClassifier()
Logistic_regression_Model = LogisticRegression(solver='lbfgs', max_iter=10000)
SVM_model = svm.SVC(kernel='linear')
RF_model = RandomForestClassifier(n_estimators=100)

# Train models
Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)

# Predict with models
DT_Prediction = Decision_tree_model.predict(x_test)
LR_Prediction = Logistic_regression_Model.predict(x_test)
SVM_Prediction = SVM_model.predict(x_test)
RF_Prediction = RF_model.predict(x_test)

# Evaluate models
DT_score = accuracy_score(y_test, DT_Prediction)
LR_score = accuracy_score(y_test, LR_Prediction)
SVM_score = accuracy_score(y_test, SVM_Prediction)
RF_score = accuracy_score(y_test, RF_Prediction)

print("Decision Tree accuracy =", DT_score * 100, "%")
print("Logistic Regression accuracy =", LR_score * 100, "%")
print("Support Vector Machine accuracy =", SVM_score * 100, "%")
print("Random Forest accuracy =", RF_score * 100, "%")

# Make a prediction
predict = RF_model.predict([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871]])
print("Prediction:", predict)

# Save the model
joblib.dump(RF_model, 'my.joblib')
