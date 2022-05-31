import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import display
from sklearn.metrics import plot_confusion_matrix


data = pd.read_csv('conversion_data_train.csv')
print('Set with labels (our train+test) :', data.shape)

# Basic stats
print("Number of rows : {}".format(data.shape[0]))
print()
print("Number of columns : {}".format(data.shape[1]))
print()

print("Display of dataset: ")
display(data.head())
print()

print("Basics statistics: ")
data_desc = data.describe(include='all')
display(data_desc)
print()

print("Percentage of missing values: ")
display(100*data.isnull().sum()/data.shape[0])

#removing outliers
data = data[data.loc[:,'age'] < 100] 
data.describe(include ='all')

#repartition des conversion rate par new user:
plt.figure(figsize=[12,5])
sns.countplot(data['converted'], hue = data['new_user']).set(title='Part of conversion per new user')


sns.relplot(data=data, x="total_pages_visited", y="converted")

# Univariate analysis
#Distribution of quantitative variables

from plotly.subplots import make_subplots

# Distribution of each numeric variable
num_features = ['age','total_pages_visited']
fig1 = make_subplots(rows = len(num_features), cols = 1, subplot_titles = num_features)
for i in range(len(num_features)):
    fig1.add_trace(
        go.Histogram(
            x = data[num_features[i]], nbinsx = 4),
        row = i + 1,
        col = 1)
fig1.update_layout(
        title = go.layout.Title(text = "Distribution of quantitative variables", x = 0.5), showlegend = False, 
            autosize=False, height=500)
fig1.show()


# Univariate analysis

# Barplot of each qualitative variable

cat_features = ['country','new_user','source','converted']
fig2 = make_subplots(rows = len(cat_features), cols = 1, subplot_titles = cat_features)
for i in range(len(cat_features)):
    
    x_coords = data[cat_features[i]].value_counts().index.tolist()
    y_coords = data[cat_features[i]].value_counts().tolist()
    
    fig2.add_trace(
        go.Bar(
            x = x_coords,
            y = y_coords),
        row = i + 1,
        col = 1)
fig2.update_layout(
        title = go.layout.Title(text = "Barplot of qualitative variables", x = 0.5), showlegend = False, 
            autosize=False, height=500)
fig2.show()


# Correlation matrix
corr_matrix = data.corr()

import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(corr_matrix.values,
                                  x = corr_matrix.columns.values.tolist(),
                                  y = corr_matrix.index.values.tolist())


fig.show()



#Let's try a first basic model : simple logistic regression with only one feature.
#We choose the total_pages_visited because we just noticed that it is strongly correlated to the converted.
# Separate target variable Y from features X
print("Separating labels from features...")
features_list = ["total_pages_visited"]
target_variable = "converted"

X = data.loc[:,features_list]
Y = data.loc[:,target_variable]

print("...Done.")
print()

print('Y : ')
print(Y.head())
print()
print('X :')
print(X.head())


# Divide dataset Train set & Test set 
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=0,stratify = Y)
print("...Done.")
print()


# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.tolist()
Y_test = Y_test.tolist()
print("...Done")

print(X_train[0:5,:])
print(X_test[0:2,:])
print()
print(Y_train[0:5])
print(Y_test[0:2])

# Standardizing numerical features
print("Standardizing numerical features...")
print()
print(X_train[0:5,:])

# Normalization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
print("...Done")
print(X_train[0:5,:])


# Train model with a logistic regression:
print("Train model...")
classifier  = LogisticRegression()
classifier.fit(X_train, Y_train)
print("...Done.")

# Predictions on training set
print("Predictions on training set...")
Y_train_pred = classifier.predict(X_train)
print("...Done.")
print(Y_train_pred)
print()


# Standardizing numerical features on Test set:
print("Standardizing numerical features...")
print()
print(X_test[0:5,:])

X_test = scaler.transform(X_test)
print("...Done")
print(X_test[0:5,:])

# Predictions on test set
print("Predictions on test set...")
Y_test_pred = classifier.predict(X_test)
print("...Done.")
print(Y_test_pred)
print()


# Print scores
print("accuracy on training set : ", accuracy_score(Y_train, Y_train_pred))
print("accuracy on test set : ", accuracy_score(Y_test, Y_test_pred))
print()

print("f1-score on training set : ", f1_score(Y_train, Y_train_pred))
print("f1-score on test set : ", f1_score(Y_test, Y_test_pred))
print()

#le modèle a une bonne accuracy (taux de prédiction exacte très correct) 
# mais le F1 score (mesure de tous les cas positifs) n'est pas forcément bon même si pas si mauvais il fait quelques erreurs quand même (pas assez de features),
#  modèle pas assez complexe

# Visualize confusion matrices on train set
_ , ax = plt.subplots() # Get subplot from matplotlib
ax.set(title="Confusion Matrix on Train set") # Set a title that we will add into ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(classifier, X_train, Y_train, ax=ax) # ConfusionMatrixDisplay from sklearn
plt.show()

# Visualize confusion matrices on test set
_ , ax = plt.subplots() # Get subplot from matplotlib
ax.set(title="Confusion Matrix on Test set") # Set a title that we will add into ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(classifier, X_test, Y_test, ax=ax) # ConfusionMatrixDisplay from sklearn
plt.show()

#Train a multivariate model by adding all the others features:
#removing outliers
data = pd.read_csv('conversion_data_train.csv')
data = data[data.loc[:,'age'] < 100] 

# Separate target variable Y from features X , on prend pas country:
print("Separating labels from features...")

features_list = [ "country","age", "new_user","source","total_pages_visited"]
target_variable = "converted"

X = data.loc[:,features_list]
Y = data.loc[:,target_variable]

print("...Done.")
print()

print('Y : ')
print(Y.head())
print()
print('X :')
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0,stratify=Y)


numeric_features = [1, 4] # Choose which column index we are going to scale
numeric_transformer = StandardScaler()

categorical_features = [0, 2, 3] # Choose which column index we are going to encode
categorical_transformer = OneHotEncoder()

# Apply ColumnTransformer to create a pipeline that will apply the above preprocessing
featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)
        ]
    )

X_train = featureencoder.fit_transform(X_train)
print("...Done.")
print("#### X_train AFTER preprocessing ####")
print(X_train[0:5,:]) # print first 5 rows (not using iloc since now X_train became a numpy array)
print()

# Training model 
print("Training model...")
classifier =LogisticRegression() # Instanciate model 
classifier.fit(X_train, y_train) # Fit model
print("...Done.")

# Predictions on training set
print("Predictions on train set...")
y_train_pred = classifier.predict(X_train)
print("...Done.")
print()

print("#### First five predictions on TRAIN set ####")
print(y_train_pred[0:5])


### Test pipeline ###
print("--- Test pipeline ---") 

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()

X_test = featureencoder.transform(X_test)
print("...Done.")

print("#### X_test AFTER preprocessing ####")
print(X_test[0:5,:])
print()


# Predictions on training set
print("Predictions on test set...")
y_test_pred = classifier.predict(X_test)
print("...Done.")
print()

print("#### First five predictions on TEST set ####")
print(y_train_pred[0:5])


### Assessment of performances ###
print("--- Assessment of performances ---")

# Plot confusion matrix
cm = plot_confusion_matrix(classifier, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ") # Simply to set a title
plt.show() # Show graph
print("accuracy-score on train set : ", classifier.score(X_train, y_train))


cm = plot_confusion_matrix(classifier, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show() # Show graph
print("accuracy-score on test set : ", classifier.score(X_test, y_test))

print("-----------")

print("f1-score on training set : ", f1_score(y_train, y_train_pred))
print("f1-score on test set : ", f1_score(y_test, y_test_pred))
print()

#F1 & accuracy scores improved on train & test set


# Check coefficients 

print("coefficients are: ", classifier.coef_) 

# Access transformers from feature_encoder
print("All transformers are: ", featureencoder.transformers_)

# Access one specific transformer
print("One Hot Encoder transformer is: ", featureencoder.transformers_[0][1])

# Print categories
categorical_column_names = featureencoder.transformers_[0][1].categories_
categorical_column_names = np.concatenate(categorical_column_names).ravel()
print("Categorical columns are: ", categorical_column_names)

# Print numerical columns
numerical_column_names = X.iloc[:, numeric_features].columns # using the .columns attribute gives us the name of the column 
print("numerical columns are: ", numerical_column_names)

# Append all columns 
all_column_names = np.append(categorical_column_names, numerical_column_names)
print("All column names are: ", all_column_names)

# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients":classifier.coef_.squeeze() # CAREFUL HERE. We need to access first index of our list because 
                                            # Data need to be 1 dimensional
                                            # That's what .squeeze()
})

feature_importance 


# Set coefficient to absolute values to rank features
feature_importance["coefficients"] = feature_importance["coefficients"].abs()

# Visualize ranked features using seaborn
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=3) # Resize graph


 #le modèle a une très bonne accuracy (taux de prédiction exacte très correct)
 # le F1 score s'est amélioré en ajoutant des features (le modele a réussi à faire moins d'erreurs )
 # F1 se rapproche de 1        


# Concatenate our train and test set to train your best classifier on all data with labels
X = np.append(X_train,X_test,axis=0)
Y = np.append(y_train,y_test)

classifier.fit(X,Y)

# Read data without labels
data_without_labels = pd.read_csv('conversion_data_test.csv')
print('Prediction set (without labels) :', data_without_labels.shape)

# Warning : check consistency of features_list (must be the same than the features 
# used by your best classifier)
features_list = ["country","age", "new_user","source","total_pages_visited"]
X_without_labels = data_without_labels.loc[:, features_list]

# 'new_user' is a categorical feature in fact 'yes' or 'no' not a numerical one
#so we amend the list of categorical features to:
categorical_features = ['country','source', 'new_user']
numeric_features = ['total_pages_visited', 'age']


print("Encoding categorical features and standardizing numerical features...")

X_without_labels = featureencoder.transform(X_without_labels)
print("...Done")
print(X_without_labels[0:5,:])

# Make predictions and dump to file
data = {
    'converted': classifier.predict(X_without_labels)
}

Y_predictions = pd.DataFrame(columns=['converted'],data=data)
Y_predictions.to_csv('conversion_data_test_predictions_RICHARDGH-M2.csv', index=False)

#On va essayer un modele de Régression logistique sans le critère 'total_pages_visited':


data = pd.read_csv('conversion_data_train.csv')
print('Set with labels (our train+test) :', data.shape)
#removing outliers
data = data[data.loc[:,'age'] < 100] 

# Separate target variable Y from features X , on prend pas country:
print("Separating labels from features...")

features_list = [ "country","age", "new_user","source"]
target_variable = "converted"

X = data.loc[:,features_list]
Y = data.loc[:,target_variable]

print("...Done.")
print()

print('Y : ')
print(Y.head())
print()
print('X :')
print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0,stratify=Y)


numeric_features = [1] # Choose which column index we are going to scale
numeric_transformer = StandardScaler()

categorical_features = [0, 2, 3] # Choose which column index we are going to encode
categorical_transformer = OneHotEncoder()

# Apply ColumnTransformer to create a pipeline that will apply the above preprocessing
featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)
        ]
    )

X_train = featureencoder.fit_transform(X_train)
print("...Done.")
print("#### X_train AFTER preprocessing ####")
print(X_train[0:5,:]) # print first 5 rows (not using iloc since now X_train became a numpy array)
print()

# Training model 
print("Training model...")
classifier =LogisticRegression() # Instanciate model 
classifier.fit(X_train, y_train) # Fit model
print("...Done.")


# Predictions on training set
print("Predictions on train set...")
y_train_pred = classifier.predict(X_train)
print("...Done.")
print()

print("#### First five predictions on TRAIN set ####")
print(y_train_pred[0:5])


### Test pipeline ###
print("--- Test pipeline ---") 

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()

X_test = featureencoder.transform(X_test)
print("...Done.")

print("#### X_test AFTER preprocessing ####")
print(X_test[0:5,:])
print()


# Predictions on training set
print("Predictions on test set...")
y_test_pred = classifier.predict(X_test)
print("...Done.")
print()

print("#### First five predictions on TEST set ####")
print(y_train_pred[0:5])



### Assessment of performances ###
print("--- Assessment of performances ---")

# Plot confusion matrix
cm = plot_confusion_matrix(classifier, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ") # Simply to set a title
plt.show() # Show graph
print("accuracy-score on train set : ", classifier.score(X_train, y_train))


cm = plot_confusion_matrix(classifier, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show() # Show graph
print("accuracy-score on test set : ", classifier.score(X_test, y_test))

print("-----------")

print("f1-score on training set : ", f1_score(y_train, y_train_pred))
print("f1-score on test set : ", f1_score(y_test, y_test_pred))
print()


# Check coefficients 

print("coefficients are: ", classifier.coef_) 
print("Constant is: ", classifier.intercept_)

# Access transformers from feature_encoder
print("All transformers are: ", featureencoder.transformers_)

# Access one specific transformer
print("One Hot Encoder transformer is: ", featureencoder.transformers_[0][1])

# Print categories
categorical_column_names = featureencoder.transformers_[0][1].categories_
categorical_column_names = np.concatenate(categorical_column_names).ravel()
print("Categorical columns are: ", categorical_column_names)

# Print numerical columns
numerical_column_names = X.iloc[:, numeric_features].columns # using the .columns attribute gives us the name of the column 
print("numerical columns are: ", numerical_column_names)

# Append all columns 
all_column_names = np.append(categorical_column_names, numerical_column_names)
print("All column names are: ", all_column_names)

# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients":classifier.coef_.squeeze() # CAREFUL HERE. We need to access first index of our list because 
                                            # Data need to be 1 dimensional
                                            # That's what .squeeze()
})

feature_importance 



# Set coefficient to absolute values to rank features
feature_importance["coefficients"] = feature_importance["coefficients"].abs()

# Visualize ranked features using seaborn
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=3) # Resize graph


#Le modèle a perdu en accuracy le taux de prédication correct a baissé mais surtout le F1 score est tombé à 0 sur le train et le test 
# car il n'a pu prédire aucun positif du tout (conversion rate = "1") 
# sans la variable 'total_pages_visited'


#DECISION TREE MODEL:

data = pd.read_csv('conversion_data_train.csv')
print('Set with labels (our train+test) :', data.shape)
#removing outliers
data = data[data.loc[:,'age'] < 100]

# Separate target variable Y from features X:
print("Separating labels from features...")

features_list = [ "country","age", "new_user","source","total_pages_visited"]
target_variable = "converted"

X = data.loc[:,features_list]
Y = data.loc[:,target_variable]

print("...Done.")
print()

print('Y : ')
print(Y.head())
print()
print('X :')
print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0,stratify=Y)

numeric_features = [1,4] # Choose which column index we are going to scale
numeric_transformer = StandardScaler()

categorical_features = [0, 2, 3] # Choose which column index we are going to encode
categorical_transformer = OneHotEncoder()

# Apply ColumnTransformer to create a pipeline that will apply the above preprocessing
featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)
        ]
    )

X_train = featureencoder.fit_transform(X_train)
print("...Done.")
print("#### X_train AFTER preprocessing ####")
print(X_train[0:5,:]) # print first 5 rows (not using iloc since now X_train became a numpy array)
print()


# Training model
print("Training model...")
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
print("...Done.")


# Predictions on training set
print("Predictions on train set...")
y_train_pred = DT.predict(X_train)
print("...Done.")
print()


### Test pipeline ###
print("--- Test pipeline ---") 

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()

X_test = featureencoder.transform(X_test)
print("...Done.")

print("#### X_test AFTER preprocessing ####")
print(X_test[0:5,:])
print()

# Predictions on test set
print("Predictions on test set...")
y_test_pred = DT.predict(X_test)
print("...Done.")
print()


## Assessment of performances ###
print("--- Assessment of performances ---")


# Plot confusion matrix
cm = plot_confusion_matrix(DT, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ") # Simply to set a title
plt.show() # Show graph
print("accuracy-score on train set : ", DT.score(X_train, y_train))


cm = plot_confusion_matrix(DT, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show() # Show graph
print("accuracy-score on test set : ", DT.score(X_test, y_test))
print('-----')

print("f1-score on training set : ", f1_score(y_train, y_train_pred))
print("f1-score on test set : ", f1_score(y_test, y_test_pred))
print()


# Check coefficients 

print("coefficients are: ", DT.feature_importances_) 

# Access transformers from feature_encoder
print("All transformers are: ", featureencoder.transformers_)

# Access one specific transformer
print("One Hot Encoder transformer is: ", featureencoder.transformers_[0][1])

# Print categories
categorical_column_names = featureencoder.transformers_[0][1].categories_
categorical_column_names = np.concatenate(categorical_column_names).ravel()
print("Categorical columns are: ", categorical_column_names)

# Print numerical columns
numerical_column_names = X.iloc[:, numeric_features].columns # using the .columns attribute gives us the name of the column 
print("numerical columns are: ", numerical_column_names)

# Append all columns 
all_column_names = np.append(categorical_column_names, numerical_column_names)
print("All column names are: ", all_column_names)


# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients": DT.feature_importances_
                                        
})

feature_importance


# Visualize ranked features using seaborn
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=3) # Resize graph


#Le modèle conserve une très bonne accuracy et améliore son F1 score sur le train set, cependant il diminue sur le Test Set 
# The most important feature to subscribe to the newsletter is the total_pages_visited

#PERFORM GRID SEARCH , to get the best parameter for our model:

# Perform Grid Search on model to get the best parameter for our model:
print("Grid search...")

# Grid of values to be tested
params = {
    'max_depth': [4, 6, 8, 10], #nombre de couches maximales pour le DT
    'min_samples_leaf': [1, 2, 5], #specifies the minimum number of samples required to be at a leaf node
    'min_samples_split': [2, 4, 8] #min_samples_split specifies the minimum number of samples required to split an internal node
}



gridsearch = GridSearchCV(DT, param_grid = params, cv = 3) # cv : the number of folds to be used for CV , on divise les datas en 3
gridsearch.fit(X_train, y_train)
print("...Done.")
print("Best hyperparameters : ", gridsearch.best_params_)
print("Best validation accuracy : ", gridsearch.best_score_)


# Predictions on training set
print("Predictions on training set...")
y_train_pred = gridsearch.predict(X_train)
print("...Done.")
print(y_train_pred)
print()


# Predictions on test set
print("Predictions on training set...")
y_test_pred = gridsearch.predict(X_test)
print("...Done.")
print(y_test_pred)
print()

# Print scores
print("accuracy on training set : ", accuracy_score(y_train, y_train_pred))
print("accuracy on test set : ", accuracy_score(y_test, y_test_pred))
print()

print("f1-score on training set : ", f1_score(y_train, y_train_pred))
print("f1-score on test set : ", f1_score(y_test, y_test_pred))
print()


#We can see that the score decreased on train set 
# but increased on the test set after the grid search (pick the best possible hyper-parameters)
# on DT hyperparameter optimization worked




#our best model is the logistic regression with alle features to predict

#Notre meilleur modèle est la régression logistique avec toutes les features pour faire la prédiction, on voit que le taux de conversion dépend pour beaucoup du nombre de pages visitées, Le modele de régression logistique 
#avec toutes les features en variables explicatives donne les meilleurs scores.
## Bilan:

#Notre meilleur modèle est la régression logistique avec toutes les features pour faire la prédiction,
#on voit que le taux de conversion dépend pour beaucoup du nombre de pages visitées,
#Le modele de régression logistique avec toutes les features en variables explicatives donne les meilleurs scores.

#Afin d'améliorer le conversion rate à la souscription à la newsletter ,il faudrait que la page d'accueil soit plus conviviale
#et propose des liens plus visibles pour visiter d'autres pages et non une page d'accueil qui met en avant la souscription directement,
#ce qui pourrait pousser l'utilisateur à souscrire plus 'facilement' après avoir consulté plus de pages













































