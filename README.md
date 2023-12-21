# Investigating the Iris Dataset # 
#### By Emily Darby ####

## Background ##

The iris dataset is a model 'toy' dataset within the scikit-learn website, which provides data about the size of three different species of iris. The species of interest include Setosa, Versicolor and Virginica. A total of 150 plants were measured in this study, with an even spread of 50 plants from each of the three species type. Four key measurements were taken from each plant, including the petal length, petal width, sepal length and sepal width. 
The following data exploration was conducted to identify whether these different measurements could be used to predict the iris species. This may offer a more facile method of identification of iris species in the wild, by performing a brief assessment by measuring petal length or width to determine the identity of an unknown iris plant of interest.

## Methods ##

Since the dataset explored in this code is a model 'toy' dataset, the iris dataset can be easily loaded from scikit-learn following the code included in the **Iris.ipynb** file. The dataset was initially named **" iris "**, however the code had to be modified to present the data in a tabular format, therefore the dataset was subsequently renamed as **" iris_df "**.

 ### To run the code within the **Iris.ipynb** file, open the **Iris.ipynb** file in JupyterLab and run the code according to the chronological the instructions / pipeline. ###
 The code for loading the required software packages to fully explore the data as intended were all included within the code at each necessary point throughout the exploratory pipeline. 
These packages include:
- pandas
- sklearn
- matplotlib
- seaborn

#### The following pipeline was created to explore the dataset: ####

##### 1. Loading in the Dataset #####
The model 'toy' dataset, *Iris*, was loaded into JupyterLab and then reformatted using the following code:

 - from sklearn.datasets import load_iris
 - pd.DataFrame(X, columns=iris.feature_names)
 - iris.target

##### 2. Visualising the Data #####
To assess the size of the dataset and initially explore the different variables included, the following functions in the code:

- .count()
- .plot()
- .mean()

##### 3. Filtering the Data #####
Since three different iris species were investigated within the dataset, it seemed logical to create filters to subset the different species, as this may aid in comparing features of the different species which may facilitate distinctly identifying each species. The filter for each species was named according to the species type:

- Setosa = iris_df[iris_df['species'] == "setosa"]
- Versicolor = iris_df[iris_df['species'] == "versicolor"]
- Virginica = iris_df[iris_df['species'] == "virginica"]

##### 4. Data Exploration: Correlation Analysis #####
To assess which variables may be strongly correlated with one another, a large correlation scatterplot was created to visualise these potential associations. This enabled selection of variables to be used in a later linear regression model:

- from pandas.plotting import scatter_matrix

The variables which appeared to be strongly correlated were then separately plotted, with each species plotted in a different colour, using the following code:

- import matplotlib.pyplot as plt

- _, ax = plt.subplots()
- scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
- ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
- _ = ax.legend(
- scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Species")

##### 5 & 6. Identification of Potential Predictor Variables #####
The distribution of samples within each species type according to each separate variable was then explored to identify potential appropriate predictor varibales which could be used to predict iris species type. Histograms were first made to look at sample distribution, followed by scatterplots and boxplots:

- import seaborn as sns
- sns.displot()
- sns.catplot( kind = "box" )

##### 7. Creating a Linear Regression Model #####
A linear regression model was generated to predict petal width (y) from petal length (X):

- from sklearn.linear_model import LinearRegression
- model = LinearRegression(fit_intercept=True)
- import seaborn as sns 
- sns.relplot()
- model.fit(X, y)
- pred = pd.DataFrame()
- model.predict(pred)
- sns.replot()
- sns.lineplot()

#####  8. Testing the Model #####

- from sklearn.model_selection import train_test_split
- train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)
- iris_df.loc[train_X.index, "train/test"] = "train"
- iris_df.loc[test_X.index, "train/test"] = "test"
- sns.replot()
- from sklearn.linear_model import LinearRegression
- model = LinearRegression(fit_intercept=True)
- model.fit(train_X, train_y)
- model.score(test_X, test_y)

##### 9. Nearest Neighbours #####
A nearest neigbours model was finally made to predict iris species from input petal size (petal length and petal width):

- from pandas import DataFrame
- from sklearn.datasets import load_iris
- from sklearn.model_selection import train_test_split

- from sklearn.model_selection import GridSearchCV
- from sklearn.neighbors import KNeighborsClassifier
- parameters = {
    "n_neighbors" : range(1, 40),
}
- clf = GridSearchCV(KNeighborsClassifier(), parameters).fit(train_X, train_y)

- cv_results = DataFrame(clf.cv_results_)
- cv_results = cv_results.sort_values(["rank_test_score", "mean_test_score"])
- cv_results.head()[["param_n_neighbors", "mean_test_score", "std_test_score", "rank_test_score"]]
- cv_results.plot.scatter("param_n_neighbors", "mean_test_score", yerr="std_test_score")

- from sklearn.inspection import DecisionBoundaryDisplay
- import seaborn as sns
- DecisionBoundaryDisplay.from_estimator(clf, X, cmap="Pastel2")
- sns.scatterplot(data=X, x="petal length (cm)", y="petal width (cm)", hue=y, palette="Dark2")
- clf.score(test_X, test_y)

## Summary ##
Models were successfully made to firstly predict iris etal width from iris petal length, and secondly to predict iris species from petal length and width. High model scores indicated effective models in predicting the desired iris characteristics from the inputs.




