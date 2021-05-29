# important_informationML
![Image of ii](http://educationprofessional.info/wp-content/uploads/2020/04/important-update.jpg)

## Outlier Detection

### With Quantile
![Image of ii](https://naysan.ca/wp-content/uploads/2020/06/box_plot_ref_needed.png)

- Quantile fonksiyonu ile bir feature'ın sahip olabileceği min ve max threshold belirlenir.
- Bu değer aralıklarında olmayanlar outlier kabul edilir ve bunlar verisetinden kaldırılırı. 
- Böylelikle veri daha düzenli ve temiz hale gelir.
---
```
max_threshold=heart_data["istenilenfeature"].quantile(0.99)
max_threshold
```
```
heart_data[heart_data["istenilenfeature"]>max_threshold]
```
```
min_threshold=heart_data["istenilenfeature"].quantile(0.01)
min_threshold
```
```
df=heart_data[(heart_data["istenilenfeature"]<max_threshold) & (heart_data["istenilenfeature"]>min_threshold)]
df
```
---
## Scaling

- Decision tree için scaling yapmak şart değildir.
- Knn, k-means clustering, linear regression gibi algoritmalar scaling gerektirir.
- Scaling ile tüm değerler belli aralıklarda tutulur. Böylelikle herhangi bir değerin başka bir değeri domine etmesinin önüne geçilmiş olur.
```
from sklearn.preprocessing import StandardScaler
X_scale=StandardScaler().fit_transform(X)
```
---
## Feature Selection
```
import matplotlib.pyplot as plt
feature_imp=pd.Series(model.feature_importances_,index=X.columns)
feature_imp.plot(kind='barh')

best_features=feature_imp.nlargest(8).index
best_features
```
- Feature selection, modelin eğitim süresini kısaltma avantajı sağlar.
---
## Correlation | Correlation Matrix

- In the simplest terms, correlation explains to how one or more variables are related to each other. 
- If there is a high correlation, it means that these two features are really related and values affect each other highly.
- If two columns show high correlation with each other, we need to remove one of them.
---
 ```
 corr=final_dataset.corr()
top_corr_features=corr.index
plt.figure(figsize=(10,10))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#corr uses pearson correlation
```
---
# Modelling
## KNN ALgorithm
```
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(Xrd_train,yrd_train)
preds=knn.predict(Xrd_test)
```
- We should find a k which have best value for knn.
```
#best k for Knn

train_score=[]
test_score=[]

for i in range(1,15):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xrd_train,yrd_train)
    
    train_score.append(knn.score(Xrd_train,yrd_train))
    test_score.append(knn.score(Xrd_test,yrd_test))
```
```
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_score,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_score,marker='o',label='Test Score')
```
---
## Decision Trees
- One of the many qualities of decision trees is that they require very little data preparation.
- In fact, they don't require feature scaling or centering at all.
- Decision trees are intuitive, and their decisions are easy to interpret.
- Such model are often called "white box models"
- A node's gini attribute measures its impurity: a node is "pure"(gini=0) if all training instances it applies to belong to the same class.
- Increasing min_* hyperparameters or reducing max_* hyperparameters will regularize the model.
- ************************************************************************************************
- Veriseti en az kuralla bölünmeye çalışılır.
- Verisetinin ayrımını en iyi yapan feature en tepeye yani kök node'a yerleştirilir.
- Amaç veriyi en iyi modelleyen en küçük ağaca ulaşmaktır.
### Entropy
- The concept of entropy originated in thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered.
- Using gini impurity or entropy, most of the time it does not make a big difference:they lead to similar trees. Gini impurity is slightly faster to compute.
- Saflık çok yüksek olursa, entropy çok düşük olur.
- Örneğin hilesiz bir madeni parada yazı veya tura gelme olasılıkları %50 ise yani eşit ise; entropy en yüksek değerinde olur.
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
```
```
parameter={'max_leaf_nodes': range(2, 10), 'max_depth': range(1,7), 'min_samples_split' : [2,3,4]}
grid_searchDT = GridSearchCV(dt,parameter,cv=5)
grid_searchDT.fit(Xrd_train,yrd_train) 

grid_searchDT.best_params_
grid_searchDT.best_estimator_
grid_searchDT.best_score_

```
### Visualization Tree
```
tree = DecisionTreeClassifier(max_depth=4, random_state=42,max_leaf_nodes=8,min_samples_split= 2)
tree.fit(Xrd_train, yrd_train)

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["0", "1"],
                feature_names=None, impurity=False, filled=True)
                
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))
```
---
## Random Forests





