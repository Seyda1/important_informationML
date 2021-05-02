# important_informationML
![Image of ii](http://educationprofessional.info/wp-content/uploads/2020/04/important-update.jpg)

## Correlation | Correlation Matrix

In the simplest terms, correlation explains to how one or more variables are related to each other. 
If there is a high correlation, it means that these two features are really related and values affect each other highly.
If two columns show high correlation with each other, we need to remove one of them.
---
 "` corr=final_dataset.corr()
top_corr_features=corr.index
plt.figure(figsize=(10,10))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#corr uses pearson correlation "`




