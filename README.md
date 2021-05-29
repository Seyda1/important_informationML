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




