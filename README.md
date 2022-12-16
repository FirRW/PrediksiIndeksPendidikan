## **Backgrounds**
- The education index is one of the benchmarks in determining how well the quality of education is in an area. The value of the education index can also determine related stakeholders such as the regional education office to determine strategic steps to develop related education conditions, such as whether supporting facilities need to be provided, or the need for schools to be built, and so on.
- In this project proposal, a machine learning model will be built to predict the size of the education index based on many related factors and also determine which factors are most influential. The main objective is that users can find out the amount of the Education Index in each region and find out how well education is developing in that area. In terms of stakeholders, it is also to be able to find out more about what is actually the main factor in determining the size of the education index and needs to be addressed.

![](https://cdnwpedutorenews.gramedia.net/wp-content/uploads/2021/12/09103810/tujuan-pendidikan-nasional.jpg)


## **Objectives**
- Creating prediction system using a machine learning algorithm that uses data from previous "Indeks Pendidikan"
- The prediction system will input the many features related to "Pendidikan"
- The success would be if the trained model prediction reaches a lowest rmse score, compared to other models


## **Methods**
1. Collect the data from data source
2. Prepare the data
3. Exploratory Data Analysis & Feature Selection and Engineering for the dataset
4. Machine learning training using the dataset previously selected and modified, hyperparameters of model is adjusted on crossvalidations.
5. The best model is then picked and saved into our prediction app, check the most important feature
6. Deploy the project into Deployment using API Streamlit and platform HEROKU or similar (optional)

## **Overview**
- Most of the data features are numerical value, and some of them are aggregated or encoded
- One of example for feature engineering target can be shown below:
```python
#Auto encodes any dataframe column of type category or object.
def Encode(df,variable):
        le = LabelEncoder()
        try:
            df[variable] = le.fit_transform(df[variable])
        except:
            print('Error encoding '+variable)
        return df

#Aggregate sum
def aggsum(df,variable_year,variable_kabupaten,target):
   df= df.groupby([variable_year,variable_kabupaten])[target].agg('sum')
   df = df.to_frame()
   df= df.reset_index()
   return df
```
- Education Index in West Java (Jawa Barat) range around 50-70 based on the data

![](https://i.ibb.co/HxdRMty/EDA-2019-Indeks.png)

- The features data is then trained on many models, such as Linear Regression, Lasso, Ridge, Bayesian Ridge, Elastic Net, Huber Regression, Random Forest, Decision Tree, support vector, SGD, gradient boosting, neighbors, lgbm, Adaboost and XGboost. with the help of cross validations for finding out the best parameters to be used, using the help of RandomizedSearchCV. 

## **Conclusion**
- The best model of this is XGBOOST, with an RMSE 0.39 and R2 Score 0.74, lowest compared to other models. As such, the prediction app would be made based on the saved xgboost trained model.

![](https://i.ibb.co/Zg6Hzb9/Best-model.png)

- Other models used are Linear Regression, Lasso, Ridge, Bayesian Ridge, Elastic Net, Huber Regression, Random Forest, Decision Tree, support vector, SGD, gradient boosting, neighbors, lgbm, and Adaboost.
- The most influential feature is the number of learning center activities(kegiatan pusat belajar). but this feature has a negative impact, so that the more learning center activities(kegiatan pusat belajar), the lower the value of the education index
- Features that can be controlled by the government and have a positive impact based on this modeling are, by increasing the number of libraries (Jumlah Perpustakaan) & increasing illiteracy eradication activities (kegiatan pemberantasan buta aksara)

![](https://i.ibb.co/wyztKbf/Feature-importance-SHAP-Bar.png)

![](https://i.ibb.co/ZBr6wHf/Feature-importance-SHAP.png)

## **Best model Explanation**
- XGBoost, which stands for Extreme Gradient Boosting. When using gradient boosting for regression, the weak learners are regression trees, and each regression tree maps an input data point to one of its leafs that contains a continuous score. XGBoost minimizes a regularized objective function that combines a convex loss function (based on the difference between the predicted and target outputs) and a penalty term for model complexity. The training proceeds iteratively, adding new trees that predict the residuals or errors of prior trees that are then combined with previous trees to make the final prediction. 

## **Limitation**
- There is a missing data from source of a certain kabupaten/kota, it would be beneficial if the government would improve the dataset based on these missing value, especially on waktu tempuh sekolah
- Limited amount of data is used in our modelling (only in West Java), as Education index in each kabupaten/kota would be differ on each province, so to make this modelling better, a whole kabupaten/kota on all provinces in Indonesia would make more impact on our project

## **Developed By**
- Firdan Rahman W. : https://github.com/FirRW
- Zulfikar Aditya : https://github.com/zulfikarnj

# **Indeks Pendidikan Prediction **
Indeks Pendidikan using Scikit Model

Incoming Deployment @ https://predictindekspendidikan-ml2.herokuapp.com/ 

## **Reference**
Dataset: 
- https://opendata.jabarprov.go.id/
- https://jabar.bps.go.id/

Heroku:
- https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83
"# PrediksiIndeksPendidikan" 
