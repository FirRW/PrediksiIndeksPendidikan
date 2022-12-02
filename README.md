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
6. Deploy the project into Deployment using API Streamlit and platform HEROKU (optional)

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
- The features data is then trained on Lasso Linear Model, Random Forest Regression, Support Vector Regression, and Gradient Boosting models. with the help of cross validations for finding out the best parameters to be used, using the help of RandomizedSearchCV. Random Forest shows the best result

## **Conclusion**
- From the previous history data gathered from BPS and List of models used are Lasso Linear Model, Random Forest Regression, Support Vector Regression, and Gradient Boosting model. The best model of this is  Random Forest, with an RMSE Below 0.8, lowest compared to other models. As such, the prediction app would be made based on the saved random forest trained model.
- Based on this project, local government could improve the indeks pendidikan based on the information detailed on this project

![](https://i.ibb.co/Jd43XRV/Best-Feature.png)

![](https://i.ibb.co/RpwpVbX/Best-featuress.png)

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
