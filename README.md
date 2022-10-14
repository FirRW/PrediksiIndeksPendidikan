# **IVF Prediction App**
IVF Prediction app using Scikit Model

Deployed @ https://predictivfgrupb1.herokuapp.com/


## **Backgrounds**
- It is crucial for parents to have a clear view on both father’s and mother’s condition to be able to have a pregnancy. If the cell from both parents is not strong enough to conceive naturally, one of the things that parents could do is by IVF or what we called In Vitro Fertilization. This is by combining the sperm and egg cells that would be taken from both parents in a laboratory, which then be planted again in the mother’s womb. Even then, there are many factors that would influence and lead to the successfulness of the IVF process. 
- As such, we aim to make a prediction system using previous results of IVF on other patients based on their profile and their history, while also including the condition of the IVF process, including the storage process, time reserves, etc. 
- This prediction system would conclude the successfulness from doing the IVF until birth. But this does not include the patient lifestyle during the pregnancy. From this prediction result, we would inform the patients of the outcome, whether it would be successful or not. Based on this information, the patients will now not blindly try the program without any prior knowledge and they would consider if they would risk the IVF process even if the program is expensive.  

![](https://www.medicalcenterturkey.com/wp-content/uploads/2019/03/How-much-is-IVF-cost.jpg)


## **Objectives**
- Creating prediction system using a machine learning algorithm that uses data from previous patients
- The prediction system will input the current patient’s profile and condition of the process. The output would result in whether the parents would have a successful birth or not. 
- The success would be if the trained model prediction reaches an ROC-AUC score of 75%, and is scored highest compared to other models.


## **Methods**
1. Collect the data from data source
2. Prepare the data
3. Exploratory Data Analysis & Feature Selection and Engineering for the dataset, To optimize the features to be used on the model and to avoid an overfitting if it is further proceed with modeling
4. Machine learning training using the dataset previously selected and modified, hyperparameters of model is adjusted on crossvalidations.
5. Evaluation from the performance of each model using validation data
6. The best model is then picked and saved into our prediction app
7. Deploy the project into Deployment using API Streamlit and platform HEROKU

## **Overview**
- Most of the data features are categorized by binning and converting the text variable into categorical data.
- One of example for feature engineering target can be shown below:
```python
def feature_engineering_target (df_selected):

    #For target Value
    df_selected_target = df_selected[['Total number of live births - conceived through IVF or DI','Total number of live births - conceived through IVF','Total number of live births - conceived through DI','Number of Live Births']].copy()
    df_selected_target= pd.concat([df_selected_target,df_selected['Live Birth Occurrence']], axis=1)
    #TREAT Total number of live births - conceived through IVF or DI
    #Transform >=5 to certain number
    df_selected_target['Total number of live births - conceived through IVF or DI'].replace('>=5', 6, inplace=True)
    #Transform data into integer
    df_selected_target['Total number of live births - conceived through IVF or DI']= df_selected_target['Total number of live births - conceived through IVF or DI'].astype(str).astype(float)
    #Transform dataframe to 0 and 1, with 1 is stated as successful live birth, while 0 is no recorded livebirth
    df_selected_target['Total number of live births - conceived through IVF or DI'] = np.where(df_selected_target['Total number of live births - conceived through IVF or DI']==0, 0, 1)
    #No need to convert to int as it is already numeric
    df_selected_target['Total number of live births - conceived through IVF'] = np.where(df_selected_target['Total number of live births - conceived through IVF']==0, 0, 1)
    df_selected_target['Total number of live births - conceived through DI'] = np.where(df_selected_target['Total number of live births - conceived through DI']==0, 0, 1)
    df_selected_target['Number of Live Births'] = np.where(df_selected_target['Number of Live Births']==0, 0, 1)
    #COMBINE ALL COLUMNS, if there is any number 1 in a row, then it will be added as successful birth
    # create a list of our conditions
    conditions = [
    (df_selected_target['Total number of live births - conceived through IVF'] == 1) | 
    (df_selected_target['Total number of live births - conceived through DI'] == 1) | 
    (df_selected_target['Total number of live births - conceived through DI'] == 1) |
    (df_selected_target['Number of Live Births'] == 1) |
    (df_selected_target['Live Birth Occurrence'] == 1)
    ]
    # create a list of the values we want to assign for each condition
    values = [1]
    # create a new column and use np.select to assign values to it using our lists as arguments
    df_selected_target['success or not'] = np.select(conditions, values, default=0)
    #Take only the success or not column
    df_selected_target.drop(['Total number of live births - conceived through IVF or DI', 'Total number of live births - conceived through IVF',
                         'Total number of live births - conceived through DI','Number of Live Births', 'Live Birth Occurrence'], axis=1, inplace= True)


    return df_selected_target
```
- The features data is then trained on logistic Regression, K-nearest neighbor, Decision Tree, Random Forest Tree, Support Vector machine, and XGBoost Classifier models. with the help of cross validations for finding out the best parameters to be used, using the help of GridSearchCV and RandomizedSearchCV. XGBoost Classifier shows the best result

## **Conclusion**
- From the previous history data gathered from Human Fertilization and Embryology Authority (HFEA) and List of models used are Logistic Regression, K-nearest neighbor, Decision Tree, Random Forest Tree, Support Vector machine, and XGBoost Classifier. The best model of this is XGBoost Classifier, with an ROC-AUC Score above 0.8, highest compared to other models as it optimizes using the gradient descent and boosting algorithm. As such, the prediction app is made based on the saved XGBoost Classifier trained model.


## **Developed By**
- Firdan Rahman W. : https://github.com/FearDawn
- Galih M
- Ridha 


## **Reference**
Dataset: 
- https://www.hfea.gov.uk/about-us/our-data/

Journal:
- Scientific Report: “Machine learning predicts live‑birth occurrence before in‑vitro fertilization treatment”, “AshishGoyal, Maheshwar Kuchana & Kameswari Prasada RaoAyyagari”, https://www.nature.com/scientificreports
- “Multifactor Prediction of Embryo Transfer Outcomes Based on a Machine Learning Algorithm”, “Ran Liu, Shun Bai, Xiaohua Jiang, Lihua Luo, Xianhong Tong, Shengxia Zheng, Ying Wang and Bo Xu”.https://www.frontiersin.org/articles/745039

Heroku:
- https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83
