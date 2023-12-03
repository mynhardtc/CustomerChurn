# CustomerChurn

This repo is created for a specific project to analyse banking customer churn
The source of the data is from:
https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

The format of the analyses is as follows:
1. Get data from local file
2. Perform exploratory data analysis
2.1. Check the contents of the file
2.2. Create a summary of the data to understand the range and basic statistics of the data
3. Create plots of the data to get an understanding of trends and correlations
4. Identify features that are good candidates of the classification model
5. Setup the data from the classification
6. Run the classification algorithm to obtain the classification report

The outcome:
A Linear SVC model is used for the prediction. 
The dataset was small enough to use this and a binary outcome was required
Three feature lists are created to assess which set of features are optimal for the prediction
Based on the features, the most important features were the number of products and complaints from the customer
