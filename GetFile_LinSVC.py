'''
This function file is used for:
1. Open File from base directory
2. Save contents of file to a dataframe
'''

import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def customer_churn_predict(file):
    # set display width to be detected automatically
    pd.options.display.width = 0

    file = file
    # def get_file_content(file):

    df = pd.read_csv(file)

    print('\nMain Data Information\n')
    print(df.info(verbose=1))

    print('\nData Description and Summary\n')
    print(df.describe(include='all'))
    cols = df.columns.values.tolist()

    # Display information about missing values
    print("Missing Values Before Preprocessing:")
    print(df.isnull().sum())

    # no nulls found in dataset

    # show top 100 values in the dataset
    print('\nShow the top 100 rows of data\n')
    print(df.head(n=100))

    # Visualize data in bar graph
    # Visualize the data using a pair plot for numerical data
    print('\nStarting pair plot')
    sns.pairplot(df, hue='Exited')
    plt.title('Pair Plot')
    plt.savefig('PairPlot.png')

    # Candidate Feautures
    # 1 - Full - ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned']
    # 2 - Feature Select - ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited', 'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned']


    # generate plot to check categorical columns correlation
    df_cat = df[['Geography', 'Gender', 'Card Type', 'Exited']]

    geo_exit = df_cat.groupby(['Geography', 'Exited'])['Exited'].count().to_frame(name='count').reset_index()
    gen_exit = df_cat.groupby(['Gender', 'Exited'])['Exited'].count().to_frame(name='count').reset_index()
    card_exit = df_cat.groupby(['Card Type', 'Exited'])['Exited'].count().to_frame(name='count').reset_index()


    fig, ax = plt.subplots(3)

    ax[0].scatter(geo_exit['Geography'].to_list(), geo_exit['count'].to_list(), c=geo_exit['Exited'].to_list())
    ax[0].legend()
    ax[1].scatter(gen_exit['Gender'].to_list(), gen_exit['count'].to_list(), c=gen_exit['Exited'].to_list())
    ax[1].legend()
    ax[2].scatter(card_exit['Card Type'].to_list(), card_exit['count'].to_list(), c=card_exit['Exited'].to_list())
    ax[2].legend()

    plt.savefig('CategoryPlot.png')

    # Geography has an impact on whether a customer exits or not
    # This might not be a determining factor and we would like a more generic prediction
    # other categorical values have low impact


    # Final Features to use
    # 2 - Feature Select - ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited', 'Complain', 'Satisfaction Score', 'Point Earned']
    # features = ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Complain', 'Satisfaction Score', 'Point Earned']

    features_1 = ['NumOfProducts', 'Complain']
    features_2 = ['CreditScore', 'Age', 'HasCrCard', 'IsActiveMember', 'Balance', 'EstimatedSalary']
    features_3 = ['Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'CreditScore']
    exited = ['Exited']
    features = [features_1, features_2, features_3]

    for f in features:

        X = df[f]
        y = df[exited]

        # Split the data into training and testing sets
        print('\nSplit data to test and train datasets')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features (important for SVC)
        print('\nStandardise dataset')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a Linear Support Vector Classifier
        linear_svc_model = SVC(kernel='linear')

        # Train the model
        print('\nTrain model')
        linear_svc_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        print('\nMake Predictions')
        y_pred = linear_svc_model.predict(X_test_scaled)

        # Evaluate the model
        print('\nEvaluate the model')
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        print(f"Model Evaluation: for {f}")
        print(f"Accuracy: {accuracy}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_rep)


