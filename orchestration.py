import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from prefect import task,flow



@task
def data_ingestion(file_path):
    return pd.read_csv(file_path)

@task
def splitting_input_output(data,inputs,outputs):
    x=data[inputs]
    y=data[outputs]
    return x,y

@task
def splitting_train_test_data(x,y,test_size=0.25,random_state=43):
    return train_test_split(x,y,test_size=test_size,random_state=random_state)

@task
def data_preprocessing(x_train,x_test,y_train,y_test):
    tfidf=TfidfVectorizer(max_features=500)
    x_train=tfidf.fit_transform(x_train)
    x_test=tfidf.transform(x_test)
    return x_train,x_test,y_train,y_test

@task
def model_training(x_train,y_train,hyperparameters):
    clf=RandomForestClassifier(**hyperparameters)
    clf.fit(x_train,y_train)
    return clf

@task
def model_evaluation(model,x_train,y_train,x_test,y_test):
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)
    train_score=accuracy_score(y_train,y_train_pred)
    test_score=accuracy_score(y_test,y_test_pred)
    return train_score,test_score

@flow(name="sentiment analysis")
def workflow():
    
    path="final_data.csv"
    inputs="review"
    outputs="sentiment"
    hyperparameters={"n_estimators":100}
    
    df=data_ingestion(path)
    
    x,y=splitting_input_output(df,inputs,outputs)
  
    x_train,x_test,y_train,y_test=splitting_train_test_data(x,y)
    
    x_train_scaled,x_test_scaled,y_train,y_test=data_preprocessing(x_train,x_test,y_train,y_test)
    
    model=model_training(x_train_scaled,y_train,hyperparameters)
    
    train_score,test_score=model_evaluation(model,x_train_scaled,y_train,x_test_scaled,y_test)
    print("train score" ,train_score)
    print("test score:", test_score)

if __name__ == "__main__":
    workflow.serve(
        name="sentiment_analysis_project",
        cron="* * * * *"
    )
    


