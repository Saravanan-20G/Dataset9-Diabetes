import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

df=pd.read_csv("C:\\Users\\Saravanan\\OneDrive\\Desktop\\Datasets_data\\diabetes_prediction_dataset.csv")

encode=OrdinalEncoder()
df.gender=encode.fit_transform(df[["gender"]])
df.smoking_history=encode.fit_transform(df[["smoking_history"]])

x=df.drop("diabetes",axis=1)
y=df["diabetes"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

model=DecisionTreeClassifier().fit(x_train,y_train)

y_pred=model.predict(x_test)

gender = {'Female':0, 'Male':1, 'Other':2}
smoking_history = {'never':4, 'No Info':0, 'current':1, 'former':3, 'ever':2, 'not current':5}
yes_no = {"No": 0, "Yes": 1}

row1 = st.columns(1)
row2 = st.columns(2)

def main():
    with row1[0]:
        st.title("Diabetes Prediction")
        st.write("Welcome to diabetes prediction application!")
        st.header("Fill the below information for analysis")
    with row2[0]:
        gender_selected = st.selectbox("Chouse you gender", {'Female':0, 'Male':1, 'Other':2})
        age = st.number_input("Enter your age", min_value=0, max_value=150, value=30)
        hypertension = st.selectbox("Do you have hypertension", {"No": 0, "Yes": 1})
        heart_disease = st.selectbox("Do you have heart disease?", {"No": 0, "Yes": 1})
        
    with row2[1]:
        smoking_history_selected = st.selectbox("Chouse you smoking history", {'never':4, 'current':1,
                                                                        'former':3, 'ever':2, 'not current':5})
        bmi = st.number_input("Enter your BMI", min_value=0.0, max_value=1000.0, value=25.0)
        HbA1c_level = st.number_input("Enter your HbA1c level", min_value=0.0, max_value=100.0, value=5.0)
        blood_glucose_level = st.number_input("Enter your blood glucose level", min_value=0.0, max_value=1000.0, value=100.0)


    diabetes_input = np.array([[gender[gender_selected], age, yes_no[hypertension], yes_no[heart_disease], 
                          smoking_history[smoking_history_selected], bmi, HbA1c_level, blood_glucose_level]])
    check_diabetes = st.button("Check diabetes")
    if check_diabetes:
        prediction = model.predict(diabetes_input)
        if prediction[0] == 1:
            st.error("You have diabetes")
        else:
            st.success("You don't have diabetes")

if __name__ == "__main__":
    main()