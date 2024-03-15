from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#assigns csv to variable
file_path = "GPA.csv" 

#uses pandas read csv function to open csv file
df = pd.read_csv(file_path) 

# Reshape x to a 2D array
x = df["SAT"].values.reshape(-1, 1)
y = df["GPA"]

#splits data into training and testing variables
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25, shuffle=True)

#assigns linear regression from sklearn to variable called model
model = LinearRegression() 

#runs the training data to find best fitting line for data
fit_model = model.fit(x_train,y_train) 

#uses the best fit line to predict GPA scores
x_value = model.predict(x_test) 

#predicts a single x value when given a single y value
single_predicted_X_value = np.array([[1700]])
new_x = model.predict(single_predicted_X_value)
print("new GPA score:", new_x)

#prints coeficient, intercept and mean squared error
coef_ = model.coef_
intercept = model.intercept_
mse = mean_squared_error(y_test, x_value)

print("mean squarred error:", mse)
print("coeficient:", coef_)
print("intercept:", intercept)

#graphs points and best fit line of the data
plt.scatter(x,y,color="red",label="original data")
plt.scatter(x_test,y_test, color="blue",label="Test data")
plt.scatter(single_predicted_X_value,new_x,color="green",label="predicted GPA for 1700 SAT score")
plt.plot(x_test,x_value, color="black", label="Regression Line")
plt.xlabel("SAT SCORE")
plt.ylabel("GPA SCORE")
plt.legend(loc="upper left")
plt.show()
