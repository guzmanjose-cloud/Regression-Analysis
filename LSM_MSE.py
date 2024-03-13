'''
This code uses the least squares method to calculate the 
slope for the best fitting line for the data. The csv file consists of SAT and GPA scores.
what i am trying to find out is what somones GPA would be depending on the SAT score. I specifically 
wrote out the least squares method and mean squared error instead of using sklearn or pytorch in order 
to show the math that accompanys these 2 functions

'''
import matplotlib.pyplot as plt
import pandas as pd



file_path = "GPA.csv"

data = pd.read_csv(file_path)

x = data["SAT"]
y = data["GPA"]

points = len(x)

#This function uses the least squares method equation to get the coeffcient m and intercept b
def least_squares(x, y, points):
    

    sum_x = sum(x)
    sum_y = sum(y)
    xy = sum(xi * yi for xi, yi in zip(x, y))
    x_squared = sum(xi ** 2 for xi in x)

    m = (points * xy - sum_x * sum_y) / (points * x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / points
    return m, b

#This function calculates the mean squared error of the slope 
def mse(x, y, m, b):

    residuals = [(m * xi + b - yi) ** 2 for xi, yi in zip(x, y)]
    mse = sum(residuals) / len(x)
    
    return mse



#This function inputs a random x value and returns the predicted y value
def fit(x_values , m, b):
    

    y_hat = m * x_values + b 

    return x_values, y_hat

#prints outs the coefficient and intercept
m, b = least_squares(x,y,points)
print("coeficient m:", m)
print("intercept b:", b)

#prints out the mean squared error of the slope
mse = mse(x,y,m,b)
print("mean squared error:", mse)

#predicts an x value 
x_values, y_hat = fit(1700,m,b)
print("predicted x value:", x_values)
print("gpa for predicted x value is:", y_hat)


#prints out points, slope and predicted x value
plt.scatter(x,y, color="blue", label="x and y points")
plt.scatter(x_values,y_hat,color='green')
plt.plot(x, m * x + b, color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()



