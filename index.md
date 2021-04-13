## Introduction

EDA, Exploratory Data Analysis, serves to analyze and investigate data sets for their main characteristics, the conventional process to explore data includes basic descriptive analysis, missing values analysis & imputation, outliers identity and removal, clustering, association findings, data skewness(labeling) and time-based clustering and analytics and so on. 

In this article, with the tutorial using Python with NumPy, Pandas, and Matplotlib, we will represent the EDA in an Azure Machine Learning compute instance which should be connected to the Python 3.6 - AzureML kernel. Actually there are a wide range of tools and languages that we can use to perform it, we will use Jupyter notebooks and Python in Azure.

## 1. Exploring data with NumPy

NumPy is a Python library for working with arrays, it contains functions for linear algebra, matrices and so on. So that it can be utilized to perform a number of mathematical operations on arrays.

To begin with, suppose that a college takes a sample of student grades for a data science class, there is a sample:

```python
#a sample of student grades for a data science class
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)
```
The data is a list structure, in Python it is not good for numeric analysis, so NumPy could help specify it to a NumPy array:
```python
#NumPy supports specific data types and functions for working with Numbers in Python
import numpy as np
grades = np.array(data)
print(grades)
```
What the differences between a list and a NumPy array is? try compute them by 2 in an expression:
```python
print (type(data), 'x2:', data*2)
print (type(grades),'x2:',grades*2)
```
The result:

![image](https://user-images.githubusercontent.com/71245576/114579929-5c01c700-9c4c-11eb-8182-88292a7bdf35.png)

Mutipliying a list by 2 creates a new list of twice the length with the original sequence, a NumPy array on the other hand represents an element-wise calculation, it ends up with an array of the same size in which each element has been multiplied by 2.

Something deserves noticing is that in 'numpy.ndarray', the nd indicates the structure have n dimensions. We can see how many elements in single dimension and how many dimensions this structure have as well as locating a element.

```python
grades.shape
grades[0]
```
It contains 1 diemnsion of 22 elements, the first element is 50. Let's find the simple average grade.
```python
grades.mean()
```
The mean of grades  is 49.182, now we are gonna add a set of time data recording the typical number of hours per week they went to studying, it will match with the grades it ends up gaining.

```python
#define study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
# combine study hours and grades
student_data = np.array([study_hours, grades])

# show the array
student_data
```
The array showed: it is a two dimensional array.

![image](https://user-images.githubusercontent.com/71245576/114583275-6ffaf800-9c4f-11eb-9c8d-64fa4d0396fa.png)

Let's see its shape and locate one of elements.
```python
student_data.shape
student_data[0][1]
```
Two dimensions and each of dimensions has 22 elements. The first row second column element is 11.5.

Let's get the mean value of each dimension
```python
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()
print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))
```
The average study hours is 10.52 and average grade is 49.18.

![image](https://user-images.githubusercontent.com/71245576/114584085-4a222300-9c50-11eb-90c6-405ee3c48ef1.png)

## 2. Exploring data with Pandas

We know that NumPy provides a lot of the functionalities to work with arrays of numeric values, Pandas package, in another way, offers structures to work with Dataframes.

To begin with we import the module and initialize the data set df_students.
```python

import pandas as pd

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

df_students 
```
The result:

![image](https://user-images.githubusercontent.com/71245576/114584690-e8ae8400-9c50-11eb-8311-573a1e2474b7.png)

Dataframe has the loc method to locate data for a specific index value, like
```python
df_students.loc[5] #1
df_students.loc[0:5] #2
df_students.iloc[0:5] #3
df_students.iloc[0,[1,2]] #4
df_students.loc[0,'Grade'] #5
df_students.loc[df_students['Name']=='Aisha'] #6
df_students[df_students['Name']=='Aisha'] #7
df_students.query('Name=="Aisha"') #8
df_students[df_students.Name == 'Aisha'] #9
```
The first line shows that the 6th(index value is 5) row, the name is Vicky, StudyHours is 1 and Grade is 3

The second line shows the rows from index 0 to index 5(the 1st to 6th rows)

The third shows the first 5 rows, the "0" means it scans from the first row, the "5" shows that there should be 5 rows to represent

The forth shows from the index 0(first row), and retrieve values of the index 1 and index 2 columns

The fifth locates the grade of the first row

The sixth to nineth both finds the datum from which the name is Aisha.

### Loading a dataframe from a file

In Pandas, we can read data from a existed file, for example, upload data from grades.csv in a directory.

```python
df_students = pd.read_csv('data/grades.csv',delimiter=',',header='infer')
df_students.head()
```
The result:

![image](https://user-images.githubusercontent.com/71245576/114608823-194fe700-9c6c-11eb-819d-cf796d20b18c.png)

In this case the delimiter is a comma，set the header='infer' means that it can be able to detect the header names

### Handing missing values

Missing value analysis is really common in analytics, there are several methods to detect missing values and manipulate on them, first we need to know if there are missing values, using isnull method.

```python
df_students.isnull()
df_students.isnull().sum()
```
isnull() shows the missing value infomation from every row, we can sum up the number of missing values by sum function, the result:

![image](https://user-images.githubusercontent.com/71245576/114609773-26210a80-9c6d-11eb-929d-5b850cf18ec7.png)

For column StudeyHours and Grade, they have 1 and 2 missing values for each, one thing should be noticed is that when the dataframe is retrieved, the missing numeric values show up as NaN(not a number).

Three approaches to handle with them: drop, impute or keep it, it depends on scenarios of business, statistical assumption like their features of randomization. Now we are trying to impute the mean of StudyHours using fillna method for the missing values in StudyHours.

```python
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
df_students
```
As well, using dropna method to drop rows in which any of the columns contain null values:

```python
df_students = df_students.dropna(axis=0, how='any')
df_students
```

After cleaned the missing values, let's compare the mean study hours and grades.

```python
# Get the mean study hours using to column name as an index
mean_study = df_students['StudyHours'].mean()

# Get the mean grade using the column name as a property (just to make the point!)
mean_grade = df_students.Grade.mean()

# Print the mean study hours and mean grade
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))
```
![image](https://user-images.githubusercontent.com/71245576/114612019-992b8080-9c6f-11eb-9acb-3ece18be3114.png)

To find the students who studied for more than the mean amount of time.

```python
# Get students who studied for the mean or more hours
df_students[df_students.StudyHours > mean_study]
```

![image](https://user-images.githubusercontent.com/71245576/114612150-c11ae400-9c6f-11eb-8fa0-4907a28cca75.png)

You also can compute the mean grades of the students whose grades are more than the mean grades of all students, here I pass it and we could say the mean grade of these students is 66.7.

### Pass grade

If we set the pass grade is 60 in this course, we can create a Pandas Series containing the pass/fail indicator and concatenate the series in the dataframe.

```python
passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

df_students

the axis=1 means that we add a new column in the dataframe, the result Top 10 rows:

![image](https://user-images.githubusercontent.com/71245576/114613001-cb89ad80-9c70-11eb-8cf7-55c35a578680.png)

Dataframes are designed for tabular data so that we can perform many operations as same as we do in a RDBMS, such as grouping and aggregating tables of data.

```python
print(df_students.groupby(df_students.Pass).Name.count()) #1
print(df_students.groupby(df_students.Pass)['StudyHours', 'Grade'].mean()) #2
```
The first statement shows that 15 of all students are failed to pass, others pass(7 students). 
The second statement shows the study hours and grades in average, for passed students and failed students.

![image](https://user-images.githubusercontent.com/71245576/114613728-b6f9e500-9c71-11eb-9b88-75ae67c18f28.png)

Dataframes are amazingly versatile, we can create a dataframe by sorting by grade.
```python
# Create a DataFrame with the data sorted by Grade (descending)
df_students = df_students.sort_values('Grade', ascending=False)
# Show the DataFrame
df_students
```

The result of top 10 rows:

![image](https://user-images.githubusercontent.com/71245576/114614132-253ea780-9c72-11eb-84cf-a2133499c4a3.png)

## 3. Visualization with Matplotlib

Matplotlib library provides the functionality to data visualizations, let's start with a simple bar chart that shows the grade of each students.

```python
# Ensure plots are displayed inline in the notebook
%matplotlib inline

from matplotlib import pyplot as plt

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade)

# Display the plot
plt.show()
```

The result:

![image](https://user-images.githubusercontent.com/71245576/114614480-8e261f80-9c72-11eb-8a72-ba0b655f9b04.png)

The x axis looks a little bit tricky in this plot, actually we can rotate the student name. 

### Pyplot function

The pyplot class from Matplotlic provides a whole bunch of ways to improve the visual elements of the plot: the color, title, labels, grid and rotaters of markers.

```python
# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Display the plot
plt.show()
```
![image](https://user-images.githubusercontent.com/71245576/114622172-9f275e80-9c7b-11eb-8a52-d4d0abb353e9.png)

This plot shows a bar plot of name versus grade, we customized the title, xlabel, ylabel, and plotted a grid and rotate the xticks. 

Now we discuss a conception, figure. We should know that a plot is technically contained with a Figure. In the previous examples, the figure was created implicitly for you; but you can create it explicitly. For example, the following code creates a figure with a specific size.

```python
# Create a Figure
fig = plt.figure(figsize=(8,3))

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Show the figure
plt.show()
```

![image](https://user-images.githubusercontent.com/71245576/114622459-fa595100-9c7b-11eb-91a9-f1d61ef93086.png)

A figure can contain multiple subplots, let's start to create a figure with two subplots, the one is a bar chart and the another is a pie chart.

```python
# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()
```

![image](https://user-images.githubusercontent.com/71245576/114622644-40aeb000-9c7c-11eb-8a82-d8008432c021.png)

DataFrame also provides its own methods for plotting data, for example:

```python
df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))
```

The result is:

![image](https://user-images.githubusercontent.com/71245576/114623054-c0d51580-9c7c-11eb-958d-0d37bda9c6c2.png)



