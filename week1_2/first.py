import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
countSex = data.groupby('Sex').count()

f1 = open("output1.txt", "w")
f1.write(countSex)