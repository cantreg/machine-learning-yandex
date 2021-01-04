import pandas
import numpy

def writeAnswer(num, ans):
    f = open('output'+num+'.txt', 'w')
    f.write(ans)

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

#
countSex = data.Sex.value_counts()
writeAnswer('1', '{} {}'.format(countSex['male'], str(countSex['female'])))

##
countSurvive = data.Survived.value_counts(normalize=True)
writeAnswer('2', '{:.2f}'.format(countSurvive[1]*100))

###
classCount = data.Pclass.value_counts(normalize=True)
writeAnswer('3', '{:.2f}'.format(classCount[1]*100))

####
medianAge = data.Age.median()
meanAge = data.Age.mean()
writeAnswer('4', '{:.2f} {}'.format(meanAge, medianAge))

#####
corr = numpy.corrcoef(data.SibSp, data.Parch)[0, 1]
writeAnswer('5', '{:.2f}'.format(corr))

######
names1 = data.Name.str.split('Mrs. ', expand = True)[:][1].str.extract(r'([\w\s]*)\(([\w]+)([\w\s]+)\)([\w\s]*)', expand=True)[1].dropna()
names2 = data.Name.str.split('Miss. ', expand = True)[:][1].str.extract(r'([\w]+)([\w\s]+)', expand=True)[0].dropna()
topName = pandas.concat([names1, names2]).value_counts().sort_values(ascending=False).index[0]
writeAnswer('6', topName)