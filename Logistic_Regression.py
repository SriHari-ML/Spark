from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import  LogisticRegression

spark = SparkSession.builder.appName('LR').getOrCreate()

input1 = spark.read.format('parquet').load('/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/binary-classification/')

print(input1.show(5))
print(input1.printSchema())

lr = LogisticRegression()
lrModel = lr.fit(input1)

print(lrModel.coefficients)
print(lrModel.intercept)

summary = lrModel.summary
print(summary.areaUnderROC)
print(summary.roc.show())
print(summary.pr.show())

objective_History = summary.objectiveHistory

print("objectiveHistory:") # Helps us to look at loss function value after every iteration
for objective in objective_History:
    print(objective)

# Incase the input data contains Multiclass Classification then use the below metrics

for i, f_val in enumerate(summary.fMeasureByLabel()):
    print("label", i, ": ", f_val)

fpr = summary.weightedFalsePositiveRate
tpr = summary.weightedTruePositiveRate
precision = summary.weightedPrecision
recall = summary.weightedRecall

print("FPR: ", fpr)
print("TPR: ", tpr)
print("Precision: ", precision)
print("Recall: ", recall)






