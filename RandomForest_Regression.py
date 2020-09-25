from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('RF').getOrCreate()

data = spark.read.format('parquet').load('/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/regression/')

print(data.show(25))
print(data.count())

(train_Data, test_Data) = data.randomSplit([0.7, 0.3])

rf = RandomForestRegressor()

rfModel = rf.fit(train_Data)
prediction = rfModel.transform(test_Data)
print(prediction.show())
print(prediction.count())

evaluator = RegressionEvaluator(metricName='rmse')
rmse = evaluator.evaluate(prediction)
print('RMSE: ', rmse)