from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('DT').getOrCreate()

data = spark.read.format('parquet').load('/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/multiclass-classification/')


print(data.show(10))

(training_Data, testing_Data) = data.randomSplit([0.7, 0.3])

dt = DecisionTreeClassifier(labelCol= 'label', featuresCol= 'features')

dtModel = dt.fit(training_Data)


prediction = dtModel.transform(testing_Data)

print(prediction.show())

evaluator = MulticlassClassificationEvaluator(labelCol= 'label', predictionCol= 'prediction', metricName= 'accuracy')
accuracy = evaluator.evaluate(prediction)

print('Accuracy : ', accuracy)
