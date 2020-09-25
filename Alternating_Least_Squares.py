from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('ALS').getOrCreate()

ratings = spark.read.text('/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/sample_movielens_ratings.txt')
print(ratings.show(5))

print(ratings.count())

data = ratings.selectExpr("split(value, '::') as col").selectExpr("cast(col[0] as int) as userID", "cast(col[1] as int) as movieID", "cast(col[2] as float) as rating",  "cast(col[3] as long) as timestamp")
print(data.show(5))

(training, test) = data.randomSplit([0.7, 0.3])

als = ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userID").setItemCol("movieID").setRatingCol("rating")
alsModel = als.fit(training)
predictions = alsModel.transform(test)

alsModel.recommendForAllUsers(10).selectExpr("userId", "explode(recommendations)").show()

evaluator = RegressionEvaluator().setMetricName('rmse').setLabelCol('rating').setPredictionCol('prediction')
rmse = evaluator.evaluate(predictions)
print('RMSE : ', rmse)
