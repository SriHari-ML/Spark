from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from time import  sleep
spark = SparkSession.builder.appName("streaming").getOrCreate()

static = spark.read.json("/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/activity-data/")
dataSchema = static.schema
print(dataSchema)

streaming = spark.readStream.schema(dataSchema).option("maxFilesPerTrigger", 1).json("/Users/srihari/Downloads/Spark-The-Definitive-Guide-master/data/activity-data/")
activityCount = streaming.groupBy("gt").count()
spark.conf.set("spark.sql.shuffle.partitions", "5")

activityQuery = streaming.writeStream.queryName("activityCount").format("memory").outputMode("complete").start()
activityQuery.awaitTermination()

for x in range(5):
    print(spark.sql("select * from activityCount").show())
    sleep(1)