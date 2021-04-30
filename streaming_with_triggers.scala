import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object streamTriggers {
  def main(args: Array[String]): Unit = {

    // Create Spark Session
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Trigger")
      .getOrCreate()

    // Set Spark logging level to ERROR to avoid various other logs on console.
    spark.sparkContext.setLogLevel("ERROR")

    val schema = StructType(List(
      StructField("Date", StringType, true),
      StructField("Open", DoubleType, true),
      StructField("High", DoubleType, true),
      StructField("Low", DoubleType, true),
      StructField("Close", DoubleType, true),
      StructField("Adjusted Close", DoubleType, true),
      StructField("Volume", DoubleType, true)
    ))

    
    def getFileName : Column = {
      val file_name = reverse(split(input_file_name(), "/")).getItem(0)
      split(file_name, "_").getItem(0)
    }

    // Create Streaming DataFrame by reading data from socket.
    val initDF = (spark
      .readStream
      .option("maxFilesPerTrigger", 1) 
      .option("header", true)
      .schema(schema)
      .csv("data/stream")
      .withColumn("Name", getFileName)
      .withColumn("timestamp", current_timestamp())
      )

    // Aggregation on streaming DataFrame.
    val resultDF = initDF
      .groupBy(col("Name"), year(col("Date")).as("Year"))
      .agg(max("High").as("Max"),
        max("timestamp").as("timestamp"))
      .orderBy(col("timestamp").desc)

  
 
    resultDF
      .writeStream
      .outputMode("complete")
      .trigger(Trigger.ProcessingTime("1 minute")) //Trigger.Once(), Trigger.ProcessingTime("1 minute")
      .format("console")
      .option("truncate", false)
      .start()
      .awaitTermination()
  }
}
