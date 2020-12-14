import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, from_json}
import org.apache.spark.sql.types.{DataTypes, StringType, StructType}

object Task3 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("SparkHomework3")
      .master("local[*]")
      .getOrCreate()

    val stream = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 8080)
      .load()

    val structType = new StructType()
      .add("id", StringType, nullable = true)
      .add("text", StringType, nullable = true)

    val inputData = stream
      .withColumn("json", from_json(col("value"), structType))
      .select(col("json.id"), col("json.text"))

    val pipelineModel = PipelineModel.read.load("./myModel2/")
    val result = pipelineModel
      .transform(inputData)
      .select(col("id"), col("predictionTarget").as("target").cast(DataTypes.IntegerType))

    result
      .repartition(1)
      .writeStream
      .outputMode("append")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("path", "results/")
      .option("checkpointLocation", "checkpointLocation/")
      .start()
      .awaitTermination()
  }
}
