import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, corr, desc, lit, row_number, stddev, udf, variance}

object Main {
    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("SparkHomework1")
            .master("local")
            .getOrCreate()

        val sparkContext = spark.sparkContext
        sparkContext.setLogLevel("ERROR")

        val dataFrame = spark.read
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .option("escape", "\"")
            .csv("src/main/resources/AB_NYC_2019.csv")

        val castedDataFrame = dataFrame
            .withColumn("price", dataFrame("price").cast("Long"))
            .withColumn("latitude", dataFrame("latitude").cast("Double"))
            .withColumn("longitude", dataFrame("longitude").cast("Double"))

        println("Mean")
        castedDataFrame
            .groupBy("room_type")
            .mean("price")
            .show()

        println("Median")
        castedDataFrame.createOrReplaceTempView("dataFrame")
        spark.sql(
            """
              |SELECT room_type, percentile_approx(price, 0.5) as `median price`
              |FROM dataFrame
              |GROUP BY room_type
            """.stripMargin
        ).show()

        println("Mode")
        dataFrame
            .groupBy("room_type", "price")
            .count()
            .withColumn(
                "row_number",
                row_number()
                    .over(
                        Window.partitionBy("room_type")
                            .orderBy(desc("count"))
                    )
            )
            .select("room_type", "price", "count")
            .where(col("row_number") === 1)
            .show()

        println("Variance")
        dataFrame
            .groupBy("room_type")
            .agg(variance(dataFrame("price")))
            .show()

        println("Standard deviation")
        dataFrame
            .groupBy("room_type")
            .agg(stddev(dataFrame("price")))
            .show()

        println("Offer with maximal price")
        castedDataFrame
            .orderBy("price")
            .limit(1)
            .select("id", "name", "price")
            .show()

        println("Offer with minimal price")
        castedDataFrame
            .orderBy(castedDataFrame("price").desc)
            .limit(1)
            .select("id", "name", "price")
            .show()

        println("Correlation between price and minimum_nights")
        dataFrame
            .select(
                corr("price", "minimum_nights")
            )
            .show()


        println("Correlation between price and number_of_reviews")
        dataFrame
            .select(
                corr("price", "number_of_reviews")
            )
            .show()

        val udfEncode = udf(
            (lat: Double, lng: Double, precision: Int) => GeoHash.encode(lat, lng, precision)
        )
        val udfDecode = udf(
            (geoHash: String) => GeoHash.decode(geoHash)
        )

        println("The most expensive home area 5km x 5km")
        castedDataFrame
            .withColumn(
                "geoHash",
                udfEncode(
                    castedDataFrame("latitude"),
                    castedDataFrame("longitude"),
                    lit(5)
                ))
            .groupBy(col("geoHash"))
            .avg("price")
            .orderBy("avg(price)")
            .limit(1)
            .withColumn(
                "coordinates",
                udfDecode(
                    col("geoHash")
                )
            )
            .show()

    }
}
