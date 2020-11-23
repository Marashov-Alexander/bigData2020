import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType

object Main {
    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("SparkHomework2")
            .master("local")
            .getOrCreate()

        val sparkContext = spark.sparkContext
        sparkContext.setLogLevel("ERROR")

        val dataFrame = spark.read
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .option("escape", "\"")
            .csv("src/main/resources/train.csv")

        val castedDataFrame = dataFrame
            .withColumn("id", dataFrame("id").cast("Long"))
            .withColumn("target", dataFrame("target").cast("Int"))

        castedDataFrame.printSchema()

        val regexTokenizer = new RegexTokenizer()
            .setInputCol("text")
            .setOutputCol("words")
            .setPattern("[\\W]")

        val remover = new StopWordsRemover()
            .setInputCol(regexTokenizer.getOutputCol)
            .setOutputCol("filtered_words")

        val stemmer = new Stemmer()
            .setInputCol(remover.getOutputCol)
            .setOutputCol("stemmed_words")
            .setLanguage("English")

        val hashingTF = new HashingTF()
            .setNumFeatures(10000)
            .setInputCol(stemmer.getOutputCol)
            .setOutputCol("rowFeatures")

        val idf = new IDF()
            .setInputCol(hashingTF.getOutputCol)
            .setOutputCol("features")

        val stringIndexer = new StringIndexer()
            .setInputCol("target")
            .setOutputCol("label")

        val gbt = new GBTClassifier()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setPredictionCol("predictionTarget")
            .setMaxIter(25)

        val pipeline = new Pipeline()
            .setStages(Array(regexTokenizer, remover, stemmer, hashingTF, idf, stringIndexer, gbt))

        val pipelineModel = pipeline.fit(castedDataFrame)

        val testDataFrame = spark.read
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .option("escape", "\"")
            .csv("src/main/resources/test.csv")

        val pipelinedTestData = pipelineModel.transform(testDataFrame)

        val result = pipelinedTestData
            .select("id", "predictionTarget")
            .withColumn("target", pipelinedTestData("predictionTarget").cast(IntegerType))
            .drop("predictionTarget")

        result
            .write
            .format("com.databricks.spark.csv")
            .option("header", "true")
            .save("src/main/resources/my_result.csv")

    }
}
