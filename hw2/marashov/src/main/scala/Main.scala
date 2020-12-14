import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.sql.SparkSession

object Main {
    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("SparkHomework2")
            .master("local")
            .getOrCreate()

        val sparkContext = spark.sparkContext
        sparkContext.setLogLevel("OFF")

        val dataFrame = spark.read
            .option("header", "true")
            .option("mode", "DROPMALFORMED")
            .option("escape", "\"")
            .csv("marashov/src/main/resources/train.csv")

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

//        val stemmer = new Stemmer()
//            .setInputCol(remover.getOutputCol)
//            .setOutputCol("stemmed_words")
//            .setLanguage("English")

        val hashingTF = new HashingTF()
            .setNumFeatures(5000)
            .setInputCol(remover.getOutputCol)
            .setOutputCol("rowFeatures")

        val idf = new IDF()
            .setInputCol(hashingTF.getOutputCol)
            .setOutputCol("features")

        val stringIndexer = new StringIndexer()
            .setInputCol("target")
            .setOutputCol("label")

        val rfc = new RandomForestClassifier()
          .setLabelCol("label")
          .setFeaturesCol("features")
          .setPredictionCol("predictionTarget")
          .setNumTrees(10)

        val pipeline = new Pipeline()
            .setStages(Array(regexTokenizer, remover, hashingTF, idf, stringIndexer, rfc))

        val pipelineModel = pipeline.fit(castedDataFrame)

        pipelineModel.write.overwrite().save("./myModel2")

    }
}
