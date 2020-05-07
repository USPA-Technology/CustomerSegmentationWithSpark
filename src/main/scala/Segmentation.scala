import java.io.File

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, DoubleType}
import org.apache.spark.sql.functions.{col, count, to_timestamp, round, max, date_add, 
									   lit, datediff, min, sum, udf, concat, mean, asc, desc}
import org.apache.spark.ml.feature.{VectorAssembler, MinMaxScaler}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object Segmentation {

	def RScore(recencyPercentiles: Array[Double]) = udf((recency: Int) => {
		if (recency <= recencyPercentiles(0)) {
			1
		}
		else if (recency <= recencyPercentiles(1)) {
			2
		}
		else if (recency <= recencyPercentiles(2)) {
			3
		}
		else {
			4
		}
	})

	def FScore(frequencyPercentiles: Array[Double]) = udf((frequency: Int) => {
		if (frequency <= frequencyPercentiles(0)) {
			4
		}
		else if (frequency <= frequencyPercentiles(1)) {
			3
		}
		else if (frequency <= frequencyPercentiles(2)) {
			2
		}
		else {
			1
		}
	})

	def MScore(monetaryPercentiles: Array[Double]) = udf((monetary: Int) => {
		if (monetary <= monetaryPercentiles(0)) {
			4
		}
		else if (monetary <= monetaryPercentiles(1)) {
			3
		}
		else if (monetary <= monetaryPercentiles(2)) {
			2
		}
		else {
			1
		}
	})

	def main(args: Array[String]): Unit = {
		// set log levels
		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.OFF)

		// create spark SparkSession
		val spark = SparkSession
			.builder
			.appName("RFM Analysis")
			.getOrCreate()
		import spark.implicits._

		val retailDataHomeDir = "/home/vignesh/my_drive/Data Science/spark/CustomerSegmentationOnRetailData"

		val retails_raw = spark.read
			.format("csv")
			.option("header", "true")
			.load(new File(retailDataHomeDir, "ecommerce.csv").toString())
			.withColumn("Quantity", col("Quantity").cast(IntegerType))
			.withColumn("UnitPrice", col("UnitPrice").cast(DoubleType))

		// data exploration
		// println("retail data : ")
		// println(retails_raw.show(5))

		// println("retail data schema : ")
		// println(retails_raw.printSchema())

		// println("no. of records : " + retails_raw.count())

		// // check for null values column wise
		// println("column wise non-null value count : ")
		// println(retails_raw.select(retails_raw.columns.map(c => count(col(c)).alias(c)):_*).show())

		// drop null valued rows
		val retailsNullRem = retails_raw.na.drop
		// println("no. of records after dropping null values : ")
		// println(retailsNullRem.select(retailsNullRem.columns.map(c => count(col(c)).alias(c)):_*).show())

		// // lets convert InvoiceDate column to timeStamp format
		val timeFmt = "yyyy-MM-dd HH:mm:ss"
		val retailsPrep = retailsNullRem
			.withColumn("InvoiceDateTS", to_timestamp(col("InvoiceDate"), timeFmt))

		val snapDate = retailsPrep.select(max(col("InvoiceDateTS"))).first().get(0)

		val retails = retailsPrep
			.withColumn("snapDate", date_add(lit(snapDate), 1))
			.withColumn("duration", datediff(col("snapDate"), col("InvoiceDate")))
			.withColumn("TotalPrice", round(col("Quantity") * col("UnitPrice"), 2))

		println("retail data : ")
		println(retails.show(10))

		println("retail data schema : ")
		println(retails.printSchema())

		val retailsGrpd = retails.groupBy("CustomerID")
							.agg(
								min("duration").alias("Recency"),
								count("InvoiceNo").alias("Frequency"),
								round(sum("TotalPrice"), 2).alias("Monetary")
							)

		println("data grouped by customer : ")
		println(retailsGrpd.show(10))
		// println("no. of customer records : " + retailsGrpd.count())

		val recencyPercentiles = retailsGrpd.stat.approxQuantile("Recency", Array(0.25, 0.50, 0.75), 0.1)
		val frequencyPercentiles = retailsGrpd.stat.approxQuantile("Frequency", Array(0.25, 0.50, 0.75), 0.1)
		val monetaryPercentiles = retailsGrpd.stat.approxQuantile("Monetary", Array(0.25, 0.50, 0.75), 0.1)

		// println(recencyPercentiles.mkString(", "))
		// println(frequencyPercentiles.mkString(", "))
		// println(monetaryPercentiles.mkString(", "))

		// send both col value and percentile array using currying in scala 
		val retailsRFM = retailsGrpd
			.withColumn("RScore", RScore(recencyPercentiles)(col("Recency")))
			.withColumn("FScore", FScore(frequencyPercentiles)(col("Frequency")))
			.withColumn("MScore", MScore(monetaryPercentiles)(col("Monetary")))
			.withColumn("RFMScore", concat(col("RScore"), col("FScore"), col("MScore")))
			.select("CustomerID", "Recency", "Frequency", "Monetary", "RScore", "FScore", "MScore", "RFMScore")

		// println("final df: ")
		// println(retailsRFM.show(10))
		println(retailsRFM.getClass())

		println("Statictics based on RFM clustering")
		println(retailsRFM.groupBy("RFMScore")
					.agg(count("CustomerID").alias("No_of_customers"),
						 round(mean("Recency"), 2).alias("avg_Recency"),
						 round(mean("Frequency"), 2).alias("avg_Frequency"),
						 round(mean("Monetary"), 2).alias("avg_Monetary")
						)
					.orderBy(asc("RFMScore"))
					.show()
				)

		/*
			For RScore, FScore and MScore the values range from 1 to 4, 
			with 1 being the best score and 4 being the worst score.
			RFM Score can be interpreted as below :
			111  -> Best Customer purchases most often and most recently, with high spending pattern
			311 -> Haven't purchased for some time, but purchased frequently and spend the most
			444 -> Last purchases long ago, purchased few and spent little
			etc..
		*/

		// SEGMENTATION USING K-Means CLUSTERING
		val cols = Array("Recency", "Frequency", "Monetary")
		val assembler = new VectorAssembler()
			.setInputCols(cols)
			.setOutputCol("rfm")
		val rfmDF = assembler.transform(retailsRFM.select("CustomerID", "Recency", "Frequency", "Monetary"))
		
		// println("rfm df : ")
		// println(rfmDF.show())
		// println(rfmDF.printSchema())

		val scalar = new MinMaxScaler()
			.setInputCol("rfm")
			.setOutputCol("features")
			.setMax(1)
			.setMin(-1)
		val featureDF = scalar.fit(rfmDF).transform(rfmDF)

		// println("feature df : ")
		// println(featureDF.show(20, false))
		
		var best_silhouette: Double = -2.00
		var best_k_val: Int = -1
		var best_model: KMeansModel = null

		for(k_val <- 3 to 20) {
			val kmeans = new KMeans()
				.setK(k_val)
				.setFeaturesCol("features")
				.setPredictionCol("prediction")

			// train the model 
			val model = kmeans.fit(featureDF)

			// make predictions
			val predictions = model.transform(featureDF)

			// evaluate clustering by computing silhouette Score
			val evaluator = new ClusteringEvaluator()

			val silhouette = evaluator.evaluate(predictions)
			println(s"for k : $k_val silhouette with squared euclidean distance : $silhouette")

			if (silhouette > best_silhouette) {
				best_silhouette = silhouette
				best_k_val = k_val
				best_model = model
			}
		}

		println(s"best k value : $best_k_val and has a Silhouette : $best_silhouette")

		val predictions = best_model.transform(featureDF)
		println("prediction with the best model ")
		println(predictions.select("CustomerID", "Recency", "Frequency", "Monetary", "prediction").show())

		println("statistics based on KMeans clustering")
		println(predictions.groupBy("prediction")
					.agg(
						count("CustomerID").alias("No_of_customers"),
						round(mean("Recency"), 2).alias("avg_Recency"),
						round(mean("Frequency"), 2).alias("avg_Frequency"),
						round(mean("Monetary"), 2).alias("avg_Monetary")
					)
					.orderBy(desc("No_of_customers"))
					.show()
			)

	}
}