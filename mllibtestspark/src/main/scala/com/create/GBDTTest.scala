package com.create

import java.io

import org.apache.avro.generic.GenericData.StringType
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{DataTypes, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object GBDTTest {
  val labelColName = "safe_loans"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("gbdt_tets").master("local[4]").getOrCreate()
    var df = spark.read.option("header", "true").option("inferSchema", "true").csv("./gbdt_100.csv")
      .repartition(4)
      .na.replace("grade", Map("A" -> "2", "B" -> "5", "C" -> "1", "D" -> "3", "E" -> "4", "F" -> "6", "G" -> "7"))
      .where("grade != 'missing'")
      .withColumn("grade", col("grade").cast(DoubleType))
    val udfT = udf[Double, String](input => input.replaceFirst("[ ]+", " ").split(" ").head.toDouble)
    df = df.withColumn("term1", udfT(col("term"))).drop("term").withColumnRenamed("trem1", "term")

    val (piplineStages, featureColName) = this.featureTrans(df.schema)
//    val gbtClassfier = new GBTRegressor()
    val gbtClassfier = new GBTClassifier()
      .setLabelCol(this.labelColName)
      .setFeaturesCol(featureColName)
    val pipline = new Pipeline().setStages(piplineStages :+ gbtClassfier)
    val dfArr = df.randomSplit(Array(0.7, 0.3))
    val (trainningDF, testDF) = (dfArr.head, dfArr.last)


    val gridSearch = new ParamGridBuilder()
      .addGrid(gbtClassfier.maxDepth, Array(10))
      //      .addGrid(gbtClassfier.maxIter, Array(5 , 10))
      //      .addGrid(gbtClassfier.maxBins, Array(14, 18))
      .build()
    val metrics = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol(labelColName)
      .setMetricName("areaUnderROC")

    val crossValidator = new CrossValidator()
      .setEstimator(pipline)
      .setEvaluator(metrics)
      .setEstimatorParamMaps(gridSearch)
      .fit(trainningDF)
    crossValidator.transform(testDF).toJSON.foreach(println(_))
    println("auc:" + metrics.evaluate(crossValidator.transform(testDF)))
    println("avgMetrics:" + crossValidator.avgMetrics.mkString(","))
  }
  def normailizer(input:String)={
    val outFeatureName = input+"_normalizer"
    val normalizer = new Normalizer().setP(1).setInputCol(input).setOutputCol(outFeatureName)
    (normalizer,outFeatureName)
  }
  def featureTrans(schme: StructType) = {
    val (emps, strColArr) = this.stringFea(schme, Array[String](this.labelColName))
    val numColArr = this.numFea(schme, Array[String](this.labelColName))
    val (vectorAss, vectFeatureName) = this.calArr2Vector(numColArr ++ strColArr)
    val (mms, mmsFeatureName) = this.maxminScaler(vectFeatureName)
    val (normalizer,nomorColname) = this.normailizer(mmsFeatureName)
    (emps ++ Array(vectorAss, mms,normalizer), nomorColname)
//    (emps ++ Array(vectorAss, mms), vectFeatureName)
  }

  def stringFea(schema: StructType, excludeCol: Array[String]) = {
    val colAndFieldMap: Map[String, StructField] = schema.fields.map(tup => (tup.name, tup)).toMap -- excludeCol
    val strTypeColName: Array[String] = colAndFieldMap.filter(tup => tup._2.dataType == DataTypes.StringType).keys.toArray
    val stringToIndex: Array[(StringIndexer, String)] = this.string2Index(strTypeColName)
    val oneHotEncoder = this.oneHotEncoder(stringToIndex)
    //    val vectorAssembler = this.calArr2Vector(oneHotEncoder)
    Tuple2((stringToIndex ++ oneHotEncoder).map(_._1), oneHotEncoder.map(_._2))
  }

  def numFea(schema: StructType, exclude: Array[String]) = {
    val colAndFieldMap = schema.fields.map(tup => (tup.name, tup)).toMap -- exclude
    val numTyp = Set(
      DataTypes.ByteType,
      DataTypes.ShortType,
      DataTypes.IntegerType,
      DataTypes.LongType,
      DataTypes.FloatType,
      DataTypes.DoubleType
    )

    val cols = colAndFieldMap.values.filter(sf => numTyp.contains(sf.dataType)).map(_.name).toArray
    println("num type col:" + cols.mkString(","))
    cols
  }

  def maxminScaler[T](inputCol: String): (MinMaxScaler, String) = {
    (
      new MinMaxScaler()
        .setInputCol(inputCol)
        .setOutputCol(inputCol + "_mms")
        .setMin(-1)
        .setMax(1),
      inputCol + "_mms"
    )
  }

  def calArr2Vector[T](inputColArr: Array[String]): (VectorAssembler, String) = {
    val random = (Math.random() * 100).toInt
    (new VectorAssembler().setInputCols(inputColArr).setOutputCol("vect_" + random), "vect_" + random)
  }

  def string2Index(inputArr: Array[String]) = {
    println("string col name :" + inputArr.mkString(","))
    inputArr.map(field =>
      (new StringIndexer().setInputCol(field).setOutputCol(field + "_s2i"), field + "_s2i")
    )
  }

  def oneHotEncoder[T](input: Array[(T, String)]) = {
    input.map(field => {
      (new OneHotEncoder().setInputCol(field._2).setOutputCol(field._2 + "_ohe").setDropLast(true), field._2 + "_ohe")
    })
  }

}
