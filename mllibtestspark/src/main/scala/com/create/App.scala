package com.create

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{BooleanType, DoubleType, IntegerType}
import org.netlib.blas.Ddot

/**
  * Hello world!
  *
  */
object App {
  lazy val spark = SparkSession.builder().appName("spark_mllib").master("local[10]").getOrCreate()
  val dataSetPath = "./gbdt_100.csv"
  val labelFieldName = "safe_loans"

  def main(args: Array[String]): Unit = {
    val dataSet = spark.read.option("header", "true").option("inferSchema", "true").csv("./processed_gbdt_100.csv")
    val dataSetArr = dataSet.select(dataSet.schema.fields.map(field => {
      if (!field.name.equals(labelFieldName)) {
        col(field.name).cast(DoubleType)
      } else {
        col(field.name).cast(IntegerType)
      }
    }): _*).randomSplit(Array(0.7, 0.3))
    val (tainningDataSet, testDataSet) = (dataSetArr(0), dataSetArr(1))
    val vectorAssembler = new VectorAssembler().setInputCols(dataSet.schema.fields.map(_.name)/*.filter(!_.equals(labelFieldName))*/)
      .setOutputCol("features")
//        val gbt = new GBTRegressor()
    val gbt = new GBTClassifier()
      .setLabelCol(labelFieldName)
      .setFeaturesCol("features")
//      .ensuring(true)
    //      .setMaxIter(40)
    //      .setMaxDepth(20)

    val pipeline = new Pipeline().setStages(Array(vectorAssembler, gbt))
    val model: PipelineModel = pipeline.fit(tainningDataSet)
    val preductionAndLabel = model.transform(testDataSet)
      .withColumnRenamed(labelFieldName, "label")
    preductionAndLabel.toJSON.foreach(println(_))
//      .select("prediction", "label")
    preductionAndLabel.printSchema()
//    preductionAndLabel.foreach(println(_))
//    val pre = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
//      .setLabelCol("label")
//      .setRawPredictionCol("prediction")
//    println(pre.evaluate(preductionAndLabel))
    //    val auc = mitrics.evaluate(preductionAndLabel)
    //    val mitrics = new BinaryClassificationEvaluator()
    //      .setLabelCol(labelFieldName)
    //      .setRawPredictionCol("prediction")
    //      .setMetricName("areaUnderROC")
    //    val paramGrid = new ParamGridBuilder()
    //        .addGrid(gbt.maxBins,Array(5,10,20,25))
    //        .addGrid(gbt.maxDepth,Array(4,6,8,10,12))
    ////        .addGrid(gbt.impurity,Array("entropy","gini"))
    ////        .addGrid(gbt.impurity,Array("gini"))
    //        .build()
    //    val cv = new CrossValidator()
    //        .setEstimator(pipeline)
    //        .setEvaluator(mitrics)
    //        .setEstimatorParamMaps(paramGrid)
    //        .setNumFolds(2)
    //    val piplineFittedModel: CrossValidatorModel = cv.fit(tainningDataSet)
    //    piplineFittedModel.bestModel.params
    //    val predictions: DataFrame = piplineFittedModel.transform(testDataSet)
    //
    //    val accuracy: Double = mitrics.evaluate(predictions)
    //    println(accuracy)
  }

  def tmp() = {
    val dataSet = spark.read.option("header", "true").option("inferSchema", "true").csv("./processed_gbdt_100.csv")
    dataSet.foreach(println(_))
    dataSet.printSchema()
    val dataSetArr = dataSet.select(dataSet.schema.fields.map(field => {
      if (!field.name.equals(labelFieldName)) {
        col(field.name).cast(DoubleType)
      } else {
        col(field.name).cast(IntegerType)
      }
    }): _*).randomSplit(Array(0.7, 0.3))
    val (tainningDataSet, testDataSet) = (dataSetArr(0), dataSetArr(1))
    val vectorAssembler = new VectorAssembler().setInputCols(dataSet.schema.fields.map(_.name).filter(!_.equals(labelFieldName)))
      .setOutputCol("features")
        val gbt = new GBTRegressor()
//    val gbt = new GBTClassifier()
      .setLabelCol(labelFieldName)
      .setFeaturesCol("features")
      .setMaxIter(40)
      .setMaxDepth(20)
    val pipeline = new Pipeline().setStages(Array(vectorAssembler, gbt))
    //    val model = pipeline.fit(tainningDataSet)
    //    val preductionAndLabel = model.transform(testDataSet)
    //      .withColumnRenamed(labelFieldName, "label")
    //      .select("prediction", "label")
    val mitrics = new BinaryClassificationEvaluator()
      .setLabelCol(labelFieldName)
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    //    val auc = mitrics.evaluate(preductionAndLabel)
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxBins, Array(5, 10, 20, 25))
      .addGrid(gbt.maxDepth, Array(4, 6, 8, 10, 12))
      //        .addGrid(gbt.impurity,Array("entropy","gini"))
      //        .addGrid(gbt.impurity,Array("gini"))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(mitrics)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
    val piplineFittedModel: CrossValidatorModel = cv.fit(tainningDataSet)
    piplineFittedModel.bestModel.params
    val predictions: DataFrame = piplineFittedModel.transform(testDataSet)

    val accuracy: Double = mitrics.evaluate(predictions)
    println(accuracy)
  }

  def splitRank(dataFrame: DataFrame, typeName: Int): DataFrame = {
    new Bucketizer().setInputCol("dti").setOutputCol("dtiIndex").setSplits(Array(0.0, 10.0, 30.0, 50.0, Double.PositiveInfinity))
      .transform(dataFrame)
  }

  def geneAllFieldStringToIndex(dataFrame: DataFrame, labelFieldName: String): Array[(String, String, StringIndexer)] = {
    dataFrame.schema.fields.filter(!_.name.equals(labelFieldName)).map(fieldStr => {
      val field = fieldStr.name
      (field, s"${field}Index", new StringIndexer().setInputCol(field).setOutputCol(s"${field}Index"))
    })
  }

  def pipLine() = {
    /** 加载数据集 */
    var dataSet: DataFrame = spark.read.option("header", "true").csv(dataSetPath)
      .withColumn(labelFieldName, col(labelFieldName).cast(IntegerType))
      .withColumn("dti", col("dti").cast(DoubleType))
    /** 开发测试 */
    val dataSetArr = dataSet.randomSplit(Array(0.7, 0.3))
    val (trainingDataSet, testDataSet) = (dataSetArr(0), dataSetArr(1))
    val stringToIndex = this.geneAllFieldStringToIndex(dataSet, this.labelFieldName)


    new ChiSqSelector()
    stringToIndex.foreach(tmp => {
      if (tmp._1.equals("dti")) {
        dataSet = new Bucketizer()
          .setInputCol("dti")
          .setOutputCol("dtiIndex")
          .setSplits(Array(0.0, 10.0, 30.0, 50.0, Double.PositiveInfinity))
          .transform(dataSet)
      } else {
        dataSet = tmp._3.fit(dataSet).transform(dataSet)
      }

    })
    val vectorAsse = new VectorAssembler()
      .setInputCols(stringToIndex.map(tmp => tmp._2))
      .setOutputCol("feature")
    val gbt = new GBTRegressor()
      .setLabelCol(labelFieldName)
      .setFeaturesCol("feature")
      .setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(vectorAsse, gbt))
    val model = pipeline.fit(dataSet)

    val prediction = model.transform(dataSet)
    prediction.printSchema()

  }

  def predictData(testDataSet: RDD[LabeledPoint], model: GradientBoostedTreesModel) = {
    val modelBC = spark.sparkContext.broadcast(model)
    testDataSet.mapPartitions(iterator => {
      val modelTmp = modelBC.value
      iterator.map(item => {
        (item.label, modelTmp.predict(item.features))
      })
    })
  }

  /**
    * 训练模型
    *
    * @param trainingDataSet
    * @return
    */
  def trainModel(trainingDataSet: RDD[LabeledPoint]): GradientBoostedTreesModel = {
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    GradientBoostedTrees.train(trainingDataSet, boostingStrategy)
  }
}