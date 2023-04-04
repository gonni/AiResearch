package me.yg.hyper

import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.ROC
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.io.File
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils

import java.io.IOException
import java.util
import java.net.URL
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.lang.Byte
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.regression.RegressionEvaluation

import scala.jdk.CollectionConverters._

object ClinicalTimeMain extends App {
  println("Active Training ..")

  val DATA_PATH = "/Users/ygkim/dev/sample/"

  val NB_TRAIN_EXAMPLES = 1 //3200 // number of training examples
  val NB_TEST_EXAMPLES = 800 // number of testing examples

  // init data
  val path = FilenameUtils.concat(DATA_PATH, "physionet2012/") // set parent directory

  val featureBaseDir = FilenameUtils.concat(path, "sequence") // set feature directory
  val mortalityBaseDir = FilenameUtils.concat(path, "mortality") // set label directory

  // Load training data

  val trainFeatures = new CSVSequenceRecordReader(1, ",")
  trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1))

  val trainLabels = new CSVSequenceRecordReader()
  trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1))

  val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
    32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)


  println("-----------------------------")
  println(trainLabels.next().get(0).toString)

//  trainDt.asList().asScala.map(ds => ds.asList().forEach(println))
//  trainFeatures.next().asScala.foreach(println)

//  println("--------------ds---------------")
////  println("==========>" + trainData.next().asList().size())
////  trainData.next().asScala.foreach(println)
//  trainData.next().asScala.foreach(ds => {
////    println("row ===========================")
//    ds.get(0).asList().asScala.foreach(col => {
////      println("col =============")
//      col.toString
//    })
//  })


  // Load testing data
  val testFeatures = new CSVSequenceRecordReader(1, ",");
  testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

  val testLabels = new CSVSequenceRecordReader();
  testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

  val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
    32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

  // ----------------
  processing

  def processing() = {
    // net configuration
    // Set neural network parameters
    val NB_INPUTS = 86 // 시간 포함 시간당 측정 항목, 총 86건
    val NB_EPOCHS = 10
    val RANDOM_SEED = 1234
    val LEARNING_RATE = 0.005
    val BATCH_SIZE = 32
    val LSTM_LAYER_SIZE = 200
    val NUM_LABEL_CLASSES = 2

    val conf  = new NeuralNetConfiguration.Builder()
      .seed(RANDOM_SEED)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(LEARNING_RATE))
      .weightInit(WeightInit.XAVIER)
      .dropOut(0.25)
      .graphBuilder()
      .addInputs("trainFeatures")
      .setOutputs("predictMortality")
      .addLayer("L1", new GravesLSTM.Builder()
        .nIn(NB_INPUTS)
        .nOut(LSTM_LAYER_SIZE)
        .forgetGateBiasInit(1)
        .activation(Activation.TANH)
        .build(),
        "trainFeatures")
      .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(LSTM_LAYER_SIZE)
        .nOut(NUM_LABEL_CLASSES)
        .build(), "L1")
      .setOutputs("predictMortality")
      .setInputTypes(InputType.recurrent(86))
      .build()

    // Load
    val model = ComputationGraph.load(new File("model/clinic.mdl"), false)

    // Training
//    val model = new ComputationGraph(conf)
//    model.fit(trainData, 2)
//    model.save(new File("model/clinic.mdl"))

    // RegEvaluation
//    println("--------------------------------------------------------")
//    val eval = model.evaluateRegression[RegressionEvaluation](testData)
//
//    testData.reset()
//    println
//
//    println("Eval => " + eval.stats())



    // Model Evaluation
    val roc = new ROC(2);
    while (testData.hasNext()) {
      val batch = testData.next();
      val output = model.output(batch.getFeatures());
      println("-- (processing) --------->")
//      for (elem <- output.toList) {
//        println("elem :" + elem.getDouble(0L) + "/" + elem.getDouble(1L)
//          + "/" + elem.getDouble(2L) + "/" + elem.getDouble(3L))
//      }

//      println("batchLabel =>" + batch.getLabels.getRow(0L))
      println(output.length + " -- " + output(0))
      roc.evalTimeSeries(batch.getLabels(), output(0));
      println("AUC in Processing = " + roc.calculateAUC())
    }


    println("FINAL TEST AUC: " + roc.calculateAUC());


//    println("Eval => " + model.evaluate(testData))
//    println("-----------------------------------")

    // Evaluation2
//    val eval = new Evaluation(2)
//    while(testData.hasNext) {
//      val next = testData.next()
//      val output = model.output(next.getFeatures)
//      eval.eval(next.getLabels, output(0))
//    }
//
//    println(eval.stats)
//    println("Finished ..")
  }
}

