package me.yg.hyper

import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.ROC
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
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


object ClinicalTimeMain extends App {
  println("Active Training ..")

  val DATA_PATH = "/Users/a1000074/dev/sample/"

  val NB_TRAIN_EXAMPLES = 3200 // number of training examples
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

  trainData.next().asList().forEach(a => println(a))
  println("-----------------------------")


//  while(trainData.hasNext) {
//
//  }

  // Load testing data
  val testFeatures = new CSVSequenceRecordReader(1, ",");
  testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

  val testLabels = new CSVSequenceRecordReader();
  testLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

  val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
    32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
//
//  // net configuration
//  // Set neural network parameters
//  val NB_INPUTS = 86
//  val NB_EPOCHS = 10
//  val RANDOM_SEED = 1234
//  val LEARNING_RATE = 0.005
//  val BATCH_SIZE = 32
//  val LSTM_LAYER_SIZE = 200
//  val NUM_LABEL_CLASSES = 2
//
//  val conf = new NeuralNetConfiguration.Builder()
//    .seed(RANDOM_SEED)
//    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//    .updater(new Adam(LEARNING_RATE))
//    .weightInit(WeightInit.XAVIER)
//    .dropOut(0.25)
//    .graphBuilder()
//    .addInputs("trainFeatures")
//    .setOutputs("predictMortality")
//    .addLayer("L1", new GravesLSTM.Builder()
//      .nIn(NB_INPUTS)
//      .nOut(LSTM_LAYER_SIZE)
//      .forgetGateBiasInit(1)
//      .activation(Activation.TANH)
//      .build(),
//      "trainFeatures")
//    .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//      .activation(Activation.SOFTMAX)
//      .nIn(LSTM_LAYER_SIZE)
//      .nOut(NUM_LABEL_CLASSES)
//      .build(), "L1")
//    .setOutputs("predictMortality")
//    .setInputTypes(InputType.recurrent(86))
//    .build()
//
//  val model = new ComputationGraph(conf)
//
//  // Training
//  model.fit(trainData, 2)
//
//  // Model Evaluation
//  val roc = new ROC(100);
//
//  while (testData.hasNext()) {
//    val batch = testData.next();
//    val output = model.output(batch.getFeatures());
//    roc.evalTimeSeries(batch.getLabels(), output(0));
//  }
//
//  println("FINAL TEST AUC: " + roc.calculateAUC());

}
