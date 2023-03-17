package me.yg.hyper

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, LSTM, RnnOutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder
import org.deeplearning4j.nn.conf.Updater
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.nn.conf.inputs.InputType

import java.io.File
import java.net.URL
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import org.apache.commons.io.FilenameUtils
import org.apache.commons.io.FileUtils
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.nd4j.linalg.learning.config.{AdaGrad, Adam}

object SeaTempLstmMain extends App {
  val DATA_URL = "https://dl4jdata.blob.core.windows.net/training/seatemp/sea_temp.tar.gz"
//  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_seas/")
  val DATA_PATH = "/Users/a1000074/dev/sample/"

//  // Init File
//  val directory = new File(DATA_PATH)
//  directory.mkdir()
//
//  val archizePath = DATA_PATH + "sea_temp.tar.gz"
//  val archiveFile = new File(archizePath)
//  val extractedPath = DATA_PATH + "sea_temp"
//  val extractedFile = new File(extractedPath)
//
//  FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
//
//  // Extract source file
//  var fileCount = 0
//  var dirCount = 0
//  val BUFFER_SIZE = 4096
//  val tais = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(archizePath))))
//
//  var entry = tais.getNextEntry().asInstanceOf[TarArchiveEntry]
//
//  while (entry != null) {
//    if (entry.isDirectory()) {
//      new File(DATA_PATH + entry.getName()).mkdirs()
//      dirCount = dirCount + 1
//      fileCount = 0
//    }
//    else {
//
//      val data = new Array[scala.Byte](4 * BUFFER_SIZE)
//
//      val fos = new FileOutputStream(DATA_PATH + entry.getName());
//      val dest = new BufferedOutputStream(fos, BUFFER_SIZE);
//      var count = tais.read(data, 0, BUFFER_SIZE)
//
//      while (count != -1) {
//        dest.write(data, 0, count)
//        count = tais.read(data, 0, BUFFER_SIZE)
//      }
//
//      dest.close()
//      fileCount = fileCount + 1
//    }
//    if (fileCount % 1000 == 0) {
//      print(".")
//    }
//
//    entry = tais.getNextEntry().asInstanceOf[TarArchiveEntry]
//  }

  println("Active Deep Learning ..")
  // DataSetIterators
  val path = FilenameUtils.concat(DATA_PATH, "sea_temp/") // set parent directory

  val featureBaseDir = FilenameUtils.concat(path, "features") // set feature directory
  val targetsBaseDir = FilenameUtils.concat(path, "targets") // set label directory

  // Init Train & Test Data
  val numSkipLines = 1;
  val regression = true;
  val batchSize = 32;

  val trainFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
  trainFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1, 1936));
  val trainTargets = new CSVSequenceRecordReader(numSkipLines, ",");
  trainTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1, 1936));

  val train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainTargets, batchSize,
    10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);


  val testFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
  testFeatures.initialize(new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1937, 2089));
  val testTargets = new CSVSequenceRecordReader(numSkipLines, ",");
  testTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1937, 2089));

  val test = new SequenceRecordReaderDataSetIterator(testFeatures, testTargets, batchSize,
    10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);

  // Neural Network
  val V_HEIGHT = 13;
  val V_WIDTH = 4;
  val kernelSize = 2;
  val numChannels = 1;

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .seed(12345)
    .weightInit(WeightInit.XAVIER)
    .updater(new AdaGrad(0.005))
    .list()
    .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
      .nIn(1) //1 channel
      .nOut(7)
      .stride(2, 2)
      .activation(Activation.RELU)
      .build())
    .layer(1, new LSTM.Builder()
      .activation(Activation.SOFTSIGN)
      .nIn(84)
      .nOut(200)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(10)
      .build())
    .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
      .activation(Activation.IDENTITY)
      .nIn(200)  //200
      .nOut(52)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(10)
      .build())
    .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
    .inputPreProcessor(1, new CnnToRnnPreProcessor(6, 2, 7))
//    .setInputType(InputType.convolutionalFlat(V_HEIGHT, V_WIDTH, numChannels))
    .build();

//  val conf = new NeuralNetConfiguration.Builder()
//    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//    .seed(12345)
//    .updater(new Adam())
//    .weightInit(WeightInit.XAVIER)
//    .list()
//    .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
//      .nIn(numChannels) //1 channel
//      .nOut(7)
//      .stride(2, 2)
//      .activation(Activation.RELU)
//      .build())
//    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//      .kernelSize(kernelSize, kernelSize)
//      .stride(2, 2).build())
//    .layer(2, new LSTM.Builder()
//      .activation(Activation.SOFTSIGN)
//      .nIn(21)
//      .nOut(100)
//      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//      .gradientNormalizationThreshold(10)
//      .build())
//    .layer(3, new RnnOutputLayer.Builder(LossFunction.MSE)
//      .activation(Activation.IDENTITY)
//      .nIn(100)
//      .nOut(52)
//      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//      .gradientNormalizationThreshold(10)
//      .build())
//    .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
//    .inputPreProcessor(2, new CnnToRnnPreProcessor(3, 1, 7))
//    .build();

  val net = new MultiLayerNetwork(conf);
  net.init();

  // Model Training
  // Train model on training set
  net.fit(train, 25)

  // Model Evaluation
  val eval = net.evaluateRegression[RegressionEvaluation](test);

  test.reset();
  println()

  println(eval.stats());

  println("Processing Completed !!")
}
