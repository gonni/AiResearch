package me.yg.lstm;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class SimplePrediction {
    private static Logger log = LoggerFactory.getLogger(SimplePrediction.class);
    private static int batchSize = 64;
    private static long seed = 123;
    private static int numEpochs = 3;
    private static boolean modelType = true;

    public static String dataLocalPath = "/Users/ygkim/dl4j-examples-data/dl4j-examples/lottery";

    public static void main(String ... v) {
        System.out.println("Active System ..");

        DataSetIterator trainIterator = new LotteryDataSetIterator(new File(dataLocalPath,"cqssc_train.csv").getAbsolutePath(), batchSize, modelType);
        DataSetIterator testIterator = new LotteryDataSetIterator(new File(dataLocalPath, "cqssc_test.csv").getAbsolutePath(), 2, modelType);
        DataSetIterator validateIterator = new LotteryDataSetIterator(new File(dataLocalPath, "cqssc_validate.csv").getAbsolutePath(), 2, modelType);

        MultiLayerNetwork model = getNetModel(trainIterator.inputColumns(), trainIterator.totalOutcomes());
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
//        uiServer.attach(statsStorage);

        // print layers and parameters
        System.out.println(model.summary());

        // training
//        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

        try {
            model = MultiLayerNetwork.load(new File("/Users/ygkim/dl4j-examples-data/dl4j-examples/lottery/" +
                    "lotteryPredictModel.json"), true);
        } catch(Exception e) {
            e.printStackTrace();
        }

        int luckySize = 5;
        if (modelType && model != null) {
            while (testIterator.hasNext()) {
                DataSet ds = testIterator.next();
                //predictions all at once
                INDArray output = model.output(ds.getFeatures());
                INDArray label = ds.getLabels();
                INDArray preOutput = Nd4j.argMax(output, 2);
                INDArray realLabel = Nd4j.argMax(label, 2);
                StringBuilder peLabel = new StringBuilder();
                StringBuilder reLabel = new StringBuilder();
                for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                    peLabel.append(preOutput.getInt(dataIndex));
                    reLabel.append(realLabel.getInt(dataIndex));
                }
                log.info("test-->real lottery {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), peLabel.toString().equals(reLabel.toString()));
            }
            while (validateIterator.hasNext()) {
                DataSet ds = validateIterator.next();
                //predictions all at once
                INDArray output = model.feedForward(ds.getFeatures()).get(0);
                INDArray label = ds.getLabels();
                INDArray preOutput = Nd4j.argMax(output, 2);
                INDArray realLabel = Nd4j.argMax(label, 2);
                StringBuilder peLabel = new StringBuilder();
                StringBuilder reLabel = new StringBuilder();
                for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                    peLabel.append(preOutput.getInt(dataIndex));
                    reLabel.append(realLabel.getInt(dataIndex));
                }
                log.info("validate-->real lottery {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), peLabel.toString().equals(reLabel.toString()));
            }

            String currentLottery = "27578";
            INDArray initCondition = Nd4j.zeros(1, 5, 10);
            String[] featureAry = currentLottery.split("");
            for (int j = 0; j < featureAry.length; j++) {
                int p = Integer.parseInt(featureAry[j]);
                initCondition.putScalar(new int[]{0, j, p}, 1);
            }
            INDArray output = model.output(initCondition);
            INDArray preOutput = Nd4j.argMax(output, 2);
            StringBuilder latestLottery = new StringBuilder();
            for (int dataIndex = 0; dataIndex < 5; dataIndex++) {
                latestLottery.append(preOutput.getInt(dataIndex));
            }
            System.out.println("current lottery numbers==" + currentLottery + "==prediction===next lottery numbers==" + latestLottery.toString());

        }
    }

    //create the neural network
    private static MultiLayerNetwork getNetModel(int inputNum, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp.Builder().rmsDecay(0.95).learningRate(1e-2).build())
                .list()
                .layer(new LSTM.Builder().name("lstm1")
                        .activation(Activation.TANH).nIn(inputNum).nOut(100).build())
                .layer(new LSTM.Builder().name("lstm2")
                        .activation(Activation.TANH).nOut(80).build())
                .layer(new RnnOutputLayer.Builder().name("output")
                        .activation(Activation.SOFTMAX).nOut(outputNum).lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

}
