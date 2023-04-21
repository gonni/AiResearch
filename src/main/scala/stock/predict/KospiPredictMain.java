package stock.predict;

import javafx.util.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import stock.kospidata.KospIvstDataSetIter;
import stock.model.RecurrentNets;
import stock.utils.PlotUtil;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class KospiPredictMain {
    private static final Logger log = LoggerFactory.getLogger(KospiPredictMain.class);
    static int exampleLength = 22 ;
    public static void main(String ... v) throws IOException {
        System.out.println("Active ..");

        String file = "datafile/kospi_with_investor.csv";

        int batchSize = 64; // mini-batch size
        double splitRatio = 0.98; // 90% for training, 10% for testing
        int epochs = 10; // training epochs

        KospIvstDataSetIter iterator = new KospIvstDataSetIter(file, batchSize, exampleLength, splitRatio);

        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        File modelFile = new File("model/KospiIndex.mdl");

        log.info("Start Training ..");
        net.fit(iterator, epochs);
        log.info("Save Model ..");
        ModelSerializer.writeModel(net, modelFile, true);

        log.info("Testing ..");
        double maxIndexValue = iterator.getMaxIndexValue();
        double minIndexValue = iterator.getMinIndexValue();

        predictIndexValue(net, test, maxIndexValue, minIndexValue);

        System.out.println("Fin..");
    }

    static void predictIndexValue(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData,
                                  double max, double min) {
        log.info("Start Verification ..") ;
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];

        for (int i = 0; i < testData.size(); i++) {
            INDArray output = net.rnnTimeStep(testData.get(i).getKey());
            System.out.println(i + "\toutput =>" + output);

            predicts[i] = output.getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);

        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict,Actual");
        for (int i = 0; i < predicts.length; i++) log.info(predicts[i] + "," + actuals[i]);
        log.info("Plot...");
        PlotUtil.plot(predicts, actuals, "KOSPI");

    }

}
