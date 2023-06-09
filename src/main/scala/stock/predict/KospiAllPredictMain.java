package stock.predict;

import javafx.util.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import stock.kospidata.KospIvstDataSetIter;
import stock.kospidata.KospiAllDsIterator;
import stock.model.RecurrentNets;
import stock.representation.PriceCategory;
import stock.utils.PlotUtil;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

public class KospiAllPredictMain {
    private static final Logger log = LoggerFactory.getLogger(KospiPredictMain.class);
    static int exampleLength = 22 ;

    public static void main(String ... v) throws IOException {
        System.out.println("Active ..");

        String file = "datafile/kospi_with_investor.csv";

        int batchSize = 64; // mini-batch size
        double splitRatio = 0.98; // 90% for training, 10% for testing
        int epochs = 10; // training epochs

        KospiAllDsIterator iterator = new KospiAllDsIterator(file, batchSize, exampleLength, splitRatio);

        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        log.info("Build lstm networks...");
        MultiLayerNetwork net = RecurrentNets.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

        File modelFile = new File("model/KospiAllIndex.mdl");

        log.info("Start Training ..");
        net.fit(iterator, epochs);
        log.info("Save Model ..");
        ModelSerializer.writeModel(net, modelFile, true);

        net = MultiLayerNetwork.load(modelFile, true);

        log.info("Testing ..");
        INDArray max = Nd4j.create(iterator.getMaxArray());
        INDArray min = Nd4j.create(iterator.getMinArray());

        predictAllCategories(net, test, max, min);

        System.out.println("Fin..");
    }

    /** Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead */
    static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        log.info("Print out Predictions and Actual Values...");
        log.info("Predict\tActual");
        for (int i = 0; i < predicts.length; i++)
            log.info("Predict : Actual => " + predicts[i] + "\t" + actuals[i]);
        log.info("Plot...");

        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name = "idx:" + n ;
//            switch (n) {
//                case 0: name = "Stock OPEN Price"; break;
//                case 1: name = "Stock CLOSE Price"; break;
//                case 2: name = "Stock LOW Price"; break;
//                case 3: name = "Stock HIGH Price"; break;
//                case 4: name = "Stock VOLUME Amount"; break;
//                default: throw new NoSuchElementException();
//            }
            PlotUtil.plot(pred, actu, name);
        }
    }
}
