package stock.kospidata;

import com.google.common.collect.ImmutableMap;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import stock.representation.PriceCategory;
import stock.representation.StockData;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class KospIvstDataSetIter implements DataSetIterator {

    private final int VECTOR_SIZE = 9; // number of features for a stock data
    private int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month
    private int predictLength = 1; // default 1, say, one day ahead prediction

    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** mini-batch offset */
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<KospiData> train ;
    private List<Pair<INDArray, INDArray>> test;

    public KospIvstDataSetIter(String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio) {

    }

    public KospIvstDataSetIter() {
    }

    private List<KospiData> readKospiDataFromFile(String filename) {
        List<KospiData> kospiDataList = new ArrayList<>();
        try {
            for (int i = 0; i < maxArray.length; i++) { // initialize max and min arrays
                maxArray[i] = Double.MIN_VALUE;
                minArray[i] = Double.MAX_VALUE;
            }
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
            for (String[] arr : list) {
                if (arr[0].contains("TARGET")) continue;
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length - 1; i++) {
                    nums[i] = Double.valueOf(arr[i + 1]);
                    if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                    if (nums[i] < minArray[i]) minArray[i] = nums[i];
                }
                //stockDataList.add(new StockData(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], nums[4]));
                kospiDataList.add(new KospiData(
                        arr[0],nums[0],nums[1],nums[2],nums[3],nums[4],nums[5],nums[6],nums[7],nums[8]
                ));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return kospiDataList;
    }

    public static void main(String ... v) {
        System.out.println("Test Module");
        KospIvstDataSetIter test = new KospIvstDataSetIter();
        test.readKospiDataFromFile("datafile/kospi_with_investor.csv").forEach(System.out::println);
    }

    @Override
    public DataSet next(int num) {
        return null;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public DataSet next() {
        return null;
    }
}
