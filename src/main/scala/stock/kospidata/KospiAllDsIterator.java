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

public class KospiAllDsIterator implements DataSetIterator {

    private final int VECTOR_SIZE = 9; // number of features for a stock data
    private int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month
    private int predictLength = 9; // default 1, say, one day ahead prediction

    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** mini-batch offset */
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<KospiData> train ;
    private List<Pair<INDArray, INDArray>> test;

    public KospiAllDsIterator(String filename, int miniBatchSize, int exampleLength, double splitRatio) {
        List<KospiData> allKospiDataList = readKospiDataFromFile(filename);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        int split = (int) Math.round(allKospiDataList.size() * splitRatio);
        System.out.println("----- train:test split :" + split);
        train = allKospiDataList.subList(0, split);
        test = generateTestDataSet(allKospiDataList.subList(split, allKospiDataList.size()));

    }

    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<KospiData> stockDataList) {
        int window = exampleLength + predictLength; // = 22 + 1
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();

        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + exampleLength; j++) {
                // -- Create 22(exampleLength) rows with 9 columns per row
                KospiData stock = stockDataList.get(j);
                input.putScalar(new int[] {j - i, 0}, (stock.getIndexValue() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[] {j - i, 1}, (stock.getTotalEa() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[] {j - i, 2}, (stock.getTotalVolume() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[] {j - i, 3}, (stock.getAnt() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[] {j - i, 4}, (stock.getForeigner() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[] {j - i, 5}, (stock.getCompany() - minArray[5]) / (maxArray[5] - minArray[5]));
                input.putScalar(new int[] {j - i, 6}, (stock.getInvestBank() - minArray[6]) / (maxArray[6] - minArray[6]));
                input.putScalar(new int[] {j - i, 7}, (stock.getInvestTrust() - minArray[7]) / (maxArray[7] - minArray[7]));
                input.putScalar(new int[] {j - i, 8}, (stock.getPensionFund() - minArray[8]) / (maxArray[8] - minArray[8]));
            }

            // create Label
            KospiData stock = stockDataList.get(i + exampleLength);
//            INDArray label =  Nd4j.create(new int[] {1}, 'f');
//            label.putScalar(new int[]{0}, stock.getIndexValue());

            INDArray label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');
            label.putScalar(new int[]{0}, stock.getIndexValue());
            label.putScalar(new int[]{1}, stock.getTotalEa());
            label.putScalar(new int[]{2}, stock.getTotalVolume());
            label.putScalar(new int[]{3}, stock.getAnt());
            label.putScalar(new int[]{4}, stock.getForeigner());
            label.putScalar(new int[]{5}, stock.getCompany());
            label.putScalar(new int[]{6}, stock.getInvestBank());
            label.putScalar(new int[]{7}, stock.getInvestTrust());
            label.putScalar(new int[]{8}, stock.getPensionFund());

            test.add(new Pair<>(input, label));
        }
        return test ;
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
                // Only Single Label (= Closed Value)
                kospiDataList.add(new KospiData(
                        arr[0],nums[0],nums[1],nums[2],nums[3],nums[4],nums[5],nums[6],nums[7],nums[8]
                ));

                // Full Load
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return kospiDataList;
    }


    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();    // private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());    // all - (22+1)

        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f'); // _,9,22
        INDArray label = Nd4j.create(new int[] {actualMiniBatchSize, predictLength, exampleLength}, 'f'); // _,1,22

        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
//            System.out.println("----------start=" + startIdx + ", endIdx=" + endIdx);
            KospiData curData = train.get(startIdx);    // get first - end block list  |start - [window] - end|

            KospiData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;
                input.putScalar(new int[]{index, 0, c}, (curData.getIndexValue() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[]{index, 1, c}, (curData.getTotalEa() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[]{index, 2, c}, (curData.getTotalVolume() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[]{index, 3, c}, (curData.getAnt() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[]{index, 4, c}, (curData.getForeigner() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[]{index, 5, c}, (curData.getCompany() - minArray[5]) / (maxArray[5] - minArray[5]));
                input.putScalar(new int[]{index, 6, c}, (curData.getInvestBank() - minArray[6]) / (maxArray[6] - minArray[6]));
                input.putScalar(new int[]{index, 7, c}, (curData.getInvestTrust() - minArray[7]) / (maxArray[7] - minArray[7]));
                input.putScalar(new int[]{index, 8, c}, (curData.getPensionFund() - minArray[8]) / (maxArray[8] - minArray[8]));

                nextData = train.get(i + 1);

                // only single label
                //label.putScalar(new int[]{index, 0, c}, (nextData.getIndexValue() - minArray[0]) / (maxArray[0] - minArray[0]));
                // --

                label.putScalar(new int[]{index, 0, c}, (nextData.getIndexValue() - minArray[0]) / (maxArray[0] - minArray[0]));
                label.putScalar(new int[]{index, 1, c}, (nextData.getTotalEa() - minArray[1]) / (maxArray[1] - minArray[1]));
                label.putScalar(new int[]{index, 2, c}, (nextData.getTotalVolume() - minArray[2]) / (maxArray[2] - minArray[2]));
                label.putScalar(new int[]{index, 3, c}, (nextData.getAnt() - minArray[3]) / (maxArray[3] - minArray[3]));
                label.putScalar(new int[]{index, 4, c}, (nextData.getForeigner() - minArray[4]) / (maxArray[4] - minArray[4]));
                label.putScalar(new int[]{index, 5, c}, (nextData.getCompany() - minArray[5]) / (maxArray[5] - minArray[5]));
                label.putScalar(new int[]{index, 6, c}, (nextData.getInvestBank() - minArray[6]) / (maxArray[6] - minArray[6]));
                label.putScalar(new int[]{index, 7, c}, (nextData.getInvestTrust() - minArray[7]) / (maxArray[7] - minArray[7]));
                label.putScalar(new int[]{index, 8, c}, (nextData.getPensionFund() - minArray[8]) / (maxArray[8] - minArray[8]));

                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }

    public double[] getMaxArray() { return maxArray; }

    public double[] getMinArray() { return minArray; }

    @Override
    public int inputColumns() {
        return VECTOR_SIZE;
    }

    @Override
    public int totalOutcomes() {
        return VECTOR_SIZE ;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        initializeOffsets();
    }

    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + predictLength; // 22 + 1
        for (int i = 0; i < train.size() - window; i++) { exampleStartOffsets.add(i); }
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }


    public static void main(String ... v) {
        System.out.println("Test Module");
//        KospIvstDataSetIter test = new KospIvstDataSetIter();
//        test.readKospiDataFromFile("datafile/kospi_with_investor.csv").forEach(System.out::println);
    }
}
