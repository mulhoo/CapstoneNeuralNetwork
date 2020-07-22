package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.examples.utils.PlotUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class WineClassifier {
    public static String dataLocalPath;
    public static boolean visualize = true;

    public static void main(String[] args) throws Exception {
        int BATCH_SIZE = 130;
        int SEED = 123;

        //LEARNING RATE
        Scanner learn_rate_user_input = new Scanner(System.in);
        System.out.println("Learning Rate: ");
        double LEARNING_RATE = learn_rate_user_input.nextDouble(); //Epsilon

        //MOMENTUM
        Scanner momentum_user_input = new Scanner(System.in);
        System.out.println("Momentum: ");
        double MOMENTUM = momentum_user_input.nextDouble(); //Alpha

        //HIDDEN NEURONS
        Scanner hidden_neurons_user_input = new Scanner(System.in);
        System.out.println("Hidden Neurons: ");
        int HIDDEN_NEURONS = hidden_neurons_user_input.nextInt(); //Alpha
        hidden_neurons_user_input.close();

        int INPUT_NEURONS = 13;
        int OUTPUT_NEURONS = 3;

        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath, "wine_training.csv")));
        DataSetIterator TRAIN_DATA = new RecordReaderDataSetIterator(rr, BATCH_SIZE, 0, 3);

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(dataLocalPath, "wine_eval.csv")));
        DataSetIterator TEST_DATA = new RecordReaderDataSetIterator(rrTest, BATCH_SIZE, 0, 3);

        DataSet TRAIN_DATA_SET = TRAIN_DATA.next();
        TRAIN_DATA_SET.shuffle();
        DataSet TEST_DATA_SET = TEST_DATA.next();
        TEST_DATA_SET.shuffle();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(TRAIN_DATA_SET);
        normalizer.transform(TRAIN_DATA_SET);
        normalizer.transform(TEST_DATA_SET);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(LEARNING_RATE, MOMENTUM))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(INPUT_NEURONS)
                .nOut(HIDDEN_NEURONS)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(HIDDEN_NEURONS)
                .nOut(OUTPUT_NEURONS)
                .activation(Activation.SOFTMAX)
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for(int i=0; i<1000; i++ ) {
            model.fit(TRAIN_DATA_SET);
        }
        System.out.println("Test model....");
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(TEST_DATA_SET.getFeatures());
        eval.eval(TEST_DATA_SET.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println("\n****************COMPLETE********************");

//        generateVisuals(model, TRAIN_DATA_SET, TEST_DATA_SET);
    }

    //visualization code from DL4J website
    public static void generateVisuals(MultiLayerNetwork model, DataSet TRAIN_DATA_SET, DataSet TEST_DATA_SET) throws Exception {
        if (visualize) {
            double xMin = -1.;
            double xMax = 1;
            double yMin = -1;
            double yMax = 1;

            int nPointsPerAxis = 100;

            INDArray allXYPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);

            PlotUtil.plotTrainingData(model, TRAIN_DATA_SET, allXYPoints, nPointsPerAxis);
            TimeUnit.SECONDS.sleep(3);

            PlotUtil.plotTestData(model, TEST_DATA_SET, allXYPoints, nPointsPerAxis);
        }
    }

}
