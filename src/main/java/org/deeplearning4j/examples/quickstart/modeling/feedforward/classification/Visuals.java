package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.examples.utils.PlotUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.concurrent.TimeUnit;

public class Visuals {
    public static String dataLocalPath;
    public static boolean visualize = true;

    public static void main(String[] args) throws Exception {
        int BATCH_SIZE = 97;
        int SEED = 50;

        double LEARNING_RATE = .5; //Epsilon
        double MOMENTUM = .9; //Alpha


        int HIDDEN_NEURONS = 6;
        int INPUT_NEURONS = 2;
        int OUTPUT_NEURONS = 2;

        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(dataLocalPath, "wine_train_visuals.csv")));
        DataSetIterator TRAIN_DATA = new RecordReaderDataSetIterator(rr, BATCH_SIZE, 0, 2);

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(dataLocalPath, "wine_eval_visuals.csv")));
        DataSetIterator TEST_DATA = new RecordReaderDataSetIterator(rrTest, BATCH_SIZE, 0, 2);

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
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(HIDDEN_NEURONS)
                        .nOut(OUTPUT_NEURONS)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for(int i=0; i<1000; i++ ) { model.fit(TRAIN_DATA_SET);
        }
        System.out.println("Test model....");
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(TEST_DATA_SET.getFeatures());
        eval.eval(TEST_DATA_SET.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println("\n****************COMPLETE********************");

        generateVisuals(model, TRAIN_DATA_SET, TEST_DATA_SET);
    }

    public static void generateVisuals(MultiLayerNetwork model, DataSet TRAIN_DATA_SET, DataSet TEST_DATA_SET) throws Exception {
        if (visualize) {
            double xMin = -1.;
            double xMax = 1;
            double yMin = -1;
            double yMax = 1;

            int nPointsPerAxis = 100;

            INDArray allXYZPoints = PlotUtil.generatePointsOnGraph(xMin, xMax, yMin, yMax, nPointsPerAxis);

            PlotUtil.plotTrainingData(model, TRAIN_DATA_SET, allXYZPoints, nPointsPerAxis);
            TimeUnit.SECONDS.sleep(3);

            PlotUtil.plotTestData(model, TEST_DATA_SET, allXYZPoints, nPointsPerAxis);
        }
    }
}
