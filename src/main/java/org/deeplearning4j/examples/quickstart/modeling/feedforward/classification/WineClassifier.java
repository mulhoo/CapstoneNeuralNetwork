package org.deeplearning4j.examples.quickstart.modeling.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utils.DownloaderUtility;
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

public class WineClassifier {
    public static String dataLocalPath;

    public static void main(String[] args) throws Exception {
        int BATCH_SIZE = 130;
        int SEED = 123;

        System.out.println("Welcome to an interactive Neural Network where you can see how changing 4 different values can impact the accuracy of your network. This neural ");
        System.out.println("network classifies wine from three different cultivars in Italy, taking into consideration 13 different attributes of each wine.");
        System.out.println("\nThe 13 attributes are: Alcohol, Malic Acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, ");
        System.out.println("Color intensity, Hue, OD280/OD315 of diluted wines, and Proline.");

        System.out.println("\n\n#############################################################################################################################");
        System.out.println("This data was provided courtesy of the University of California, Irvine, Center for Machine Learning and Intelligent Systems.");
        System.out.println("#############################################################################################################################");


        //LEARNING RATE
        System.out.println("\n\nThe Learning Rate controls how much to change the model in response to the estimated error each time the model weights are updated. " +
                "The value for");
        System.out.println("learning rate should be a positive decimal value between 0 and 1.");
        Scanner learn_rate_user_input = new Scanner(System.in);
        System.out.print("Learning Rate: ");
        double LEARNING_RATE = learn_rate_user_input.nextDouble(); //Epsilon

        //MOMENTUM
        System.out.println("\nThe basic idea of momentum in machine learning is to increase the speed of training. " +
                "The value for momentum should be a positive decimal value between 0 and 1.");
        Scanner momentum_user_input = new Scanner(System.in);
        System.out.print("Momentum: ");
        double MOMENTUM = momentum_user_input.nextDouble(); //Alpha

        //HIDDEN NEURONS
        System.out.println("\nAn easy rule of thumb for how many input neurons your network should have is to have the number be somewhere between" +
                "the number of input neurons");
        System.out.println("and the number of output neurons, in this case some number between 3 and 13.");
        System.out.println("However, the number of hidden neurons can be more or less than either of those values. Only requirement is that it must be a positive integer. ");
        Scanner hidden_neurons_user_input = new Scanner(System.in);
        System.out.print("Hidden Neurons: ");
        int HIDDEN_NEURONS = hidden_neurons_user_input.nextInt();

        //ITERATIONS
        System.out.println("\nIterations is the number of batches needed to complete one epoch.");
        System.out.println("One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.");
        Scanner iterations_user_input = new Scanner(System.in);
        System.out.print("Iterations: ");
        int ITERATIONS = iterations_user_input.nextInt();
        iterations_user_input.close();

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

        //NUMBER OF ITERATIONS

        for(int i=0; i<ITERATIONS; i++ ) { model.fit(TRAIN_DATA_SET);
        }
        System.out.println("Test model....");
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(TEST_DATA_SET.getFeatures());
        eval.eval(TEST_DATA_SET.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println("\n****************COMPLETE********************");
    }

}
