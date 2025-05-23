import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Utils;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import javax.swing.JFrame;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Discretize;

public class j48_runner {
    
    public static void main(String[] args) {
        try {
            String trainPath = "Data/BTC_train.csv";
            String testPath = "Data/BTC_test.csv";
            int seed = 1000; // Seed for cross-validation
            
            System.out.println("\n=== Loading Training Data ===");
            Instances trainData = loadCSVData(trainPath);
            System.out.println("Training data loaded: " + trainData.numInstances() + " instances, " + 
                             trainData.numAttributes() + " attributes");
            
            System.out.println("\n=== Loading Test Data ===");
            Instances testData = loadCSVData(testPath);
            System.out.println("Test data loaded: " + testData.numInstances() + " instances, " + 
                             testData.numAttributes() + " attributes");
            
            // Print attribute information
            System.out.println("\n=== Attribute Information ===");
            System.out.println("Training data attributes:");
            for (int i = 0; i < trainData.numAttributes(); i++) {
                System.out.println("Attribute " + (i + 1) + ": " + trainData.attribute(i).name() + 
                                 " (Type: " + trainData.attribute(i).typeToString(trainData.attribute(i)) + ")");
                if (trainData.attribute(i).isNumeric()) {
                    double[] values = trainData.attributeToDoubleArray(i);
                    double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
                    for (double val : values) {
                        if (!Double.isNaN(val)) {
                            min = Math.min(min, val);
                            max = Math.max(max, val);
                        }
                    }
                    System.out.println("  Range: [" + min + ", " + max + "]");
                }
            }
            
            System.out.println("\n=== Missing Values Check ===");
            int missingCount = 0;
            for (int i = 0; i < trainData.numInstances(); i++) {
                for (int j = 0; j < trainData.numAttributes(); j++) {
                    if (trainData.instance(i).isMissing(j)) {
                        missingCount++;
                    }
                }
            }
            System.out.println("Missing values in training data: " + missingCount);
            
            // Set class attribute (last attribute)
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);
            
            // Convert class attribute to nominal
            trainData = convertClassToNominal(trainData);
            testData = convertClassToNominal(testData);
            
            // Discretize numeric attributes 
            System.out.println("\n=== Discretizing Numeric Attributes ===");
            Discretize discretize = new Discretize();
            discretize.setAttributeIndices("1-" + (trainData.numAttributes() - 1)); 
            discretize.setBins(10); 
            discretize.setInputFormat(trainData);
            trainData = weka.filters.Filter.useFilter(trainData, discretize);
            testData = weka.filters.Filter.useFilter(testData, discretize);
            
            // Print class distribution 
            System.out.println("\n=== Class Distribution Analysis ===");
            System.out.println("Class distribution in training data:");
            for (int i = 0; i < trainData.numClasses(); i++) {
                System.out.println("Class " + trainData.classAttribute().value(i) + ": " +
                                 trainData.attributeStats(trainData.classIndex()).nominalCounts[i] + " instances");
            }
            
            // Print correlation analysis
            System.out.println("\n=== Feature Correlation Analysis ===");
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                for (int j = i + 1; j < trainData.numAttributes() - 1; j++) {
                    if (trainData.attribute(i).isNumeric() && trainData.attribute(j).isNumeric()) {
                        double[] attr1 = trainData.attributeToDoubleArray(i);
                        double[] attr2 = trainData.attributeToDoubleArray(j);
                        double corr = calculateCorrelation(attr1, attr2);
                        System.out.printf("Correlation between %s and %s: %.4f%n", 
                                        trainData.attribute(i).name(), 
                                        trainData.attribute(j).name(), 
                                        corr);
                    }
                }
            }
            
            // configure J48 classifier
            System.out.println("\n=== Configuring J48 Decision Tree ===");
            J48 j48 = new J48();
            
            // Set J48 parameters
            String[] options = {
                // "-U" // Unpruned tree 
                "-C", "0.25", // Confidence factor for pruning 
                "-M", "2" 
            };
            j48.setOptions(options);
            
            System.out.println("J48 Configuration:");
            System.out.println("- Unpruned Tree: Enabled");
            
            // Train the classifier
            System.out.println("\n=== Training J48 Decision Tree ===");
            long startTime = System.currentTimeMillis();
            j48.buildClassifier(trainData);
            long endTime = System.currentTimeMillis();
            
            System.out.println("Training completed in " + (endTime - startTime) + " ms");
            System.out.println("Number of leaves: " + j48.measureNumLeaves());
            System.out.println("Size of tree: " + j48.measureTreeSize());
            
            // Display the decision tree structure
            System.out.println("\n=== Decision Tree Structure ===");
            System.out.println(j48.toString());
            
            // Save tree
            saveTreeToFile(j48.toString(), "decision_tree_structure.txt");
            
            // Evaluate on training data
            System.out.println("\n=== Training Set Evaluation ===");
            Evaluation evalTrain = new Evaluation(trainData);
            evalTrain.evaluateModel(j48, trainData);
            
            System.out.println("Training Accuracy: " + String.format("%.4f", evalTrain.pctCorrect() / 100.0));
            System.out.println("Training F1-Score: " + String.format("%.4f", evalTrain.fMeasure(1)));
            System.out.println("Training Precision: " + String.format("%.4f", evalTrain.precision(1)));
            System.out.println("Training Recall: " + String.format("%.4f", evalTrain.recall(1)));
            
            // Evaluate on test data
            System.out.println("\n=== Test Set Evaluation ===");
            Evaluation evalTest = new Evaluation(trainData);
            evalTest.evaluateModel(j48, testData);
            
            System.out.println("Test Accuracy: " + String.format("%.4f", evalTest.pctCorrect() / 100.0));
            System.out.println("Test F1-Score: " + String.format("%.4f", evalTest.fMeasure(1)));
            System.out.println("Test Precision: " + String.format("%.4f", evalTest.precision(1)));
            System.out.println("Test Recall: " + String.format("%.4f", evalTest.recall(1)));
            
            // Detailed evaluation results
            System.out.println("\n=== Detailed Test Results ===");
            System.out.println(evalTest.toSummaryString());
            System.out.println("\n=== Confusion Matrix ===");
            System.out.println(evalTest.toMatrixString());
            
            // Cross-validation on training data
            System.out.println("\n=== 10-Fold Cross-Validation on Training Data ===");
            Evaluation evalCV = new Evaluation(trainData);
            evalCV.crossValidateModel(j48, trainData, 10, new Random(seed));
            
            System.out.println("CV Accuracy: " + String.format("%.4f", evalCV.pctCorrect() / 100.0));
            System.out.println("CV F1-Score: " + String.format("%.4f", evalCV.fMeasure(1)));
            System.out.println("CV Standard Deviation: " + String.format("%.4f", evalCV.rootMeanSquaredError()));
            
            // Display results summary table
            displayResultsTable(evalTrain, evalTest, evalCV);
            
            // Save detailed results
            saveDetailedResults(j48, evalTrain, evalTest, evalCV, seed);
            
            // Show tree visualization 
            System.out.println("\n=== Tree Visualization ===");
            System.out.print("Would you like to display the tree visualization? (y/n): ");
            Scanner scanner = new Scanner(System.in);
            String showViz = scanner.nextLine();
            
            if (showViz.toLowerCase().startsWith("y")) {
                displayTreeVisualization(j48, trainData);
            }
            
            // Demonstrate predictions on a few test instances
            System.out.println("\n=== Sample Predictions ===");
            demonstratePredictions(j48, testData, 5);
            
            System.out.println("\n=== J48 Decision Tree Analysis Complete ===");
            System.out.println("Files generated:");
            System.out.println("- decision_tree_structure.txt");
            System.out.println("- j48_detailed_results.txt");
            
            scanner.close();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Load CSV data using Weka's CSVLoader
     */
    private static Instances loadCSVData(String filePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
        
        if (data == null) {
            throw new Exception("Could not load data from: " + filePath);
        }
        
        return data;
    }
    
    /**
     * Save decision tree structure to file
     */
    private static void saveTreeToFile(String treeString, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("=== J48 Decision Tree Structure ===\n\n");
            writer.write(treeString);
            System.out.println("Decision tree structure saved to: " + filename);
        } catch (IOException e) {
            System.err.println("Error saving tree structure: " + e.getMessage());
        }
    }
    
    /**
     * Convert class attribute to nominal type
     */
    private static Instances convertClassToNominal(Instances data) throws Exception {
        // Check if class attribute is already nominal
        if (data.classAttribute().isNominal()) {
            System.out.println("Class attribute is already nominal.");
            return data;
        }

        System.out.println("Converting class attribute from numeric to nominal...");
        
        // Print some statistics about the numeric class values
        double[] classValues = data.attributeToDoubleArray(data.classIndex());
        java.util.Set<Double> uniqueValues = new java.util.HashSet<>();
        for (double val : classValues) {
            uniqueValues.add(val);
        }
        System.out.println("Unique class values found: " + uniqueValues);
        
        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[]{"-R", Integer.toString(data.classIndex() + 1)};
        convert.setOptions(options);
        convert.setInputFormat(data);
        
        Instances newData = weka.filters.Filter.useFilter(data, convert);
        newData.setClassIndex(data.classIndex());
        
        System.out.println("Class attribute converted to nominal.");
        System.out.println("Class attribute name: " + newData.classAttribute().name());
        System.out.println("Number of class values: " + newData.numClasses());
        for (int i = 0; i < newData.numClasses(); i++) {
            System.out.println("  Class " + i + ": " + newData.classAttribute().value(i));
        }
        
        return newData;
    }
    
    /**
     * Calculate Pearson correlation coefficient between two attributes
     */
    private static double calculateCorrelation(double[] x, double[] y) {
        int n = x.length;
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        int validCount = 0;
        
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(x[i]) && !Double.isNaN(y[i])) {
                sumX += x[i];
                sumY += y[i];
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
                sumY2 += y[i] * y[i];
                validCount++;
            }
        }
        
        if (validCount == 0) return 0.0;
        
        double meanX = sumX / validCount;
        double meanY = sumY / validCount;
        double numerator = sumXY - validCount * meanX * meanY;
        double denominator = Math.sqrt((sumX2 - validCount * meanX * meanX) * 
                                     (sumY2 - validCount * meanY * meanY));
        
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
    
    /**
     * Display results in table format
     */
    private static void displayResultsTable(Evaluation evalTrain, Evaluation evalTest, Evaluation evalCV) {
        System.out.println("\n=== RESULTS SUMMARY TABLE ===");
        System.out.println("┌─────────────────┬──────────┬──────────┬──────────┐");
        System.out.println("│     Dataset     │ Accuracy │ F1-Score │ Precision│");
        System.out.println("├─────────────────┼──────────┼──────────┼──────────┤");
        System.out.printf("│ %-15s │ %8.4f │ %8.4f │ %8.4f │%n", 
                         "Training", 
                         evalTrain.pctCorrect() / 100.0,
                         evalTrain.fMeasure(1),
                         evalTrain.precision(1));
        System.out.printf("│ %-15s │ %8.4f │ %8.4f │ %8.4f │%n", 
                         "Test", 
                         evalTest.pctCorrect() / 100.0,
                         evalTest.fMeasure(1),
                         evalTest.precision(1));
        System.out.printf("│ %-15s │ %8.4f │ %8.4f │ %8.4f │%n", 
                         "Cross-Validation", 
                         evalCV.pctCorrect() / 100.0,
                         evalCV.fMeasure(1),
                         evalCV.precision(1));
        System.out.println("└─────────────────┴──────────┴──────────┴──────────┘");
    }
    
    /**
     * Save detailed results to file
     */
    private static void saveDetailedResults(J48 j48, Evaluation evalTrain, Evaluation evalTest, 
                                          Evaluation evalCV, int seed) {
        try (FileWriter writer = new FileWriter("j48_detailed_results.txt")) {
            writer.write("=== J48 Decision Tree - Detailed Results ===\n\n");
            writer.write("Random Seed (for CV): " + seed + "\n");
            writer.write("Tree Size: " + j48.measureTreeSize() + "\n");
            writer.write("Number of Leaves: " + j48.measureNumLeaves() + "\n\n");
            
            writer.write("=== Training Set Results ===\n");
            writer.write("Accuracy: " + String.format("%.4f", evalTrain.pctCorrect() / 100.0) + "\n");
            writer.write("F1-Score: " + String.format("%.4f", evalTrain.fMeasure(1)) + "\n");
            writer.write("Precision: " + String.format("%.4f", evalTrain.precision(1)) + "\n");
            writer.write("Recall: " + String.format("%.4f", evalTrain.recall(1)) + "\n\n");
            
            writer.write("=== Test Set Results ===\n");
            writer.write("Accuracy: " + String.format("%.4f", evalTest.pctCorrect() / 100.0) + "\n");
            writer.write("F1-Score: " + String.format("%.4f", evalTest.fMeasure(1)) + "\n");
            writer.write("Precision: " + String.format("%.4f", evalTest.precision(1)) + "\n");
            writer.write("Recall: " + String.format("%.4f", evalTest.recall(1)) + "\n\n");
            
            writer.write("=== Cross-Validation Results ===\n");
            writer.write("CV Accuracy: " + String.format("%.4f", evalCV.pctCorrect() / 100.0) + "\n");
            writer.write("CV F1-Score: " + String.format("%.4f", evalCV.fMeasure(1)) + "\n\n");
            
            writer.write("=== Confusion Matrix (Test Set) ===\n");
            try {
                writer.write(evalTest.toMatrixString());
            } catch (Exception e) {
                writer.write("Error generating confusion matrix: " + e.getMessage() + "\n");
            }
            
            System.out.println("Detailed results saved to: j48_detailed_results.txt");
        } catch (IOException e) {
            System.err.println("Error saving detailed results: " + e.getMessage());
        }
    }
    
    /**
     * Display tree visualization in GUI
     */
    private static void displayTreeVisualization(J48 j48, Instances data) {
        try {
            TreeVisualizer tv = new TreeVisualizer(null, j48.graph(), new PlaceNode2());
            JFrame jf = new JFrame("J48 Decision Tree Visualization");
            jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jf.setSize(800, 600);
            jf.getContentPane().add(tv);
            jf.setVisible(true);
            
            System.out.println("Tree visualization window opened.");
            System.out.println("Close the window to continue...");
        } catch (Exception e) {
            System.err.println("Could not display tree visualization: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate predictions on sample instances
     */
    private static void demonstratePredictions(J48 j48, Instances testData, int numSamples) {
        try {
            System.out.println("Showing predictions for first " + numSamples + " test instances:");
            System.out.println("┌─────┬────────────┬──────────┬─────────┐");
            System.out.println("│ No. │  Predicted │  Actual  │ Correct │");
            System.out.println("├─────┼────────────┼──────────┼─────────┤");
            
            int correct = 0;
            for (int i = 0; i < Math.min(numSamples, testData.numInstances()); i++) {
                double predicted = j48.classifyInstance(testData.instance(i));
                double actual = testData.instance(i).classValue();
                boolean isCorrect = (predicted == actual);
                if (isCorrect) correct++;
                
                System.out.printf("│ %3d │ %10s │ %8s │ %7s │%n", 
                                i + 1,
                                testData.classAttribute().value((int)predicted),
                                testData.classAttribute().value((int)actual),
                                isCorrect ? "✓" : "✗");
            }
            System.out.println("└─────┴────────────┴──────────┴─────────┘");
            System.out.println("Sample accuracy: " + correct + "/" + Math.min(numSamples, testData.numInstances()));
            
        } catch (Exception e) {
            System.err.println("Error demonstrating predictions: " + e.getMessage());
        }
    }
}