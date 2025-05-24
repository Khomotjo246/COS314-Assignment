import java.util.*;
import java.io.*;

public class Main {
    public static void main(String[] args) {
        String trainingData = "BTC_train.csv";
        String testData = "BTC_test.csv";

        // Train on training data
        GeneticProgramming gp = new GeneticProgramming(trainingData);
        gp.run();

        // === Evaluate on Training Data ===
        EvaluationResult trainEval = evaluate(gp, DataLoader.inputs, DataLoader.outputs);
        System.out.printf("Training Accuracy: %.2f%% | F1 Score: %.4f%n",
                trainEval.accuracy * 100, trainEval.f1Score);

        // === Load and Evaluate on Test Data ===
        List<double[]> testInputs = new ArrayList<>();
        List<Integer> testLabels = new ArrayList<>();
        loadCSV(testData, testInputs, testLabels);

        EvaluationResult testEval = evaluate(gp, testInputs, testLabels);
        System.out.printf("Test Accuracy: %.2f%% | F1 Score: %.4f%n",
                testEval.accuracy * 100, testEval.f1Score);
    }

    static void loadCSV(String filePath, List<double[]> inputs, List<Integer> outputs) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // skip header
                }
                String[] tokens = line.split(",");
                double[] input = new double[5];
                for (int i = 0; i < 5; i++) {
                    input[i] = Double.parseDouble(tokens[i]);
                }
                int output = Integer.parseInt(tokens[5]);
                inputs.add(input);
                outputs.add(output);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static EvaluationResult evaluate(GeneticProgramming gp, List<double[]> inputs, List<Integer> labels) {
        int correct = 0, tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < inputs.size(); i++) {
            int actual = labels.get(i);
            int predicted = gp.predict(inputs.get(i));

            if (predicted == actual) correct++;
            if (predicted == 1 && actual == 1) tp++;
            if (predicted == 1 && actual == 0) fp++;
            if (predicted == 0 && actual == 1) fn++;
        }

        double accuracy = (double) correct / inputs.size();
        double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
        double recall = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
        double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

        return new EvaluationResult(accuracy, f1);
    }
}

// Simple container for evaluation results
class EvaluationResult {
    double accuracy;
    double f1Score;

    EvaluationResult(double accuracy, double f1Score) {
        this.accuracy = accuracy;
        this.f1Score = f1Score;
    }
}
