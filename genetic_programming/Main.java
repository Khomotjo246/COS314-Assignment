import java.util.List;

public class Main {
    public static void main(String[] args) {
        String trainingData = "BTC_train.csv";           // must include labels
        String testData = "BTC_test.csv";          // must also include labels in 6th column

        GeneticProgramming gp = new GeneticProgramming(trainingData);
        gp.run();

        // Load test data with labels
        List<double[]> testInputs = new java.util.ArrayList<>();
        List<Integer> testLabels = new java.util.ArrayList<>();

        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(testData))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // skip header
                }
                String[] tokens = line.split(",");
                double[] features = new double[5];
                for (int i = 0; i < 5; i++) features[i] = Double.parseDouble(tokens[i]);
                int label = Integer.parseInt(tokens[5]);
                testInputs.add(features);
                testLabels.add(label);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // Evaluate accuracy
        int correct = 0;
        for (int i = 0; i < testInputs.size(); i++) {
            int prediction = gp.predict(testInputs.get(i));
            if (prediction == testLabels.get(i)) {
                correct++;
            }
        }

        double accuracy = (double) correct / testInputs.size() * 100.0;
        System.out.printf("Prediction Accuracy: %.2f%% (%d/%d correct)%n", accuracy, correct, testInputs.size());
    }
}
