import java.io.*;
import java.util.*;

public class DataLoader {
    public static List<double[]> inputs = new ArrayList<>();
    public static List<Integer> outputs = new ArrayList<>();

    public static void load(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // skip header
                }
                String[] tokens = line.split(",");
                double[] features = new double[5];
                for (int i = 0; i < 5; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }
                int label = Integer.parseInt(tokens[5]);
                inputs.add(features);
                outputs.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static List<double[]> loadUnseen(String path) {
        List<double[]> unseen = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // skip header
                }
                String[] tokens = line.split(",");
                double[] features = new double[5];
                for (int i = 0; i < 5; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }
                unseen.add(features);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return unseen;
    }

}
