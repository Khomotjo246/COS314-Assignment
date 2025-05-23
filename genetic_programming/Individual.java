public class Individual {
    public Node tree;
    public double fitness;

    public Individual(Node tree) {
        this.tree = tree;
    }

    public void evaluateFitness() {
        int correct = 0;
        for (int i = 0; i < DataLoader.inputs.size(); i++) {
            double[] input = DataLoader.inputs.get(i);
            int expected = DataLoader.outputs.get(i);
            double val = tree.evaluate(input);
            int prediction = val > 0 ? 1 : 0;
            if (prediction == expected) correct++;
        }
        fitness = (double) correct / DataLoader.inputs.size();
    }

    public Individual clone() {
        return new Individual(tree.clone());
    }
}
