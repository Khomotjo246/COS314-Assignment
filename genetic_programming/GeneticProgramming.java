import java.util.*;

public class GeneticProgramming {
    static final int POP_SIZE = 50, MAX_GEN = 50, TOURNAMENT_SIZE = 3;
    static final double MUTATION_RATE = 0.2, CROSSOVER_RATE = 0.8;
    public Individual bestIndividual;


    Random rand = new Random(1000);
    List<Individual> population = new ArrayList<>();

    public GeneticProgramming(String csvPath) {
        DataLoader.load(csvPath);
    }

    public void run() {
        initPopulation();
        for (int gen = 0; gen < MAX_GEN; gen++) {
            for (Individual ind : population) ind.evaluateFitness();
            population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
            System.out.printf("Gen %d: Best Fitness = %.4f%n", gen, population.get(0).fitness);
            bestIndividual = population.get(0);

            List<Individual> newPop = new ArrayList<>();
            newPop.add(bestIndividual); // Elitism

            while (newPop.size() < POP_SIZE) {
                Individual parent1 = tournamentSelection();
                Individual parent2 = tournamentSelection();
                Individual child;

                if (rand.nextDouble() < CROSSOVER_RATE) {
                    child = crossover(parent1, parent2);
                } else {
                    child = parent1.clone();
                }

                if (rand.nextDouble() < MUTATION_RATE) {
                    mutate(child.tree);
                }

                newPop.add(child);
            }

            population = newPop;
        }

        System.out.println("Best Individual: " + bestIndividual.tree);
    }

    public int predict(double[] input) {
        double val = bestIndividual.tree.evaluate(input);
        return val > 0 ? 1 : 0;
    }


    private void initPopulation() {
        for (int i = 0; i < POP_SIZE; i++) {
            Node tree = generateRandomTree(3);
            population.add(new Individual(tree));
        }
    }

    private Node generateRandomTree(int depth) {
        if (depth == 0) return new TerminalNode(rand.nextInt(5));
        String[] ops = {"+", "-", "*", "/"};
        return new FunctionNode(ops[rand.nextInt(4)],
                generateRandomTree(depth - 1),
                generateRandomTree(depth - 1));
    }

    private Individual tournamentSelection() {
        Individual best = null;
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            Individual candidate = population.get(rand.nextInt(POP_SIZE));
            if (best == null || candidate.fitness > best.fitness) best = candidate;
        }
        return best;
    }

    private Individual crossover(Individual p1, Individual p2) {
        return new Individual(p1.tree.clone()); // Simplified: could implement real subtree crossover
    }

    private void mutate(Node node) {
        // Simple mutation: replace a subtree
        if (rand.nextDouble() < 0.3 && node instanceof FunctionNode) {
            FunctionNode fn = (FunctionNode) node;
            fn.left = generateRandomTree(2);
        }
    }
}
