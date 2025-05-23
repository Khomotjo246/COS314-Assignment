public class TerminalNode extends Node {
    public int index; // 0 to 4 for Open to Adj Close

    public TerminalNode(int index) {
        this.index = index;
    }

    @Override
    public double evaluate(double[] input) {
        return input[index];
    }

    @Override
    public Node clone() {
        return new TerminalNode(index);
    }

    @Override
    public String toString() {
        return "x" + index;
    }
}
