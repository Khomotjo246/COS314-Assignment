import java.util.Random;

public class FunctionNode extends Node {
    public String operator;
    public Node left, right;

    public FunctionNode(String operator, Node left, Node right) {
        this.operator = operator;
        this.left = left;
        this.right = right;
    }

    @Override
    public double evaluate(double[] input) {
        double a = left.evaluate(input);
        double b = right.evaluate(input);
        switch (operator) {
            case "+": return a + b;
            case "-": return a - b;
            case "*": return a * b;
            case "/": return b == 0 ? 1 : a / b;
            default: return 0;
        }
    }

    @Override
    public Node clone() {
        return new FunctionNode(operator, left.clone(), right.clone());
    }

    @Override
    public String toString() {
        return "(" + left.toString() + " " + operator + " " + right.toString() + ")";
    }
}
