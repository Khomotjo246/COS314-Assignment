How to run:

1. Genetic programming
cd genetic_programming
run 'make'

2. Multi-layer perception
Dependencies:
Numpy: pip install numpy
Pandas: pip install Pandas
Openpyxl:  pip install openpyxl

To run:
cd mlp
run 'py mlp_model.py'

3. Decision tree
Dependecies:
intsall weka https://www.cs.waikato.ac.nz/ml/weka/

Compilation Instructions:
javac -cp decision_tree/weka.jar decision_tree/new\ one/j48_runner.java

Execution Instructions:
java --add-opens java.base/java.lang=ALL-UNNAMED -cp 
decision_tree/weka.jar:decision_tree/new\ one j48_runner