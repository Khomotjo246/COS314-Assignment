all:
	make clean && make main && java Main ${task}

main:
	javac *.java

clean:
	rm *.o main -f
	rm *.gcda main -f
	rm *.gcov.json.gz main -f
	rm *.gcno main -f
	rm *.zip -f
	rm -f *.class
	clear
	
zip:
	zip -r u23538318.zip *.java *.txt makefile

txtFile:
	make all -s > ../Marker/task${task}.txt

coverage:
	rm -r org*/ jacoco-resources default com*/ -f
	rm *.class *.html *.exec *.xml *.csv -f
	javac *.java
	rm -Rf cov
	mkdir ./cov
	java -javaagent:jacocoagent.jar=excludes=org.jacoco.*,destfile=./cov/output.exec -cp ./ ACO
	mv *.class ./cov
	java -jar ./jacococli.jar report ./cov/output.exec --classfiles ./cov --sourcefiles ./  --html ./cov/report