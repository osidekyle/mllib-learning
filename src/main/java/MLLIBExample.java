import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

import java.util.HashMap;
import java.util.Map;

public class MLLIBExample {





    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("Main")
                .setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(conf);


        String dataFile = "data\\iris.data";
        JavaRDD<String> data = sc.textFile(dataFile);

        JavaRDD<Vector> inputData = data
                .map(line -> {
                    String[] parts = line.split(",");
                    double[] v = new double[parts.length - 1];
                    for (int i = 0; i < parts.length - 1; i++) {
                        v[i] = Double.parseDouble(parts[i]);
                    }
                    return Vectors.dense(v);
                });

        Map<String, Integer> map = new HashMap<>();
        map.put("Iris-setosa", 0);
        map.put("Iris-versicolor", 1);
        map.put("Iris-virginica", 2);

        JavaRDD<LabeledPoint> labeledData = data
        .map(line -> {
            String[] parts = line.split(",");
            double[] v = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                v[i] = Double.parseDouble(parts[i]);
            }
            return new LabeledPoint(map.get(parts.length-1),Vectors.dense(v));

        });



    }
}
