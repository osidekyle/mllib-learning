import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

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
            return new LabeledPoint(map.get(parts[parts.length-1]),Vectors.dense(v));

        });

        MultivariateStatisticalSummary summary =
                Statistics.colStats( inputData.rdd());
        System.out.println("Summary mean:");
        System.out.println(summary.mean());
        System.out.println("Summary variance:");
        System.out.println(summary.variance());
        System.out.println("Summary Non-zero");
        System.out.println(summary.numNonzeros());


        Matrix correlMatrix = Statistics.corr(inputData.rdd(), "pearson");
        System.out.println("Correlation Matrix:");
        System.out.println(correlMatrix.toString());


        JavaRDD<LabeledPoint>[] splits = labeledData.randomSplit(new double[] { 0.8, 0.2}, 11l);
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(3)
                .run(trainingData.rdd());


        JavaPairRDD<Object, Object> predictionAndLabels = testData
                .mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double accuracy = metrics.accuracy();
        System.out.println("Model Accuracy on Test Data: " + accuracy);
        predictionAndLabels.collect().forEach(value -> System.out.println(value._1 + " | " + value._2));


        model.save(JavaSparkContext.toSparkContext(sc), "model\\logistic-regression");
        LogisticRegressionModel sameModel = LogisticRegressionModel.load(JavaSparkContext.toSparkContext(sc), "model\\logistic-regression");
        Vector newData = Vectors.dense(new double[]{1, 1, 1, 1});
        double prediction = sameModel.predict(newData);
        System.out.println("Model Prediction on New Data = " + prediction);





    }
}
