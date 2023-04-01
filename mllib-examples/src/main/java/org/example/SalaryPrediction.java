package org.example;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SalaryPrediction {
    public static void predictSalary(){
        SparkSession sparkSession = SparkSession.builder()
                .appName("spark-mllib-linear-regression")
                .master("local")
                .getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/static/SalaryData.csv");

        Dataset<Row> newData = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/static/test.csv");

        VectorAssembler featuresVector = new VectorAssembler().setInputCols(new String[] {"YearsExperience"})
                .setOutputCol("features");

        Dataset<Row> transform = featuresVector.transform(rawData);
        Dataset<Row> newTransform = featuresVector.transform(newData);

        Dataset<Row> finalData = transform.select("features","Salary");
        Dataset<Row>[] dataset = finalData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainData = dataset[0];
        Dataset<Row> testData = dataset[1];

        LinearRegression lr = new LinearRegression();
        lr.setLabelCol("Salary");
        LinearRegressionModel lrModel = lr.fit(trainData);

        // test
        // Dataset<Row> transformTest = lrModel.transform(testData);
        Dataset<Row> transformTest = lrModel.transform(newTransform);
        transformTest.show();

        // model evaluation
        // LinearRegressionTrainingSummary modelSummary = lrModel.summary();
        // System.out.println(modelSummary.r2());
    }
}
