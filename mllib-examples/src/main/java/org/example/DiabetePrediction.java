package org.example;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DiabetePrediction {
    public static void predictDiabete(){
        SparkSession sparkSession = SparkSession.builder().appName("spark-mllib-diabetes")
                                                            .master("local").getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/static/diabetes.csv");

        String[] headerList = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"};

        List<String> headers = Arrays.asList(headerList);
        List<String> headersCat = new ArrayList<String>();

        for (String header: headers){
            if (header.equals("Outcome")){
                StringIndexer indexTemp = new StringIndexer().setInputCol(header).setOutputCol("label");
                rawData = indexTemp.fit(rawData).transform(rawData);
                headersCat.add("label");
            }
            else {
                StringIndexer indexTemp = new StringIndexer().setInputCol(header).setOutputCol(header + "Cat");
                rawData = indexTemp.fit(rawData).transform(rawData);
                headersCat.add(header + "Cat");
            }
        }

        String[] colList = headersCat.toArray(new String[headersCat.size()]);
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(colList).setOutputCol("features");

        Dataset<Row> transformData = vectorAssembler.transform(rawData);
        Dataset<Row> finalData = transformData.select("label", "features");
        Dataset<Row>[] dataset = finalData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainData = dataset[0];
        Dataset<Row> testData = dataset[1];

        NaiveBayes nb = new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel naiveBayesModel = nb.fit(trainData);

        Dataset<Row> predictions = naiveBayesModel.transform(testData);

        predictions.show();

        // model evaluation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double evaluate = evaluator.evaluate(predictions);
        System.out.println("Accuracy = "+ evaluate);

    }
}
