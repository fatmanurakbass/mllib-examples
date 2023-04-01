package org.example;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class IdealWeatherClassification {
    public static void classifyIdealWeather(){
        SparkSession sparkSession = SparkSession.builder()
                .appName("spark-mllib-naive-bayes")
                .master("local")
                .getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/static/weatherForecast.csv");

        StringIndexer indexOutlook = new StringIndexer().setInputCol("Outlook").setOutputCol("OutlookCat");
        StringIndexer indexTemperature = new StringIndexer().setInputCol("Temperature").setOutputCol("TemperatureCat");
        StringIndexer indexHumidity = new StringIndexer().setInputCol("Humidity").setOutputCol("HumidityCat");
        StringIndexer indexWindy = new StringIndexer().setInputCol("Windy").setOutputCol("WindyCat");
        StringIndexer indexLabel = new StringIndexer().setInputCol("Play").setOutputCol("label");

        Dataset<Row> tranformOutlook = indexOutlook.fit(rawData).transform(rawData);
        Dataset<Row> tranformTemperature = indexTemperature.fit(tranformOutlook).transform(tranformOutlook);
        Dataset<Row> tranformHumidity = indexHumidity.fit(tranformTemperature).transform(tranformTemperature);
        Dataset<Row> tranformWindy = indexWindy.fit(tranformHumidity).transform(tranformHumidity);
        Dataset<Row> tranformResult = indexLabel.fit(tranformWindy).transform(tranformWindy);

        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(new String[]
                                                                            {"OutlookCat",
                                                                            "TemperatureCat",
                                                                            "HumidityCat",
                                                                            "WindyCat", "label"})
                                                                .setOutputCol("features");

        Dataset<Row> tranform = vectorAssembler.transform(tranformResult);
        Dataset<Row> finalData = tranform.select("label", "features");

        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7,0.3});
        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData = datasets[1];

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
