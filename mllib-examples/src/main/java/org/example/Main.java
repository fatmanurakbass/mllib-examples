package org.example;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Main {
    public static void main(String[] args) {
        // SalaryPrediction.predictSalary();
        // IdealWeatherClassification.classifyIdealWeather();
        DiabetePrediction.predictDiabete();

    }
}