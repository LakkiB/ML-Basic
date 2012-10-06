package cs475.classify.simpleclassifier;

import cs475.classify.Predictor;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;
import cs475.dataobject.label.RegressionLabel;

import java.text.MessageFormat;
import java.util.List;

public class EvenOddClassifier extends Predictor {
    @Override
    public void train(List<Instance> instances) {
        // no-op

        /*for (Instance instance : instances)
            computeEvenOddSums(instance);
        System.out.println(MessageFormat.format("Training EvenOddClassifier complete. evenSum = {0}, oddSum = {1}", evenSum, oddSum));*/
    }

    private void computeEvenOddSums(Instance instance) {
        for (int key : instance.getFeatureVector().getFeatureVectorKeys()) {
           // System.out.println(MessageFormat.format("computeEvenOddSums::Key={0} value = {1}", key, instance.getFeatureVector().get(key)));
            if (key % 2 == 0)
                evenSum += instance.getFeatureVector().get(key);
            else
                oddSum += instance.getFeatureVector().get(key);
        }
    }

    @Override
    public Label predict(Instance instance) {

        evenSum = 0; oddSum =0;
        computeEvenOddSums(instance);
        System.out.println(MessageFormat.format("Training EvenOddClassifier complete. evenSum = {0}, oddSum = {1}", evenSum, oddSum));

        if (instance.getLabel() instanceof RegressionLabel) {
            createNewRegressionLabel();
        } else {
            createNewClassificationLabel();
        }
        System.out.println(MessageFormat.format("Predicted label {0}", predictedLabel.getLabelValue()));
        return predictedLabel;
    }

    private void createNewClassificationLabel() {
        System.out.println("Creating new ClassificationLabel based on input instance.");
        if (evenSum >= oddSum)
            predictedLabel = new ClassificationLabel(1);
        else
            predictedLabel = new ClassificationLabel(0);
    }

    private void createNewRegressionLabel() {
        System.out.println("Creating new RegressionLabel based on input instance.");
        if (evenSum >= oddSum)
            predictedLabel = new RegressionLabel(1);
        else
            predictedLabel = new RegressionLabel(0);
    }

    Label predictedLabel;
    private double oddSum;
    private double evenSum;
}
