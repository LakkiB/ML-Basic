package cs475.classify.generalizedlearningmodels;

import cs475.classify.Predictor;
import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.ClassificationLabel;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class LinearThresholdClassifierBase extends Predictor{

    protected LinearThresholdClassifierBase() {
        wDotX               = new HashMap<Integer, Double>();
        weightVectorW       = new HashMap<Integer, Double>();
    }

    public abstract void initializeParametersToDefaults(List<Instance> instances);

    protected abstract void updateWeight(Label prediction, FeatureVector fv, HashMap<Integer, Double> weightVectorW, double learningRate);

    @Override
    public void train(List<Instance> instances) {
        initializeParametersToDefaults(instances);
        initializeParametersIfArgumentsProvided();

        for(int i = 0; i < noOfLearningIterationsI; ++i)
            for(Instance instance : instances){

                Label givenLabel            = instance.getLabel();
                FeatureVector fv            = instance.getFeatureVector();
                double summationOfWDotX     = 0;

                for(Map.Entry<Integer, Double> fvCell : fv.getEntrySet()){
                    double weight = weightVectorW.get(fvCell.getKey());
                    summationOfWDotX += fvCell.getValue() * weight;
                }

                Label prediction = makePrediction(summationOfWDotX, scalarThresholdBeta, thickness);
                if(prediction.getLabelValue() != givenLabel.getLabelValue()){
                    updateWeight(givenLabel, fv, weightVectorW, learningRateEeta);
                }
            }
    }

    private Label makePrediction(double summationOfWDotX, double scalarThresholdBeta, double thickness) {

        Label prediction;
        if(summationOfWDotX >= scalarThresholdBeta + thickness)
            prediction = new ClassificationLabel(1);
        else if(summationOfWDotX <= scalarThresholdBeta - thickness)
            prediction = new ClassificationLabel(-1);
        else
            prediction = new ClassificationLabel(0);
        return prediction;
    }

    @Override
    public Label predict(Instance instance) {
        Double summationOfWDotX = 0.0;
        for(Map.Entry<Integer, Double> fvCell : instance.getFeatureVector().getEntrySet()) {
            double weight = weightVectorW.get(fvCell.getKey());
            summationOfWDotX += fvCell.getValue() * weight;
        }

        return makePrediction(summationOfWDotX, scalarThresholdBeta, thickness);
    }

    private void initializeParametersIfArgumentsProvided() {
        if (CommandLineUtilities.hasArg("online_learning_rate"))
            learningRateEeta = CommandLineUtilities.getOptionValueAsFloat("online_learning_rate");

        noOfLearningIterationsI = 4;
        if (CommandLineUtilities.hasArg("online_training_iterations"))
            noOfLearningIterationsI = CommandLineUtilities.getOptionValueAsInt("online_training_iterations");

        thickness = 0;
        if (CommandLineUtilities.hasArg("thickness"))
            thickness = CommandLineUtilities.getOptionValueAsFloat("thickness") ;
    }


    protected double thickness;
    protected double learningRateEeta;
    protected double scalarThresholdBeta;
    protected int noOfLearningIterationsI;
    protected HashMap<Integer, Double> wDotX;
    protected HashMap<Integer, Double> weightVectorW;
}
