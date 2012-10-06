package cs475.classify.generalizedlearningmodels;

import cs475.classify.Predictor;
import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NaiveBayesPredictor extends Predictor {
    public NaiveBayesPredictor() {
        featureType                      = new HashMap<Integer, Boolean>();
        meanOfFeatures                   = new HashMap<Integer, Double>();
        probabilityOfLabels              = new HashMap<Label, Double>();
        labelsFeatureProbability         = new HashMap<Label, HashMap<Integer, Double>>();
        labelToDiscreteValueFrequencyMap = new HashMap<Label, HashMap<Integer, HashMap<Double, Double>>>();
    }

    @Override
    public void train(List<Instance> instances) {

        double lambda = getLambda();
        cookFrequencyOfFeatureValues(instances);
        makeBinaryOrContinuousFeatureClassification(instances);

        computeMeanOfFeatures(instances);
        computeProbabilityOfLabels(instances, lambda);
        computeProbabilityOfFeatureGivenLabels(instances, labelsFeatureProbability, lambda);
    }

    private void computeProbabilityOfFeatureGivenLabels(List<Instance> instances, HashMap<Label, HashMap<Integer, Double>> labelsFeatureProbability, double lambda) {
        for(Label label : labelToDiscreteValueFrequencyMap.keySet())
            for(Map.Entry<Integer, HashMap<Double, Double>> featureFrequencyMap : labelToDiscreteValueFrequencyMap.get(label).entrySet())
            {
                int countOfFeatureFiring = 0;
                for(Double aCertainValue /* xi */ : featureFrequencyMap.getValue().keySet()){
                    if(aCertainValue != 0) // Feature fired only if its value is not zero
                        countOfFeatureFiring += featureFrequencyMap.getValue().get(aCertainValue);
                }
                labelsFeatureProbability.put(label, null);
                labelsFeatureProbability.get(label).put(featureFrequencyMap.getKey(),
                        ((countOfFeatureFiring * 1.0) + lambda) / (instances.size() + lambda * instances.size()));
            }
    }

    private void computeProbabilityOfLabels(List<Instance> instances, double lambda) {
        HashMap<Label, Integer> sumsOfLabels = new HashMap<Label, Integer>();
        computeProbabilityOfAllLabels(instances, sumsOfLabels, probabilityOfLabels, lambda);
    }

    private void computeMeanOfFeatures(List<Instance> instances) {
        HashMap<Integer, Double> sumsOfFeatures = new HashMap<Integer, Double>();
        computeMeanOfAllFeatures(instances, sumsOfFeatures, meanOfFeatures);
    }


    private void makeBinaryOrContinuousFeatureClassification(List<Instance> instances) {
        markAllFeaturesAsBinary(instances);

        for (Label label : labelToDiscreteValueFrequencyMap.keySet())
            for(Map.Entry<Integer, HashMap<Double, Double>> featureMap : labelToDiscreteValueFrequencyMap.get(label).entrySet())
                for(Double value : featureMap.getValue().keySet())
                    if((value != 0.0) && (value != 1.0)){
                        featureType.put(featureMap.getKey(), false /* continuous */);
                        break;
                    }
    }

    private void markAllFeaturesAsBinary(List<Instance> instances) {
        FeatureVector aFV = instances.get(0).getFeatureVector();
        for(Integer keys : aFV.getFeatureVectorKeys())
            featureType.put(keys, true);
    }


    private void cookFrequencyOfFeatureValues(List<Instance> instances) {

        for (Instance instance : instances) {
            for (Map.Entry<Integer, Double> uniqueFeature : instance.getFeatureVector().getEntrySet()) {
                Double valueOfThisFeature = uniqueFeature.getValue();
                Integer keyOfThisFeature = uniqueFeature.getKey();

                HashMap<Integer, HashMap<Double, Double>> fvColumnIndexToFrequencyMap = labelToDiscreteValueFrequencyMap.get(instance.getLabel());

                // did not find this label. Create a new label map and add it
                if(fvColumnIndexToFrequencyMap == null) {
                    fvColumnIndexToFrequencyMap = new HashMap<Integer, HashMap<Double, Double>>();
                    labelToDiscreteValueFrequencyMap.put(instance.getLabel(), fvColumnIndexToFrequencyMap);
                }

                HashMap<Double, Double> frequencyMap = fvColumnIndexToFrequencyMap.get(keyOfThisFeature);
                // check if this works? this works! hurray!
                frequencyMap = calculateConditionalFrequencies(valueOfThisFeature, frequencyMap);
                fvColumnIndexToFrequencyMap.put(keyOfThisFeature, frequencyMap);
            }
        }
    }


    private HashMap<Double, Double> calculateConditionalFrequencies(Double valueOfThisFeature, HashMap<Double, Double> frequencyMap) {
        if (frequencyMap != null) {
            Double frequencyOfThisValue =
                    (frequencyOfThisValue = frequencyMap.get(valueOfThisFeature)) == null ? 0.0
                            : frequencyOfThisValue;

            // Increment frequency and add it back to the map
            frequencyOfThisValue++;
            frequencyMap.put(valueOfThisFeature, frequencyOfThisValue);
        } else {
            frequencyMap = new HashMap<Double, Double>();
            frequencyMap.put(valueOfThisFeature, 1.0 /*initial frequency*/);
        }
        return frequencyMap;
    }



    private void computeProbabilityOfAllLabels(List<Instance> instances, HashMap<Label, Integer> sumsOfLabels, HashMap<Label, Double> probabilityOfLabels, double lambda) {
        for (Instance instance : instances) {
            Label label = instance.getLabel();
            int noOfLabels = sumsOfLabels.get(label);
            sumsOfLabels.put(label, noOfLabels + 1);
        }

        for (Map.Entry<Label, Integer> frequencyOfLabel : sumsOfLabels.entrySet())
            probabilityOfLabels.put(frequencyOfLabel.getKey(),
                    (frequencyOfLabel.getValue().doubleValue() + lambda) / (instances.size() + lambda * instances.size()));
    }

    private void computeMeanOfAllFeatures(List<Instance> instances, HashMap<Integer, Double> sumsOfFeatures, HashMap<Integer, Double> meanOfFeatures) {
        for (Instance instance : instances)
            for (Map.Entry<Integer, Double> fvCell : instance.getFeatureVector().getEntrySet()) {
                Double value = sumsOfFeatures.get(fvCell.getKey());
                if (value == null) value = 0.0;
                sumsOfFeatures.put(fvCell.getKey(), value + fvCell.getValue());
            }
        for (Map.Entry<Integer, Double> featureSum : sumsOfFeatures.entrySet())
            meanOfFeatures.put(featureSum.getKey(), featureSum.getValue() / instances.size());
    }


    @Override
    public Label predict(Instance instance) {


        for(Map.Entry<Label, HashMap<Integer, HashMap<Double, Double>>> aLabelsProbabilityMap : labelToDiscreteValueFrequencyMap.entrySet())
            for(aLabelsProbabilityMap.getValue().entrySet())
    }


    private double getLambda() {
        double lambda = 1.0;
        if (CommandLineUtilities.hasArg("lambda"))
            lambda = CommandLineUtilities.getOptionValueAsFloat("lambda");
        return lambda;
    }


    private HashMap<Integer, Boolean>                featureType;
    private HashMap<Integer, Double>                 meanOfFeatures;
    private HashMap<Label, Double>                   probabilityOfLabels;       // P(Y)
    private HashMap<Label, HashMap<Integer, Double>> labelsFeatureProbability;

    private HashMap<Label/*outputLabel*/,
            HashMap<Integer /*columnIndex*/,
                    HashMap<Double /*Value*/, Double /*frequency*/>>> labelToDiscreteValueFrequencyMap;  // P(Xi|Y=yj) is got from this map
}
