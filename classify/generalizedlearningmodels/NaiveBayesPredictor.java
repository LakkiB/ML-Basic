package cs475.classify.generalizedlearningmodels;

import cs475.classify.Predictor;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;
import cs475.utils.CommandLineUtilities;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NaiveBayesPredictor extends Predictor {
    public NaiveBayesPredictor() {
        lambda                           = 0.0;
        featureTypes                     = new HashMap<Integer, Boolean>();
        meanOfFeatures                   = new HashMap<Integer, Double>();
        trainingInstances                = new ArrayList<Instance>();
        probabilityOfLabels              = new HashMap<Label, Double>();
        noOfFeatureOccurrences           = new HashMap<Integer, Integer>();
        labelsFeatureProbability         = new HashMap<Label, HashMap<Integer, Double>>();
        likelihoodOfLabelGivenFeatures   = new HashMap<Label, Double>();
        labelToDiscreteValueFrequencyMap = new HashMap<Label, HashMap<Integer, HashMap<Double, Double>>>();
    }

    @Override
    public void train(List<Instance> instances) {

        double lambda = getLambda();
        cloneTrainingInstances(instances);
        totalNoOfFeaturesInTrainingSet = getTotalNoOfFeatures(trainingInstances);

        makeBinaryOrContinuousFeatureClassification(trainingInstances);
        computeMeanOfFeatures(trainingInstances);
        convertContinuousFeaturesIntoBinaryBySplitting(instances, meanOfFeatures, featureTypes);
        cookFrequencyOfFeatureValues(trainingInstances);

        computeProbabilityOfLabels(trainingInstances, lambda);
        computeProbabilityOfFeatureGivenLabels(trainingInstances, labelsFeatureProbability, lambda, totalNoOfFeaturesInTrainingSet);
    }

    private void cloneTrainingInstances(List<Instance> instances) {
       // trainingInstances = instances;
        for(Instance instance : instances)
            try {
                trainingInstances.add((Instance)instance.clone());
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
    }

    private void convertContinuousFeaturesIntoBinaryBySplitting(
            List<Instance> instances,
            HashMap<Integer, Double> meanOfFeatures,
            HashMap<Integer, Boolean> binaryFeatures) {

        for(int i = 0; i < instances.size() ; i++) {
            Instance instance = instances.get(i);
            Instance trainingInstance = trainingInstances.get(i);

            for(Integer feature : instance.getFeatureVector().getFeatureVectorKeys())
                if(!(binaryFeatures.get(feature) == null? false : binaryFeatures.get(feature))){  // needs splitting
                    if(instance.getFeatureVector().get(feature) >= meanOfFeatures.get(feature)) {
                        trainingInstance.getFeatureVector().add(feature, 1);
                        trainingInstance.getFeatureVector().add(++totalNoOfFeaturesInTrainingSet, 0);
                    }
                    else {
                        trainingInstance.getFeatureVector().add(feature, 0);
                        trainingInstance.getFeatureVector().add(++totalNoOfFeaturesInTrainingSet, 1);
                    }
                }
        }
    }

    private int getTotalNoOfFeatures(List<Instance> instances) {
        int maxIndex = 0;
        for(Instance instance : instances)
            for(Integer featureIndex : instance.getFeatureVector().getFeatureVectorKeys())
               if(featureIndex > maxIndex )
                   maxIndex = featureIndex;
        return maxIndex;
    }

    private void computeProbabilityOfFeatureGivenLabels(
            List<Instance> instances,
            HashMap<Label, HashMap<Integer, Double>> labelsFeatureProbability,
            double lambda,
            int noOfFeatures) {
        for(Label label : labelToDiscreteValueFrequencyMap.keySet()){
            for(Integer feature : labelToDiscreteValueFrequencyMap.get(label).keySet())
            {
                int countOfFeatureFiring = 0; // No. of times the feature fired for this label.
                HashMap<Double, Double> featureValueFrequency = labelToDiscreteValueFrequencyMap.get(label).get(feature);

                for(Double aCertainFeatureValue /* xi */ : featureValueFrequency.keySet()){
                    countOfFeatureFiring += featureValueFrequency.get(aCertainFeatureValue); // the no. of times this feature took this value
                }
                if(labelsFeatureProbability.get(label) == null)
                        labelsFeatureProbability.put(label, new HashMap<Integer, Double>());

                int size = countInstancesThatContainFeature(instances, feature);
                noOfFeatureOccurrences.put(feature, size);

                labelsFeatureProbability.get(label).put(feature, (countOfFeatureFiring + lambda)/(size + noOfFeatures*lambda));
            }
        }
    }

    private int countInstancesThatContainFeature(List<Instance> instances, int feature)
    {
        int count = 0;
        for(Instance instance : instances)
            if(instance.getFeatureVector().get(feature) != null) count++;
        return count;
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
        markAllFeaturesAsNonBinary();

        for (Instance instance : instances)
            for(Integer feature : instance.getFeatureVector().getFeatureVectorKeys()){
                if(featureTypes.get(feature)) continue; // Already set, nothing to do

                double featureValue =  instance.getFeatureVector().get(feature);
                if(featureValue == 0.0 || featureValue == 1.0)
                    featureTypes.put(feature, true);
            }
    }

    private void markAllFeaturesAsNonBinary() {
        for(int i = 1; i <= totalNoOfFeaturesInTrainingSet; i++)
                featureTypes.put(i, false);
    }


    private void cookFrequencyOfFeatureValues(List<Instance> instances) {

        for (Instance instance : instances) {
            for (Map.Entry<Integer, Double> uniqueFeature : instance.getFeatureVector().getEntrySet()) {

                Integer keyOfThisFeature = uniqueFeature.getKey();
                Double valueOfThisFeature = uniqueFeature.getValue();

                HashMap<Integer, HashMap<Double, Double>>
                        fvColumnIndexToFrequencyMap = labelToDiscreteValueFrequencyMap.get(instance.getLabel());

                // did not find this label. Create a new label map and add it
                if(fvColumnIndexToFrequencyMap == null) {
                    fvColumnIndexToFrequencyMap = new HashMap<Integer, HashMap<Double, Double>>();
                    labelToDiscreteValueFrequencyMap.put(instance.getLabel(), fvColumnIndexToFrequencyMap);
                }

                HashMap<Double, Double> frequencyMap = fvColumnIndexToFrequencyMap.get(keyOfThisFeature);
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


    private void computeProbabilityOfAllLabels(
            List<Instance> instances,
            HashMap<Label, Integer> sumsOfLabels,
            HashMap<Label, Double> probabilityOfLabels,
            double lambda) {
        computeSumOfLabels(instances, sumsOfLabels);

        for (Map.Entry<Label, Integer> labelAndCount : sumsOfLabels.entrySet())
            probabilityOfLabels.put(labelAndCount.getKey(),
                    (labelAndCount.getValue().doubleValue() + lambda) / (instances.size() + lambda * sumsOfLabels.size()));
    }


    private void computeSumOfLabels(List<Instance> instances, HashMap<Label, Integer> sumsOfLabels) {
        for (Instance instance : instances) {
            Label label = instance.getLabel();
            int noOfLabels = sumsOfLabels.get(label) == null? 0: sumsOfLabels.get(label);
            sumsOfLabels.put(label, noOfLabels + 1);
        }
    }



    private void computeMeanOfAllFeatures(
            List<Instance> instances,
            HashMap<Integer, Double> sumsOfFeatures,
            HashMap<Integer, Double> meanOfFeatures) {
        for (Instance instance : instances)
            for (Map.Entry<Integer, Double> fvCell : instance.getFeatureVector().getEntrySet()) {
                Double value = sumsOfFeatures.get(fvCell.getKey());
                if (value == null) value = 0.0;
                sumsOfFeatures.put(fvCell.getKey(), value + fvCell.getValue());
            }
        for (Map.Entry<Integer, Double> featureSum : sumsOfFeatures.entrySet())
            meanOfFeatures.put(featureSum.getKey(), featureSum.getValue() / countInstancesThatContainFeature(instances ,featureSum.getKey()));
    }

    @Override
    public Label predict(Instance instance) {
        for (Label label : labelsFeatureProbability.keySet()) {

            double logSumOfProbabilitiesForThisFeature = 0.0;
            double logProbabilityOfThisLabel = Math.log(probabilityOfLabels.get(label));

            for (Integer feature : instance.getFeatureVector().getFeatureVectorKeys()){
                Double probabilityOfFeatureGivenLabel = labelsFeatureProbability.get(label).get(feature);
                if ( probabilityOfFeatureGivenLabel == null)
                    probabilityOfFeatureGivenLabel = getLambda() / (totalNoOfFeaturesInTrainingSet * getLambda() + noOfFeatureOccurrences.get(feature));
                    // we know this feature; compute summation(log(xi|Y))
                    logSumOfProbabilitiesForThisFeature +=
                            ((Math.log(probabilityOfFeatureGivenLabel) - logProbabilityOfThisLabel));

            }
            likelihoodOfLabelGivenFeatures.put(label, logSumOfProbabilitiesForThisFeature + logProbabilityOfThisLabel);
        }
        Label returnL = getLabelWithMaxLikelihood(likelihoodOfLabelGivenFeatures);
        likelihoodOfLabelGivenFeatures.clear();
        return returnL;
    }

    private Label getLabelWithMaxLikelihood(HashMap<Label, Double> likelihoodOfLabelGivenFeatures) {
        Label predictedL = null;
        double maxLikelihood = -9999999;
        for(Map.Entry<Label, Double> labelAndItsLikelihood : likelihoodOfLabelGivenFeatures.entrySet()) {
            if (labelAndItsLikelihood.getValue() > maxLikelihood) maxLikelihood = labelAndItsLikelihood.getValue();
            predictedL = labelAndItsLikelihood.getKey();
        }
        return predictedL;
    }


    private double getLambda() {
        if(lambda > 0) return lambda;

        lambda = 1.0;
        if (CommandLineUtilities.hasArg("lambda"))
            lambda = CommandLineUtilities.getOptionValueAsFloat("lambda");
        return lambda;
    }

    private int                                      totalNoOfFeaturesInTrainingSet;
    private double                                   lambda;
    private List<Instance>                           trainingInstances;
    private HashMap<Label, Double>                   probabilityOfLabels;       // P(Y)
    private HashMap<Label, Double>                   likelihoodOfLabelGivenFeatures;
    private HashMap<Integer, Double>                 meanOfFeatures;
    private HashMap<Integer, Integer>                noOfFeatureOccurrences;
    private HashMap<Integer, Boolean>                featureTypes;
    private HashMap<Label, HashMap<Integer, Double>> labelsFeatureProbability;

    private HashMap<Label/*outputLabel*/,
            HashMap<Integer /*columnIndex*/,
                    HashMap<Double /*Value*/, Double /*frequency*/>>> labelToDiscreteValueFrequencyMap;  // P(Xi|Y=yj) is got from this map
}
