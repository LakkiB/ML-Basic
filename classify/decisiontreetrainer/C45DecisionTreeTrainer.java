package cs475.classify.decisiontreetrainer;

import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.text.MessageFormat;
import java.util.*;

public class C45DecisionTreeTrainer {

    public C45DecisionTreeTrainer() {
        summationOfFeatureValueOverAllLabels = new HashMap<Integer, HashMap<Double, Double>>();
        labelToValueFrequencyMultiMap = new HashMap<Label,HashMap<Integer, HashMap<Double, Double>>>();
        entropiesOfFeatures = new HashMap<Integer, Double>();
    }


    public int getFeatureWithLeastEntropy(List<Instance> instances, List<Integer> usedFeatures)
    {
        cookFrequencyOfFeatureValues(instances);
        return getFeatureWithBestIG(usedFeatures);
    }

    private void printEntropyStatistics(List<Integer> usedFeatures) {
        Double maxEntropy = 0.0, minEntropy = 0.0;
        int noOfZeros = 0;
        for(Double value : entropiesOfFeatures.values()) {
            if(value > maxEntropy)
                maxEntropy = value;
            if(value == 0.0)
                noOfZeros++;
            if(value < minEntropy)
                minEntropy = value;
        }
        System.out.println("Already used features = " + usedFeatures);
        System.out.println(MessageFormat.format("Printing entropy stats: max = {0}, min = {1}, noOfZeroes {2}", maxEntropy, minEntropy, noOfZeros));
    }

    private void cookFrequencyOfFeatureValues(List<Instance> instances) {
        noOfInstances = instances.size();
        for (Instance instance : instances) {
            for (Map.Entry<Integer, Double> uniqueFeature : instance.getFeatureVector().getEntrySet()) {
                Double valueOfThisFeature = uniqueFeature.getValue();
                Integer keyOfThisFeature = uniqueFeature.getKey();

                HashMap<Integer, HashMap<Double, Double>> fvColumnIndexToFrequencyMap = labelToValueFrequencyMultiMap.get(instance.getLabel());

                // did not find this label. Create a new label map and add it
                if(fvColumnIndexToFrequencyMap == null) {
                    fvColumnIndexToFrequencyMap = new HashMap<Integer, HashMap<Double, Double>>();
                    labelToValueFrequencyMultiMap.put(instance.getLabel(), fvColumnIndexToFrequencyMap);
                }

                HashMap<Double, Double> frequencyMap = fvColumnIndexToFrequencyMap.get(keyOfThisFeature);
                // check if this works? this works! hurray!
                frequencyMap = calculateAbsoluteAndConditionalFrequencies(valueOfThisFeature, keyOfThisFeature, frequencyMap);
                fvColumnIndexToFrequencyMap.put(keyOfThisFeature, frequencyMap);
            }
        }
    }

    private HashMap<Double, Double> calculateAbsoluteAndConditionalFrequencies(Double valueOfThisFeature, Integer keyOfThisFeature, HashMap<Double, Double> frequencyMap) {
        if (frequencyMap != null)
        {
            Double frequencyOfThisValue =
                    (frequencyOfThisValue = frequencyMap.get(valueOfThisFeature)) == null? 0.0
                            :frequencyOfThisValue;

            // Increment frequency and add it back to the map
            frequencyOfThisValue++;
            frequencyMap.put(valueOfThisFeature, frequencyOfThisValue);  // for p(y_i,x_j)
            incrementAbsoluteFrequencyForThisValue(keyOfThisFeature, valueOfThisFeature); // for P(x_i)
        }
        else
        {
            frequencyMap = new HashMap<Double, Double>();
            frequencyMap.put(valueOfThisFeature, 1.0 /*initial frequency*/);
            incrementAbsoluteFrequencyForThisValue(keyOfThisFeature, valueOfThisFeature);
        }
        return frequencyMap;
    }

    private void incrementAbsoluteFrequencyForThisValue(Integer featureColIndex, Double valueOfThisFeature) {
        Double absoluteFrequencyOfThisValue;
        HashMap<Double , Double> absoluteValueFrequencyPair =  summationOfFeatureValueOverAllLabels.get(featureColIndex);
        if(absoluteValueFrequencyPair == null)
            absoluteValueFrequencyPair = new HashMap<Double, Double>();
        if((absoluteFrequencyOfThisValue = absoluteValueFrequencyPair.get(valueOfThisFeature)) == null)
            absoluteFrequencyOfThisValue = 0.0;
        absoluteValueFrequencyPair.put(valueOfThisFeature, ++absoluteFrequencyOfThisValue);
        summationOfFeatureValueOverAllLabels.put(featureColIndex, absoluteValueFrequencyPair); // for p(x_i)
    }

    private Integer getFeatureWithBestIG(List<Integer> usedFeatures) {
        // H(Y|X) =  sum-i=1ton(sumj=1ton(P(yi,xj) log P(yi,xj)/P(xj)))

        for (Label label : labelToValueFrequencyMultiMap.keySet()) {
            HashMap<Integer, HashMap<Double, Double>> featuresToValueFrequencyMap = labelToValueFrequencyMultiMap.get(label);

            for (Integer featureIndex /* A particular feature */ : featuresToValueFrequencyMap.keySet()) {
                Double entropyOfLabelGivenFeature          = 0.0;
                Double probabilityOfLabelForThisFeature;
                Double frequencyOfThisValueForThisLabel;
                HashMap<Double, Double> valueAndFrequency  = featuresToValueFrequencyMap.get(featureIndex);

                for (Map.Entry<Double, Double> uniqueFeatureValue : valueAndFrequency.entrySet()) {
                    Double probabilityOfThisFeatureValue = summationOfFeatureValueOverAllLabels
                            .get(featureIndex).get(uniqueFeatureValue.getKey())/*frequency*/ / noOfInstances;  // p(xi)

                    frequencyOfThisValueForThisLabel = uniqueFeatureValue.getValue();
                    probabilityOfLabelForThisFeature = frequencyOfThisValueForThisLabel/noOfInstances;

                    entropyOfLabelGivenFeature += -1 * (probabilityOfLabelForThisFeature *
                            (Math.log((probabilityOfLabelForThisFeature)/ probabilityOfThisFeatureValue) / Math.log(2)) );
                }

                recomputeEntropyOfLabelForGivenFeature(featureIndex, entropyOfLabelGivenFeature);
            }
        }
        printEntropyStatistics(usedFeatures);
        for(Integer feature : usedFeatures)
            entropiesOfFeatures.remove(feature);

        return returnFeatureWithLeastEntropy(entropiesOfFeatures);
    }

    private Integer returnFeatureWithLeastEntropy(HashMap<Integer, Double> entropiesOfFeatures) {
        Double minEntropy = Double.MAX_VALUE;
        Map.Entry<Integer, Double> featureWithMinEntropy = null;
        for(Map.Entry<Integer, Double> entry: entropiesOfFeatures.entrySet())
            if(entry.getValue() <  minEntropy){
                minEntropy = entry.getValue();
                featureWithMinEntropy = entry;
            }
        assert featureWithMinEntropy != null;
        System.out.println(MessageFormat.format("Feature Index with least entropy: {0}", featureWithMinEntropy.getKey()));
        return featureWithMinEntropy.getKey();
    }

    private void recomputeEntropyOfLabelForGivenFeature(Integer featureIndex, Double entropyOfLabelGivenFeature) {
        if(entropiesOfFeatures.get(featureIndex) != null)
            entropyOfLabelGivenFeature = entropiesOfFeatures.get(featureIndex) + entropyOfLabelGivenFeature;
        entropiesOfFeatures.put(featureIndex, entropyOfLabelGivenFeature);
    }

    private long noOfInstances;
    private HashMap<Integer, Double> entropiesOfFeatures;  // H(Y|X)
    private HashMap<Integer, HashMap<Double, Double>> summationOfFeatureValueOverAllLabels;

    private HashMap<Label/*outputLabel*/,
            HashMap<Integer /*columnIndex*/,
                    HashMap<Double /*Value*/, Double /*frequency*/>>> labelToValueFrequencyMultiMap;
}
