package cs475.classify.generalizedlearningmodels;

import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.util.HashMap;
import java.util.List;

public class PerceptronPredictor extends LinearThresholdClassifierBase {

    @Override
    public void initializeParametersToDefaults(List<Instance> instances) {
        int n = getTotalNoOfFeatures(instances);
        for(int i = 1; i <= n ; i++)
            weightVectorW.put(i, 0.0);

        thickness           = 0;
        learningRateEeta    = 1;
        scalarThresholdBeta = 0;
    }

    private int getTotalNoOfFeatures(List<Instance> instances) {
        int maxIndex = 0;
        for(Instance instance : instances)
            for(Integer featureIndex : instance.getFeatureVector().getFeatureVectorKeys())
                if(featureIndex > maxIndex )
                    maxIndex = featureIndex;
        return maxIndex;
    }

    @Override
    protected void updateWeight(Label yi, FeatureVector fv, HashMap<Integer, Double> weightVectorW, double learningRate) {
        double yiValue = yi.getLabelValue() == 0.0? -1: yi.getLabelValue();
        for(Integer key : fv.getFeatureVectorKeys())
        {
            double oldWeight = weightVectorW.get(key);
            weightVectorW.put(key, (oldWeight + ( learningRate * yiValue * fv.get(key) )) );
        }
    }
}
