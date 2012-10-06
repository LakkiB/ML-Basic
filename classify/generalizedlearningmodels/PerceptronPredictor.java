package cs475.classify.generalizedlearningmodels;

import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.util.HashMap;
import java.util.List;

public class PerceptronPredictor extends LinearThresholdClassifierBase {

    @Override
    public void initializeParametersToDefaults(List<Instance> instances) {
        for(Integer key : instances.get(0).getFeatureVector().getFeatureVectorKeys())
            weightVectorW.put(key, 0.0);

        thickness           = 0;
        learningRateEeta    = 1;
        scalarThresholdBeta = 0;
    }

    @Override
    protected void updateWeight(Label prediction, FeatureVector fv, HashMap<Integer, Double> weightVectorW, double learningRate) {
        for(Integer key : fv.getFeatureVectorKeys())
            weightVectorW.put(key, (weightVectorW.get(key) + ( learningRate * prediction.getLabelValue() * fv.get(key) )) );
    }
}
