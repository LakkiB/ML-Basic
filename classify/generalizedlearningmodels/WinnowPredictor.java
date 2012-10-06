package cs475.classify.generalizedlearningmodels;

import cs475.dataobject.FeatureVector;
import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.util.HashMap;
import java.util.List;

public class WinnowPredictor extends LinearThresholdClassifierBase {

    @Override
    public void initializeParametersToDefaults(List<Instance> instances) {
        for(Integer key : instances.get(0).getFeatureVector().getFeatureVectorKeys())
            weightVectorW.put(key, 1.0);

        learningRateEeta    = 2.0;
        scalarThresholdBeta = instances.size()/2;
    }


    @Override
    protected void updateWeight(Label prediction, FeatureVector fv, HashMap<Integer, Double> weightVectorW, double learningRate) {
        for(Integer key : fv.getFeatureVectorKeys())
        {
            Double wDash =  weightVectorW.get(key) * Math.pow(learningRate, prediction.getLabelValue() * sign(fv.get(key)));
            weightVectorW.put(key, (wDash > mu) ? mu : wDash );
        }
    }

    private int sign(double v) {
        if (v > 0) return 1;
        else if (v < 0) return -1;
        else return 0;
    }

    private double mu = Math.pow(10,6);
}
