package cs475.evaluate;

import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;
import cs475.classify.Predictor;

import java.util.List;

public class AccuracyEvaluator extends Evaluator
{
    @Override
    public double evaluate ( List<Instance> instances, Predictor predictor )
    {
        double match = 0;
        for ( Instance instance : instances )
        {
            Label label = instance.getLabel();
            if ( label != null && predictor.predict( instance ).getLabelValue() == label.getLabelValue() )
            {
                ++match;
            }
        }
        return match;
    }

    public double evaluateR ( List<Instance> instances, Predictor predictor )
    {
        double meanError = 0;
        int size = 0;
        for ( Instance instance : instances )
        {
            Label label = instance.getLabel();
            if ( label == null )
            {
                continue;
            }

            Label prediction = predictor.predict( instance );
            meanError += Math.abs( prediction.getLabelValue() - label.getLabelValue() );
            size++;
        }
        return meanError/size;
    }

}
