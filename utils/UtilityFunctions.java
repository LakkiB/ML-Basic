package cs475.utils;

import cs475.dataobject.Instance;

import java.util.List;

public class UtilityFunctions
{
    public static int getNumberOfFeatures ( List<Instance> instances )
    {
        int maxIndex = 0;
        for ( Instance instance : instances )
        {
            for ( Integer feature : instance.getFeatureVector().getFeatureVectorKeys() )
            {
                if ( feature > maxIndex )
                {
                    maxIndex = feature;
                }
            }
        }
        return maxIndex;
    }
}
