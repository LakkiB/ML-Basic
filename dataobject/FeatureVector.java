package cs475.dataobject;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class FeatureVector implements Serializable {

    public FeatureVector() {
        featureVector = new HashMap<Integer, Double>();
    }

    public void add(int index, double value) {
        // This is a Sparse vector
        if (value != 0)
            featureVector.put(index, value);
	}
	
	public double get(int index) {
        Double returnVal = featureVector.get(index);
		return returnVal != null ? returnVal : 0;
	}

    public Set<Integer> getFeatureVectorKeys()
    {
        return featureVector.keySet();
    }

    public  Set<Map.Entry<Integer, Double>> getEntrySet()
    {
        return featureVector.entrySet();
    }

    private HashMap<Integer, Double> featureVector;
}
