package cs475.dataobject;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class FeatureVector implements Serializable, Cloneable {

    public FeatureVector() {
        featureVector = new HashMap<Integer, Double>();
    }

    public void add(int index, double value) {
        // This is a Sparse vector
        if (value != 0)
            featureVector.put(index, value);
	}
	
	public Double get(int index) {
        return featureVector.get(index);// != null ? returnVal : 0;
	}

    public Set<Integer> getFeatureVectorKeys()
    {
        return featureVector.keySet();
    }

    public  Set<Map.Entry<Integer, Double>> getEntrySet()
    {
        return featureVector.entrySet();
    }

    public Object clone() throws CloneNotSupportedException {
        FeatureVector cloned =  new FeatureVector();
        cloned.featureVector = (HashMap<Integer, Double>)this.featureVector.clone();
        super.clone();
        return cloned;
    }


    private HashMap<Integer, Double> featureVector;
}
