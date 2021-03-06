package cs475.classifier;

import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class Predictor implements Serializable
{
    private static final long serialVersionUID = 1L;

    public abstract void train ( List<Instance> instances );

    public abstract Label predict ( Instance instance );

    public static <Temp, Entry> Set<Temp> getKeysBasedOnValue ( Map<Temp, Entry> map, Entry value )
    {
        Set<Temp> keys = new HashSet<Temp>();
        for ( Map.Entry<Temp, Entry> entry : map.entrySet() )
        {
            if ( value.equals( entry.getValue() ) )
            {
                keys.add( entry.getKey() );
            }
        }
        return keys;
    }
}
