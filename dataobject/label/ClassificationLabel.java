package cs475.dataobject.label;
import cs475.dataobject.label.*;

import java.io.Serializable;

public class ClassificationLabel extends cs475.dataobject.label.Label implements Serializable {

	public ClassificationLabel(int label) {
        classificationLabel = label;
	}

	@Override
	public String toString() {
        return String.valueOf(classificationLabel);
	}

    @Override
    public double getLabelValue() {
        return classificationLabel;
    }

    // According to Joshua Block's Effective Java
    public int hashCode() {
        return 197 * 17 + this.toString().hashCode();
    }

    public boolean equals(Object inputObject) {
        if (inputObject == null)
            return false;
        if (inputObject == this)
            return true;
        if (inputObject.getClass() != getClass())
            return false;

        cs475.dataobject.label.Label label = (cs475.dataobject.label.Label) inputObject;
        return this.toString().equalsIgnoreCase(label.toString());
    }


    private int classificationLabel;
}
