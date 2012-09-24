package cs475.dataobject.label;

import cs475.dataobject.label.*;

import java.io.Serializable;

public class RegressionLabel extends cs475.dataobject.label.Label implements Serializable {

    public RegressionLabel(double label) {
        regressionLabel = label;
    }

    @Override
    public String toString() {
        return String.valueOf(regressionLabel);
    }

    @Override
    public double getLabelValue() {
        return regressionLabel;
    }

    private double regressionLabel;
}
