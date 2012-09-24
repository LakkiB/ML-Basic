package cs475.classify;

import cs475.dataobject.Instance;
import cs475.dataobject.label.Label;

import java.util.*;

public class MajorityClassifier extends Predictor {
    public MajorityClassifier() {
        maxLabels       = new HashSet<Label>();
        labelFrequency  = new HashMap<Label, Long>();
    }

    @Override
    public void train(List<Instance> instances) {
        computeLabelFrequency(instances);
        generateMaxLabelSetEx2();
        chooseMaxLabelFromSet();
        cleanUp();
    }

    private void cleanUp() {
        labelFrequency.clear();
        maxLabels.clear();
    }

    @Override
    public Label predict(Instance instance) {
        return getMaxLabel();
    }


    private void chooseMaxLabelFromSet() {
        Random rand = new Random();
        Object[] labels = maxLabels.toArray();
        int min = 0, max = labels.length;
        // nextInt is normally exclusive of the top value,
        // so add 1 to make it inclusive
        int randomNum = rand.nextInt(max - min ) + min;
        setMaxLabel((Label) labels[randomNum]);
    }


    // This works for a sorted map implementation which I think is unnecessary here.

    /* private void generateMaxLabelSet() {
        Long maxVal;
        do {
            maxVal = labelFrequency.get(labelFrequency.lastKey());
            maxLabels.add(labelFrequency.lastKey());
            labelFrequency.remove(labelFrequency.lastKey());
        } while (maxVal >= labelFrequency.get(labelFrequency.lastKey()));
    }*/

    // This implementation is simple but non-performing perhaps
    private void generateMaxLabelSetEx2() {
        Long maxVal = Long.MIN_VALUE;
        for (Long val : labelFrequency.values())
            if (val > maxVal) maxVal = val;

        maxLabels = getKeysBasedOnValue(labelFrequency, maxVal);
    }

    private void computeLabelFrequency(List<Instance> instances) {
        for (Instance instance : instances) {
            Label label = instance.getLabel();
            Long value = labelFrequency.get(label);
            if (value == null)
                labelFrequency.put(label, (long) 1);
            else
            {
                labelFrequency.remove(label);
                labelFrequency.put(label, ++value);
            }
        }
    }

    public static <Temp, Entry> Set<Temp> getKeysBasedOnValue(Map<Temp, Entry> map, Entry value) {
        Set<Temp> keys = new HashSet<Temp>();
        for (Map.Entry<Temp, Entry> entry : map.entrySet()) {
            if (value.equals(entry.getValue())) {
                keys.add(entry.getKey());
            }
        }
        return keys;
    }

    public Label getMaxLabel() {
        return maxLabel;
    }

    public void setMaxLabel(Label maxLabel) {
        this.maxLabel = maxLabel;
    }

    private Label maxLabel;
    private Set<Label> maxLabels;
    private HashMap<Label, Long > labelFrequency;
}
