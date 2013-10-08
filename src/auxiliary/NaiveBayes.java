/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.util.HashMap;
import java.util.Iterator;

/**
 *
 * @author MF1333020 孔繁宇
 */

class Bundle  {
	int count;
	double probability;
}

public class NaiveBayes extends Classifier {
    private double[][] _features;
    private boolean[] _isCategory;
    private double[] _labels;
    private double[] _defaults;
    private int _attrCount;
    
    private HashMap<Double, Bundle> _labelCounter;
    private HashMap<Double,Bundle>[][] _attrCounter;
    private double[] _labelList;
    

    public NaiveBayes() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
        _features = features;
        _isCategory = isCategory;
        _labels = labels;
        _attrCount = _isCategory.length - 1;
        
        //处理缺失属性
        _defaults = kill_missing_data();
        
        countForLabels();
        countForAttributes();
        updateProbability();
    }

    @Override
    public double predict(double[] features) {
    	//处理缺失属性
    	for (int i = 0; i < features.length; ++i) {
    		if (Double.isNaN(features[i])) {
    			features[i] = _defaults[i];
    		}
    	}
    	
    	double probability = -1;
    	int anser = 0;
    	for (int i = 0; i < _labelList.length; ++i) {
			double temp = calculateProbabilityForLabelIndex(i, features);
			if (temp > probability) {
				probability = temp;
				anser = i;
			}
		}
    	
    	return anser;
    }
    
    private double[] kill_missing_data() {
    	int num = _isCategory.length - 1;
    	double[] defaults = new double[num];
    	
    	for (int i = 0; i < defaults.length; ++i) {
    		if (_isCategory[i]) {
    			//离散属性取最多的值
    	    	HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
    	    	for (int j = 0; j < _features.length; ++j) {
    	    		double feature = _features[j][i];
    	    		if (!Double.isNaN(feature)) {
    	    			if (counter.get(feature) == null) {
        	    			counter.put(feature, 1);
        	    		} else {
        	    			int count = counter.get(feature) + 1;
        	    			counter.put(feature, count);
        	    		}
    	    		}
    	    	}
    	    	
    	    	int max_time = 0;
    	    	double value = 0;
    	    	Iterator<Double> iterator = counter.keySet().iterator();
    	    	while (iterator.hasNext()) {
    	    		double key = iterator.next();
    	    		int count = counter.get(key);
    	    		if (count > max_time) {
    	    			max_time = count;
    	    			value = key;
    	    		}
    	    	}
    	    	defaults[i] = value;
    		} else {
    			//连续属性取平均值
    			int count = 0;
    			double total = 0;
    			for (int j = 0; j < _features.length; ++j) {
    				if (!Double.isNaN(_features[j][i])) {
    					count++;
    					total += _features[j][i];
    				}
    			}
    			defaults[i] = total / count;
    		}
    	}
    	
    	//代换
    	for (int i = 0; i < _features.length; ++i) {
    		for (int j = 0; j < defaults.length; ++j) {
    			if (Double.isNaN(_features[i][j])) {
    				_features[i][j] = defaults[j];
    			}
    		}
    	}
    	return defaults;
    }
    
    private void countForLabels() {
    	_labelCounter = new HashMap<Double, Bundle>();
    	int count = 0;
    	
    	for (double label : _labels) {
    		Bundle bundle = _labelCounter.get(label);
			if (bundle == null) {
				bundle = new Bundle();
				bundle.count = 1;
				_labelCounter.put(label, bundle);
				count++;
			} else {
				bundle.count++;
			}
			bundle.probability = (double)bundle.count / _labels.length;
		}
    	
    	_labelList = new double[count];
    	Iterator<Double> iterator = _labelCounter.keySet().iterator();
    	int index = 0;
    	while (iterator.hasNext()) {
    		_labelList[index++] = iterator.next();
    	}
    }
    
    private void countForAttributes() {
    	_attrCounter = new HashMap[_labelList.length][];
    	
    	for (int i = 0; i < _labelList.length; ++i) {
    		HashMap<Double, Bundle>[] temp = new HashMap[_attrCount];
    		_attrCounter[i] = temp;
    		for (int j = 0; j < _attrCount; ++j) {
    			//统计属性j中属于类别i的数目
    			HashMap<Double, Bundle> counter = new HashMap<Double, Bundle>();
    			temp[j] = counter;
    		}
    	}
    	
    	for (int i = 0; i < _features.length; ++i) {
    		HashMap<Double, Bundle>[] temp = _attrCounter[indexForLabel(_labels[i])];
    		for (int j = 0; j < _attrCount; ++j) {
    			HashMap<Double , Bundle> counter = temp[j];
    			Bundle bundle = counter.get(_features[i][j]);
    			
    			if (bundle == null) {
    				bundle = new Bundle();
    				bundle.count = 1;
    				counter.put(_features[i][j], bundle);
    			} else {
    				bundle.count++;
    			}
    			bundle.probability = (double)bundle.count / _labelCounter.get(_labels[i]).count;
    		}
    	}
    }
    
    private int indexForLabel(double label) {
    	for (int i = 0; i < _labelList.length; ++i) {
    		if (_labelList[i] == label) {
    			return i;
    		}
    	}
    	return 0;
    }
    
    private void updateProbability() {
    	
    }
    
    private double calculateProbabilityForLabelIndex(int index, double[] features) {
    	double label = _labelList[index];
    	double temp = _labelCounter.get(label).probability;
    	for (int i = 0; i < features.length; ++i) {
    		HashMap<Double, Bundle> counter = _attrCounter[indexForLabel(label)][i];
    		Bundle bundle = counter.get(features[i]);
    		if (bundle == null) {
    			//System.out.println("error no bundle");
    			temp *= 0;
    			//拉普拉斯校对
    		} else {
    			temp *= bundle.probability;
    		}
    	}
    	
    	return temp;
    }
    
}
