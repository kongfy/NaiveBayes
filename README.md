NaiveBayes
==========

　　朴素贝叶斯分类器的实现方法较为简单，主要的实现思路为在训练过程中统计每个类别出现的次数、计算每个类的先验概率P(C)，并分类别统计属性值出现的次数、计算对应的条件概率P(x|C)。预测时根据各个属性值计算出该实例属于各个类别的后验概率P(X|C)，找到使P(X|C)P(C)最大的类别C完成预测。
　　
　　实现中使用Hashmap作为统计出现次数并计算概率的基本单元，对于P(C)构建Hashmap直接进行统计和计算，Hashmap的key为类别的标号；对于P(x|C)则使用Hashmap的二维数组进行统计和计算，数组的下标为类别的序号（非标号）和属性的序号，Hashmap的key为属性值。
　　
　　特别的，为了避免0概率的出现，对分类器进行了拉普拉斯校准，添加虚拟样本以保证不会出现0概率。
　　
　　对缺失属性进行了简单处理，离散属性取样本中该属性出现较多的值，连续属性则取样本中该属性的平均值。
