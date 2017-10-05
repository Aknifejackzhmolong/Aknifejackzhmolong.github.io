---
layout: post
title: tensorflow实战系列（一）——实现自编码器
date: 2017-10-05 17:22:24.000000000 +09:00
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
#tensorflow实战系列（一）——实现自编码器#
##本节使用的Python模块介绍##
这里我们先导入常用库`Numpy`，还有Scikit-learn（`sklearn`）中的`preprocessing`模块，这是一个对数据预处理的常用模块，之后我们会使用其中的数据标准化功能。同时本节依然使用MNIST数据集，因此也导入Tensorflow中的`MNIST数据加载模块`。本节代码主要来自Tensorflow的开源实现
```python
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```

##定义Xavier初始化器##
* ###功能：初始化模型权重weight
* ###理论依据
    Xavier Glorot 和深度学习三巨头之一的 Yoshua Bengio 在一片论文中指出如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，那信号将在每层间传递时逐渐放大导致发散和失效。         
    
    而 Xavier 初始化器做的事情就是让初始化的权重不大不小，正好合适。    
    
    从数学角度分析，Xavier 就是让权重满足均值为0，同时方差为\\(x=\frac{2}{n_{in}+n_{out}}\\)       
    
```python
def xavier_init(fan_in, fan_out, constant=1):
    #fan_in为输入节点数量,fan_out为输出节点数量
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	#产生在low和high之间的随机分布
	return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)
```
 ###tf.random_uniform()函数中使用的参数说明
    * 第一个参数中元组 (fan_in, fan_out) 控制random_uniform()输出一个矩阵，行数为fan_in，列数为fan_out，人工神经网络中`输入参数向前传递`的数学形式就是`矩阵运算`
    * 参数`minval`和`maxval`表示随机数函数random_uniform()的值的范围
    * `dtype`则是数据类型的说明


##代码整体

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant=1):
    #fan_in为输入节点数量,fan_out为输出节点数量
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
	high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	#产生在low和high之间的随机分布
	return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)


def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train, X_test


def get_random_block_from_data(data, batch_size):
	start_index = np.random.randint(0, len(data) - batch_size)
	return data[start_index:(start_index + batch_size)]


class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
		self.n_input = n_input							 #输入数据数
		self.n_hidden = n_hidden						 #隐含层数
		self.transfer = transfer_function                #隐含层激活函数
		self.scale = tf.placeholder(tf.float32)          #占位符，或者说变量
		self.training_scale = scale                      #高斯噪声系数
		network_weights = self._initialize_weights()	 #初始化网络权重
		self.weights = network_weights

		#-------------接下来定义网络结构--------------

		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

		#--------------定义解码损失函数---------------

		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()			#全部变量初始化
		self.sess = tf.Session()							#定义会话
		self.sess.run(init)									#会话初始化


	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
		return all_weights


	def partial_fit(self, X):
		#placeholder不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
		return cost


	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})


	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})


	def generate(self, hidden=None):
		if hidden is None:
			hidden = np.random.normal(size = self.weights["b1"])
		return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})


	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})


	def getWeights(self):
		return self.sess.run(self.weights['w1'])


	def getBiases(self):
		return self.sess.run(self.weights['b1'])


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 200 						#迭代次数
batch_size = 128
display_step = 1 							#display间隔
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)

if __name__ == '__main__' :
	print "start"
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		for i in range(total_batch):
			batch_xs = get_random_block_from_data(X_train, batch_size)

			cost = autoencoder.partial_fit(batch_xs)
			avg_cost += cost / n_samples * batch_size

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

	print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
```
