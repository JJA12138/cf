# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:37:47 2018

@author: 10433
python环境：3.6.4
学习Andrew Ng机器学习公开课中，连接：http://study.163.com/course/courseMain.htm?courseId=1004570029
视屏中在损失函数中去除了m，在这里还是填上m，这样可以反应样本的平均误差，且适合迭代。
目前对协同过滤算法中的疑问以及推送方法：
        1.经过训练后的到的特征与原始的特征完全不一样，猜测特征经过了某种变换，而且这种变换，不会改变特征之间的规律
        2.因为是商品和用户之间是线性拟合，导致可能对异常点十分敏感？可以尝试其他拟合方式。
        4.运算量过大，10000个商品*1000000个用户得到10^4X10^6矩阵也就是10^10个值，而且矩阵越大，学习速率就要设置的
              越小，学习速率越小导致参数迭代次数越高，这样计算量就很爆炸了。用随机梯度的话每次迭代只能更新相对应随机的特征，
              所以迭代次数可能要增加不少，不知道效果会怎样。
        3.推送方法有三种，一：基于物品推送，也就是说用训练的到的特征预测出用户对于商品的评分，或者用商品之间相似度。
                         二：基于用户相似度推送，设置一个参数去建立一个相似度集群，这个参数可以是集群内用户个数
                                 也可以是相似度系数，然后将集合中用户评分较高的商品推荐。
                         三：结合方法一和二。
              第一种推送方法优点：较为精准，缺点：商品过于相似，对于用户来说没有新鲜商品，比如用户对动作类电影感兴趣，
                   那么此方法只会推荐动作类，而不会推荐一些用户没看过的其他类型电影。
              第二种推送方法优点：推送商品新鲜度较高，缺点：推送商品可能是用户不感兴趣的。
              第三种集合了方法一和二，可以更进准的推送感兴趣的商品，又能推荐新鲜的商品。从我用知乎的体验来说，
                   知乎应该就是这种方法，推送的文章设置了一个感兴趣的选择项，用户点击可以过滤不感兴趣
                   的推送，在以后推送基于用户相似度的商品时根据用户不感兴趣做删选推送。
                   知乎有一点做的不好，类似我这样的用户只想看关于科技的内容，不想看其他任何无关的，
                   就会饱受用户相似度推送的痛苦，导致这一原因可能是用户相似度设置的过宽，所以进一步的
                   做法是添加用户特征，这个特征用于控制相似度集群。

"""
import sys
sys.path.append(u'E:\\DATA\\Collaborative_filtering')
import numpy as np
from data_simulation import random_data
from model import collaborative_filtering
import time                

start=time.time()
data=random_data(10,100,random_noise=False)

train=data.y

        
cf=collaborative_filtering(3)
cf.set_parameter({'cv_rate':0.01,'num_iteration':5000,'alpha':0.01})
cf.fit(train.copy())

pred=cf.pred
y=cf.Y
y_copy=data.y_copy
#cv=cf.cv
end=time.time()
print(end-start)

#items_feature=cf.items_feature
#users_feature=cf.users_feature












