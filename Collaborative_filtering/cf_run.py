# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:37:47 2018

@author: 10433
python环境：3.6.4



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












