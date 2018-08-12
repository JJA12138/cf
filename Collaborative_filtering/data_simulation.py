# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:11:44 2018

@author: 10433
"""
import numpy as np
import random

class random_data(object):
    def __init__(self,num_items,num_users,random_noise=True):
        self.random_noise=random_noise
        self.num_items=num_items
        self.num_users=num_users
        self.test_items=[random.uniform(0,1) for i in range(self.num_items)]
        self.test_items=[[i,random.uniform(0,1-i)] for i in self.test_items]
        self.test_items=[[i[0],i[1],1-sum(i)] for i in self.test_items]
        self.test_users=[[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for i in range(self.num_users)]
        self.test_items=np.array(self.test_items)
        self.test_users=np.array(self.test_users)
        self.y=np.dot(self.test_items,self.test_users.T)
        if self.random_noise:
            self.y=self.y+np.random.random([self.y.shape[0],self.y.shape[1]])/10
        self.y_copy=self.y.copy()
        for i in range(int(self.y.shape[0]*self.y.shape[1]/10)):
            items_location=random.randint(0,self.y.shape[0]-1)
            users_location=random.randint(0,self.y.shape[1]-1)
            random_value=self.y[items_location,users_location]
            while np.isnan(random_value):
                items_location=random.randint(0,self.y.shape[0]-1)
                users_location=random.randint(0,self.y.shape[1]-1)
                random_value=self.y[items_location,users_location]
            self.y[items_location,users_location]=np.nan
            
