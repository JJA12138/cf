# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:15:03 2018

@author: 10433
"""
import numpy as np
import random
import matplotlib.pyplot as plt
class collaborative_filtering(object):
    def __init__(self,num_feature,a=0.01):
        self.num_feature=num_feature
        self.items_feature=None
        self.users_feature=None
        self.pred=None
        self.alpha=a
        self.Y=None
        self.R=None 
        self.num_iteration=0
        self.print_loss=True
        self.regular=0
        self.cv_rate=0
        self.train_loss=None
        self.cv_loss=None
        self.cv=None
        self.loss_ploy=True
        
    #生成特征
    def create_feature(self,num_items,num_users):
        #self.items_feature=np.random.rand(num_items,self.num_feature)
        self.items_feature=np.zeros([num_items,self.num_feature])
        self.items_feature[:,0]=1
        
        self.users_feature=np.random.rand(num_users,self.num_feature)
    
    #设置验证集,注意选取验证集时避免交叉集为空，而且尽量选取密集的位置   
    def set_cv(self,cv_rate):
        self.cv_rate=cv_rate
        num_cv=int(self.cv_rate*(self.Y.shape[0]*self.Y.shape[1]-np.sum(np.isnan(self.Y))))
        self.cv=np.zeros([self.Y.shape[0],self.Y.shape[1]])
        for i in range(num_cv):
            items_location=random.randint(0,self.Y.shape[0]-1)
            users_location=random.randint(0,self.Y.shape[1]-1)
            random_value=self.Y[items_location,users_location]
            while random_value==0:
                items_location=random.randint(0,self.Y.shape[0]-1)
                users_location=random.randint(0,self.Y.shape[1]-1)
                random_value=self.Y[items_location,users_location]  
            self.cv[items_location,users_location]=self.Y[items_location,users_location]
            self.Y[items_location,users_location]=0
            
            
            
    def output_score(self,score):
        score=score
        print(score)
        
    #设置参数
    def set_parameter(self,parameter):
        self.parameter=parameter
        for para in self.parameter:
            setattr(self,para,self.parameter[para])
    
    #训练
    def fit(self,Y):
        #创建特征
        self.create_feature(Y.shape[0],Y.shape[1])
        self.Y=Y
        #创建
        self.R=abs(np.isnan(self.Y).astype(int)-1)
#        self.R_users_num=self.R.sum(1)
#        self.R_items_num=self.R.sum(0)
#        for i in range(self.num_feature-1):
#            self.R_users_num=np.column_stack((self.R_users_num,self.R.sum(1)))
#            self.R_items_num=np.column_stack((self.R_items_num,self.R.sum(0)))        
        
        self.Y[np.isnan(self.Y)]=0
        #创建测试集
        self.set_cv(self.cv_rate)
        cv_R=(self.cv>0).astype(int)
        self.R=self.R+(self.cv==0).astype(int)
        self.R[self.R<2]=0
        self.R[self.R==2]=1
        self.R_users_num=self.R.sum(1)
        self.R_items_num=self.R.sum(0)
        for i in range(self.num_feature-1):
            self.R_users_num=np.column_stack((self.R_users_num,self.R.sum(1)))
            self.R_items_num=np.column_stack((self.R_items_num,self.R.sum(0)))        
        
        
        
        
        self.pred=np.dot(self.items_feature,self.users_feature.T)
        self.diff=(self.pred-self.Y)*self.R
        self.loss=np.sum(self.diff**2)/2+self.regular*np.sum(self.items_feature*self.items_feature)/2+\
                        self.regular*np.sum(self.users_feature*self.users_feature)/2
        
        if self.loss_ploy:
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            num_i=[]
            train_loss_plot=[]
            cv_loss_plot=[]
            ax.plot(num_i,train_loss_plot,c='b',marker='.',linewidth=1.0,label="train_loss")
            ax.plot(num_i,cv_loss_plot,c='r',marker='.',linewidth=1.0,label="cv_loss")
            plt.legend()
            plt.title('loss')        
        for i in range(self.num_iteration):               
            self.items_feature=(1-self.alpha*self.regular)*self.items_feature-\
                self.alpha*np.dot(self.diff,self.users_feature)/self.R_users_num
            self.users_feature=(1-self.alpha*self.regular)*self.users_feature-\
                self.alpha*np.dot(self.diff.T,self.items_feature)/self.R_items_num
            
            self.pred=np.dot(self.items_feature,self.users_feature.T)
            
            self.diff=self.pred-self.Y
            self.diff=(self.pred-self.Y)*self.R
            self.loss=np.sum(self.diff**2)/2+self.regular*np.sum(self.items_feature*self.items_feature)/2+\
                        self.regular*np.sum(self.users_feature*self.users_feature)/2    
            #打印train损失值
            self.train_loss=self.loss/len(self.R[self.R==1])
            cv_diff=(self.pred-self.cv)*cv_R
            self.cv_loss=np.sum(cv_diff**2)/len(cv_R[cv_R==1]) 
            
            print(u'训练集损失:')
            self.output_score(self.train_loss)
            print(u'测试集损失:')
            self.output_score(self.cv_loss)  
            
            if self.loss_ploy:
                num_i.append(i)
                train_loss_plot.append(self.train_loss)
                cv_loss_plot.append(self.cv_loss)
                if (i % 99)==0:
                    pass
                    ax.plot(num_i,train_loss_plot,c='b',marker='.',linewidth=1.0,label="train_loss")
                    ax.plot(num_i,cv_loss_plot,c='r',marker='.',linewidth=1.0,label="cv_loss")
                    plt.pause(0.001)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
