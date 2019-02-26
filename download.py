# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:33:12 2019

@author: admin
"""

import os

def download():
    data_dic = "dataset/raw/"
    if not os.path.exists(data_dic):
        os.makedirs(data_dic)
    else:
        if os.path.isfile(data_dic + "en"):
            en = data_dic + "en"
            return en

def download1():
    data_dic = "dataset/raw/"
    if not os.path.exists(data_dic):
        os.makedirs(data_dic)
    else:
        if os.path.isfile(data_dic + "ja"):
            en = data_dic + "ja"
            return en     
        
