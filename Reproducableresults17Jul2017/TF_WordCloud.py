# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:59:49 2017

@author: prade
"""
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import os
os.chdir("D:\Courses\MedIX_Project\Reproducableresults17Jul2017")


# Reading the data
def generate_word_cloud(file_name):
    print(file_name)
    ddff2 = pd.read_csv(file_name,  names = ["Obs", "Term", "Attr Score"])
    print("Pass1")    
    ddff2 = ddff2[ddff2.Term != 'Term']
    print("Pass2")        
    #ddff2.head()
    #ddff2.tail()
    # Converting the column Attr Score to numeric, second statement can be used when there are any missing values
    ddff2['Attr Score'] = pd.to_numeric(ddff2['Attr Score'])
    print("Pass3")
    #ddff2['Attr Score'] = pd.to_numeric(ddff2['Attr Score'], errors='coerce')
    pd.options.display.float_format = '{:,.0f}'.format # this will display all flot to integer type, as pandas have no direct type casting to int
    print("Pass4")    
    # Converting the dataframe to dictionary
    tf_dict = ddff2.set_index('Term')['Attr Score'].to_dict()
    print("Pass5")    
    sorted(tf_dict)
    print("Pass6")
    print(tf_dict)    
    plt.figure(figsize=(10,6))
    print("Pass7")    
    plt.axis("off"); 
    print("Pass8") 
    return plt.imshow(WordCloud(max_words=50).generate_from_frequencies(tf_dict))

# Cluster-1
generate_word_cloud("ddx_1_vs_all.csv")

# Cluster-2
generate_word_cloud("ddx_2_vs_all.csv")

# Cluster-3
generate_word_cloud("ddx_3_vs_all.csv")

# Cluster-4
generate_word_cloud("ddx_4_vs_all.csv")

# Cluster-5
generate_word_cloud("ddx_5_vs_all.csv")







