# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:59:49 2017

@author: prade
"""
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Reading the data
def generate_word_cloud(file_name):
    print(file_name)
    ddff2 = pd.read_csv(file_name,  names = ["Obs", "Term", "Attr Score"])
    ddff2 = ddff2[ddff2.Term != 'Term']
    #ddff2.head()
    #ddff2.tail()
    # Converting the column Attr Score to numeric, second statement can be used when there are any missing values
    ddff2['Attr Score'] = pd.to_numeric(ddff2['Attr Score'])
    #ddff2['Attr Score'] = pd.to_numeric(ddff2['Attr Score'], errors='coerce')
    pd.options.display.float_format = '{:,.0f}'.format # this will display all flot to integer type, as pandas have no direct type casting to int
    # Converting the dataframe to dictionary
    tf_dict = ddff2.set_index('Term')['Attr Score'].to_dict()
    sorted(tf_dict)
    plt.figure(figsize=(10,6))
    plt.axis("off"); 
    return plt.imshow(WordCloud(max_words=200).generate_from_frequencies(tf_dict))

# Cluster-1
generate_word_cloud("all_9_vs_all.csv")

# Cluster-2
generate_word_cloud("all_8_vs_all.csv")

# Cluster-3
generate_word_cloud("all_7_vs_all.csv")

# Cluster-4
generate_word_cloud("all_6_vs_all.csv")

# Cluster-5
generate_word_cloud("all_5_vs_all.csv") # Unable to get the result for this cluster

# Cluster-6
generate_word_cloud("all_4_vs_all.csv") # Unable to get the result for this cluster

# Cluster-7
generate_word_cloud("all_3_vs_all.csv")

# Cluster-8
generate_word_cloud("all_2_vs_all.csv") # Unable to get the result for this cluster

# Cluster-9
generate_word_cloud("all_1_vs_all.csv")





