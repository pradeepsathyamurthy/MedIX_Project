# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 00:30:57 2017

@author: pradeep sathyamurthy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import requests
import nltk
#import string
import re
#import math
#import operator
import os
os.chdir("D:\Courses\MedIX_Project\Reproducableresults17Jul2017")
#import csv
#from IPython.display import Image 
from itertools import compress
#from pylab import rcParams
#from collections import Counter
from sklearn.cluster import KMeans 
from nltk.corpus import stopwords
#from sklearn import neighbors
from sklearn import tree
#from nltk.stem.porter import *
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
#from sklearn.metrics import silhouette_score
from bs4 import BeautifulSoup
#from sklearn import metrics
#import pydotplus
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Load Basic Function

# A function that strips all html tags
def striphtml(dat):
    p = re.compile(r'<.*?>') # Regular expression for all html tags
    return p.sub(' ', dat)

# A function that modifies multiple spaces into a single space
def removespace(dat):
    p = re.compile(' +')
    return p.sub(' ', dat)

def get_filtered(sen):
    sentence = sen
    lower = sentence.lower()
    no_punctuation = re.sub(r'[^\w\s]','',lower)
    no_numbers = ''.join([w for w in no_punctuation if not w.isdigit()])
    tokens = nltk.word_tokenize(no_numbers)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

def unlist(ls):
    output = ''
    for v in ls:
        output = output + " " + v
    return output.strip()

# Euclidean Distance
def distance(p0, p1):
    return np.linalg.norm(p0 - p1)

# Collect all titles in a list
title_list = []

# A function to calculate the frequency of terms and return a wordcount dictionary
def RSNA_parse2(url, threshold, radlex = False, image = False, category = False):
    with urllib.request.urlopen(url) as url:
        sou = url.read()
        soup = BeautifulSoup(sou)

    global kv_pairs
    global kv_pairs2
    global kv_pairs_cat
    global title_list
    
    big_title = soup.find_all('h1')[0].text # title of TF
    title = soup.find_all('h2') # This is a title for each category (ddx, findings...)
    title[0] = 'Document'
    
    history = ''
    discussion = ''
    diagnosis = ''
    findings = ''
    ddx = ''
    
    # Those are the categories I want to remove
    kill_title = ["References", "Files", "Keywords", "Powerpoint Presentation", "Tumor Board Presetations", 
                 "Comments", "Rex shunt evaluation", "Ileocolic Intussusception", "Radiology", "Management",
                 "Clinical Presentation", "Interactive Dataset I", "Pathology", "Follow-up Clinical History",
                 "Instructions", "Document", "Heading", 
                  "AP/lateral radiographs of the right knee. AP/lateral radiographs of both knees two weeks later.",
                 'Images (#1)', 'Images 3 and 4', 'Image 3', 'Section Heading', 'Interactive Dataset II', 
                 'Images (#2)', 'Images 5 and 6', 'Images 4-6', 'Image 7', 'Slides', 'Images 1 and 2']
    
    # Those containers are needed for preprocessing 
    ddx_container = ['Ddx', 'Differential Diagnosis', 'Differential diagnosis', 'Differential', 'Differential Dx',
                    'DDx']
    findings_container = ['Findings (#1)', 'Findings (#2)']

    # Create a title container
    title_bag = []
    
    for t in title:
        clean_title = striphtml(str(t))
        clean_title = clean_title.strip()
        if clean_title not in title_list: # append into the list if title does not exist
            title_list.append(clean_title)

        if clean_title in kill_title:
            continue
        # Preprocessing some categories. For instance, Ddx, DDX, DDx, are all synonyms and can be merged
        elif clean_title in ddx_container:
            clean_title = 'DDX'
            title_bag.append(clean_title)
        elif clean_title in findings_container:
            clean_title = 'Findings'
            title_bag.append(clean_title)
        else:
            title_bag.append(clean_title)   
            
    if image == True:
        temp2 = soup.find_all('body')
    else:
        #cnt = 0
        temp2 = soup.find_all('div', class_ = "hide")

        for i in range(len(title_bag)): 
            if i < (len(temp2)-1): 
                if radlex: # Find all terms that linked to RadLex  
                    content = temp2[i+1].find_all('a')
                    for j in range(len(content)):
                        # append terms (ex) abnormal chest becomes abnormal_chest
                        if len(content[j].string.split(' ')) > 1:
                            conn_term = content[j].string.replace(' ', '_')
                        else:
                            conn_term = content[j].string
                        
                        if title_bag[i] == 'History':
                            history = history + conn_term + ' '
                        elif title_bag[i] == 'Findings':
                            findings = findings + conn_term + ' '
                        elif title_bag[i] == 'Discussion':
                            discussion = discussion + conn_term + ' '
                        elif title_bag[i] == 'Diagnosis':
                            diagnosis = diagnosis + conn_term + ' '
                        elif title_bag[i] == 'DDX':
                            ddx = ddx + conn_term + ' '
                            
                        if category: # Create a dictionary based on category
                            if title_bag[i] not in kv_pairs_cat.keys():
                                kv_pairs_cat[title_bag[i]] = [content[j].string]
                            else:
                                kv_pairs_cat[title_bag[i]].append(content[j].string)
                        else:
                            if conn_term not in kv_pairs.keys():
                                kv_pairs[conn_term] = 1
                            else:
                                kv_pairs[conn_term] += 1

                else: # Find all terms
                    content = temp2[i].text
                    # get_filtered removed all stopwords and punctuations. 
                    clean_content = get_filtered(removespace(content))
                    for j in range(len(clean_content)):
                        # Skip wordcount if a term is a title
                        if clean_content[j] in clean_title:
                            pass
                        else:
                            if clean_content[j] not in kv_pairs2.keys():
                                kv_pairs2[clean_content[j]] = 1
                            else:
                                kv_pairs2[clean_content[j]] += 1             

        if radlex:
            kv_pairs = dict((k,v) for k, v in dict(kv_pairs).items() if v >= threshold)
        else:
            kv_pairs2 = dict((k,v) for k, v in dict(kv_pairs2).items() if v >= threshold)
            
    return [big_title, history, findings, discussion, diagnosis, ddx]

# Mimic POST REQUEST

# Type any keyword in 'query' variable
query = ''
payload = {'firstresult':'1', 'maxresults':'5000','orderby':'1','server':'0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:20:21:22','document':query} # Query
r2 = requests.post("http://mirc.rsna.org/query", data=payload)
#print(r2.text)

# Extract xml files
# Convert into a beautifulsoup object
bbs = BeautifulSoup(r2.text, "lxml")
corpus = []

# Extract all xml files
kv_pairs = {}
kv_pairs2 = {}
kv_pairs_cat = {}
for xml in bbs.find_all("a", href = True):
    ret = RSNA_parse2(xml['href'], 1, radlex = True, category = True)
#     ret_container += ret
    corpus.append(ret)
# Prady
print(len(corpus)) # Total teaching files are 2548
print(corpus[2548]) # e.g. 'Pyloric Stenosis' this is a teaching file title
# Corpus will have all the Radlex terms used in all category for the whole MIRC website
# Each element in the Corpus list is a combined Radlex terms used in one particulat category
# Order of elements are big_title, history, findings, discussion, diagnosis, ddx

# Make lists of terms
history_words = []
findings_words = []
discussion_words = []
diagnosis_words = []
ddx_words = []

title_ls = []

# here we are trying to seperate consolidated Radlex terms for each category respectively 
for i in range(len(corpus)):
    # By creating a dictionary I can keep track of the index of teaching file
    
    # append title
    title_ls.append(corpus[i][0])
    
    if corpus[i][1] is not '':  
        temp_dic = {}
        temp_dic[i] = corpus[i][1]
        history_words.append(temp_dic)
    if corpus[i][2] is not '':
        temp_dic = {}
        temp_dic[i] = corpus[i][2]
        findings_words.append(temp_dic)
    if corpus[i][3] is not '':
        temp_dic = {}
        temp_dic[i] = corpus[i][3]
        discussion_words.append(temp_dic)
    if corpus[i][4] is not '':
        temp_dic = {}
        temp_dic[i] = corpus[i][4]
        diagnosis_words.append(temp_dic)
    if corpus[i][5] is not '':
        temp_dic = {}
        temp_dic[i] = corpus[i][5]
        ddx_words.append(temp_dic)

# Now create a list of words
# here we create the list of all Radlex terms used in one particular caegory
his_w = []
for k in range(len(history_words)):
    his_w.append(list(history_words[k].values())[0].strip())

find_w = []
for k in range(len(findings_words)):
    find_w.append(list(findings_words[k].values())[0].strip())
    
dis_w = []
for k in range(len(discussion_words)):
    dis_w.append(list(discussion_words[k].values())[0].strip())
    
dia_w = []
for k in range(len(diagnosis_words)):
    dia_w.append(list(diagnosis_words[k].values())[0].strip())

ddx_w = []
for k in range(len(ddx_words)):
    ddx_w.append(list(ddx_words[k].values())[0].strip())

# Create word frequency dictionary
# This is just a word count, it produce a dictionary with words and number of times it has been occured in each category
ddx_freq = {}
for tf in kv_pairs_cat['DDX']:
    # Split word corpus 
    temp_split = tf.strip().split(' ')
    for term in temp_split:
        clean_term = term.lower().replace('_', ' ')
        if clean_term in ddx_freq.keys():
            ddx_freq[clean_term] = ddx_freq[clean_term] + 1
        else:
            ddx_freq[clean_term] = 1

his_freq = {}
for tf in kv_pairs_cat['History']:
    # Split word corpus 
    temp_split = tf.strip().split(' ')
    for term in temp_split:
        clean_term = term.lower().replace('_', ' ')
        if clean_term in his_freq.keys():
            his_freq[clean_term] = his_freq[clean_term] + 1
        else:
            his_freq[clean_term] = 1
            
dis_freq = {}
for tf in kv_pairs_cat['Discussion']:
    # Split word corpus 
    temp_split = tf.strip().split(' ')
    for term in temp_split:
        clean_term = term.lower().replace('_', ' ')
        if clean_term in dis_freq.keys():
            dis_freq[clean_term] = dis_freq[clean_term] + 1
        else:
            dis_freq[clean_term] = 1
            
dia_freq = {}
for tf in kv_pairs_cat['Diagnosis']:
    # Split word corpus 
    temp_split = tf.strip().split(' ')
    for term in temp_split:
        clean_term = term.lower().replace('_', ' ')
        if clean_term in dia_freq.keys():
            dia_freq[clean_term] = dia_freq[clean_term] + 1
        else:
            dia_freq[clean_term] = 1

fin_freq = {}
for tf in kv_pairs_cat['Findings']:
    # Split word corpus 
    temp_split = tf.strip().split(' ')
    for term in temp_split:
        clean_term = term.lower().replace('_', ' ')
        if clean_term in fin_freq.keys():
            fin_freq[clean_term] = fin_freq[clean_term] + 1
        else:
            fin_freq[clean_term] = 1

# Trying to write the whole word count obtained from Dictionary in a csv file
# This is what we call as a Radlex Term Frequency file
radlex_term_freq_df_ddx = pd.Series(ddx_freq, index=ddx_freq.keys())
radlex_term_freq_df_ddx.to_csv("radlex_term_Freq_ddx_prady.csv", )
radlex_term_freq_df_his = pd.Series(his_freq, index=his_freq.keys())
radlex_term_freq_df_his.to_csv("radlex_term_Freq_his_prady.csv", )
radlex_term_freq_df_dis = pd.Series(dis_freq, index=dis_freq.keys())
radlex_term_freq_df_dis.to_csv("radlex_term_Freq_dis_prady.csv", )
radlex_term_freq_df_dia = pd.Series(dia_freq, index=dia_freq.keys())
radlex_term_freq_df_dia.to_csv("radlex_term_Freq_dia_prady.csv", )
radlex_term_freq_df_fin = pd.Series(fin_freq, index=fin_freq.keys())
radlex_term_freq_df_fin.to_csv("radlex_term_Freq_fin_prady.csv", )

# Term Frequency Matrix Generator
# This is where we create the matrix
# Where we have the Teaching file in column-1 and all Unique Radlex terms in Row-1
# We then calculate how many times a particular Radlex term has occured or been used in a particular teaching file for a particular category
# Function to calculate TF-IDF
def TFIDF_generator(w, word_list):
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(w)
    
    df = pd.DataFrame(X.toarray(), columns= vectorizer.get_feature_names())
    
    # Get the index of History category
    idx = [list(word_list[x].keys())[0] for x in range(len(word_list))]
    
    df.index = [title_ls[i] for i in idx]
    
    # Now I have a term-frequency matrix with TF index!!!!
    TF = df.T
    
    # Calculate the number of TFs and terms
    #numTerms = TF.shape[0]
    nTF = TF.shape[1]
    
    # create IDF
    DF = np.array([(TF!=0).sum(1)]).T
    
    # Create a matrix with all entries to be the number of TFs
    NMatrix=np.ones(np.shape(TF), dtype=float)*nTF
    np.set_printoptions(precision=2,suppress=True,linewidth=120)

    # Convert each entry into IDF values
    # Note that IDF is only a function of the term, so all rows will be identical.
    IDF = np.log2(np.divide(NMatrix, DF))

    # Calculate TF-IDF
    TFIDF = TF * IDF
    
    return TF, TFIDF

# History
his_tf, his_tfidf = TFIDF_generator(his_w, history_words)
his_tf.T.to_csv("his_term_freq_prady.csv")

# Findings
find_tf, find_tfidf = TFIDF_generator(find_w, findings_words)
find_tf.T.to_csv("find_term_freq_prady.csv")

# Discussions
dis_tf, dis_tfidf = TFIDF_generator(dis_w, discussion_words)
dis_tf.T.to_csv("dis_term_freq_prady.csv")

# Diagonosis
dia_tf, dia_tfidf = TFIDF_generator(dia_w, diagnosis_words)
dia_tf.T.to_csv("dia_term_freq_prady.csv")

# DDX
ddx_tf, ddx_tfidf = TFIDF_generator(ddx_w, ddx_words)
ddx_tf.T.to_csv("ddx_term_freq_prady.csv")

# Now that we have a Radlex Term Frequency Matirx for each category
# We would like to apply a unsupervised learning to find if it is forming any patterns
# For this the Clustering algorithm would help
# In order to find the optimum cluster size we are looking for Decision tree
def DTree_Calculate_Accr(df, z):
    acc_ls = []
    
    for t in range(2, 300, 2):
        # Figure out the membership of each cluster
        membership = fcluster(z, t, criterion='maxclust')
        temp_ddx_tf = df.T
        temp_ddx_tf['Membership'] = membership.tolist()  

        # Remove all cases where the total count is 1
        if np.sum(temp_ddx_tf.Membership.value_counts() >= 1):
            # Extract cluster membership that does have more than one observation
            list_a = [x for x in range(1,t+1)]
            fil = temp_ddx_tf['Membership'].value_counts() != 1
            # Sort by dataframe index
            fil.sort_index(inplace=True)
            cm_needed_ls = list(compress(list_a, fil))

            # Boolean index to filter out
            haha = [True if x in cm_needed_ls else False for x in temp_ddx_tf['Membership']]

            # Our final df
            final_dt = temp_ddx_tf[haha]

            x_train, x_test, tar_train, tar_test = cross_validation.train_test_split(final_dt,
                            final_dt['Membership'],train_size=.8, stratify=final_dt['Membership'], random_state = 33)
        else:
            x_train, x_test, tar_train, tar_test = cross_validation.train_test_split(temp_ddx_tf,
                temp_ddx_tf['Membership'],train_size=.8, stratify=temp_ddx_tf['Membership'], random_state = 33)

        treeclf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=30)
        treeclf = treeclf.fit(x_train, tar_train)

        # Accuracy on Test Set
        acc = treeclf.score(x_test, tar_test)
        acc_ls.append(acc)
        print('Tree ', t, ' has ',acc,' accuracy!')
        
#         #treecm = confusion_matrix(tar_test, treepreds_test)    
        
    return acc_ls, temp_ddx_tf, membership

# Scree plot for visualizing the accuracy
Z = linkage(ddx_tf.T, 'ward')
acc, td, memb = DTree_Calculate_Accr(ddx_tf, Z)
plt.plot([x for x in range(2, 300, 2)], acc)
plt.title('Scree Plot Accuracy')
plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.show()
# from above tree output and accuracy graph we could see until 50 branches we see 95% of accuracy for DDX
# So we can consider optimum size for clustering DDX terms would be 50

# Thus performing Hierarchical Clustering & Dendogram interpretation with the optimum cluster size obtained above
# DDX Category
# generate the linkage matrix
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=50,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

# Figure out the membership of each cluster
membership = fcluster(Z, 50, criterion='maxclust')
temp_tf = ddx_tf.T
temp_tf['Membership'] = membership.tolist()

for c in range(1, 51):
    temp_tf['Membership'] = membership.tolist() 
    temp_tf.loc[temp_tf['Membership'] != c, 'Membership'] = 0
    temp_tf.loc[temp_tf['Membership'] == c, 'Membership'] = 1
    
    # subset for storing teaching file title for 'c' cluster
#     subset_tf = temp_tf[temp_tf.Membership == 1]
    
#     for x, content in enumerate(list(subset_tf.index)):
#         if x == 0:
#             tf_writer.writerow(str(3)+","+str(len(subset_tf.index))+","+content+"\n")
#         else:
#             print(content)
#             tf_writer.writerow(",,"+content+"\n")
    
    x_train, x_test, tar_train, tar_test = cross_validation.train_test_split(ddx_tf.T,
        temp_tf['Membership'], train_size=.8, random_state = 33)

    treeclf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=25)
    treeclf = treeclf.fit(temp_tf.ix[:, temp_tf.columns != 'Membership'], temp_tf['Membership'])
    
    # retrieve sig terms in decision tree
    sig_feature_idx = list(set([x for x in list(treeclf.tree_.feature) if x != -2]))
    # print terms
    print("Cluster "+str(c)+" : ")
    print(list(ddx_tf.index[sig_feature_idx]))
    
# f.close()

# Category Analysis
# For DDX
def DTree_Calculate_Accr2(df, z):
    # Figure out the membership of each cluster
    membership = fcluster(z, 50, criterion='maxclust')
    temp_ddx_tf = df.T
    temp_ddx_tf['Membership'] = membership.tolist()  
    return temp_ddx_tf, membership

ddf, membership = DTree_Calculate_Accr2(ddx_tf, Z)

# Seperate into 5 big clusters
# We make use of the dendogram to segregate this
# From dendo gram we see the color for each group of clusters
# For example in the dendogram I received, I obtained green color groupomg with 3 clusters, so Big Cluster-1 is if size 3
# However, we have Big Cluster-2 with 4 colors in Red and 1 in Blue, thus totally 5 clusters are grouped together as part of this second big cluster
# Similarly we generate 5 Bigclusters for Green, Red, Steel Green, Blue and Purple
# Big clusters
big_clus1_df = ddf[ddf.Membership <= 4]
big_clus2_df = ddf[ddf.Membership >= 5]
big_clus2_df = big_clus2_df[big_clus2_df.Membership <= 9]
big_clus3_df = ddf[ddf.Membership >= 10]
big_clus3_df = big_clus3_df[big_clus3_df.Membership <= 20]
big_clus4_df = ddf[ddf.Membership >= 21]
big_clus4_df = big_clus4_df[big_clus4_df.Membership <= 22]
big_clus5_df = ddf[ddf.Membership >= 23]

idx = 1

def Separate_Clusters(df):
    global idx
    big_clus_freq = {}
    for row in range(df.shape[0] - 1):
        temp_row = df.ix[row,:]
        for a in range(df.shape[1] - 2):
            if temp_row.index[a] not in big_clus_freq.keys():
                big_clus_freq[temp_row.index[a]] = temp_row[a]
            else:
                big_clus_freq[temp_row.index[a]] += temp_row[a]   
    big_clus = pd.Series(big_clus_freq, index=big_clus_freq.keys())
    big_clus.to_csv("big_clus"+str(idx)+".csv")
    print("big_clus"+str(idx)+".csv file saved!")
    idx += 1

Separate_Clusters(big_clus1_df)
Separate_Clusters(big_clus2_df)
Separate_Clusters(big_clus3_df)
Separate_Clusters(big_clus4_df)
Separate_Clusters(big_clus5_df)
idx = 1

# Big Cluster Analysis
tm1 = [1 if x <= 4 else 0 for x in ddf.Membership]
tm2 = [1 if x >= 5 and x <= 9 else 0 for x in ddf.Membership]
tm3 = [1 if x >= 10 and x <= 20 else 0 for x in ddf.Membership]
tm4 = [1 if x >= 21 and x <= 22 else 0 for x in ddf.Membership]
tm5 = [1 if x >= 23 else 0 for x in ddf.Membership]

cat = 'ddx'

big_idx = 1
def BigClusterAnal(tf, tm): # tf = his_tf, ddx_tf, etc
    global big_idx
    membership = fcluster(Z, 50, criterion='maxclust')
    temp_tf = tf.T
    temp_tf['Membership'] = membership.tolist() 
    
    # Re-index Membership
    temp_tf.Membership = tm
    
    # Build a decision tree
    treeclf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=25)
    treeclf = treeclf.fit(temp_tf.ix[:, temp_tf.columns != 'Membership'], temp_tf['Membership'])

    with open("haha_"+str(big_idx)+".dot", 'w') as f:
        f = tree.export_graphviz(treeclf, out_file=f)

    os.unlink("haha_"+str(big_idx)+".dot")

    #dot_data = tree.export_graphviz(treeclf, out_file=None) 
    #graph = pydotplus.graph_from_dot_data(dot_data) 
    #graph.write_pdf("haha_"+str(big_idx)+".pdf")   
    
    term_imp = dict(zip(temp_tf.columns, treeclf.feature_importances_))
    term_imp = sorted(term_imp.items(), key = lambda x: x[1], reverse = True) 
    
    term_imp_df = pd.DataFrame.from_dict(term_imp, orient = "columns")
    term_imp_df.columns = ['Term', 'Attr Score']
    
    term_imp_df['Attr Score'] = round((100 * (term_imp_df['Attr Score'] / np.max(term_imp_df['Attr Score']))))
    
    term_imp_df.to_csv(cat+"_"+str(big_idx)+"_vs_all.csv")
    
    print(cat+"_"+str(big_idx)+"_vs_all.csv file saved!!")
    big_idx += 1

BigClusterAnal(ddx_tf, tm1)
BigClusterAnal(ddx_tf, tm2)
BigClusterAnal(ddx_tf, tm3)
BigClusterAnal(ddx_tf, tm4)
BigClusterAnal(ddx_tf, tm5)

big_idx = 1
ddf.shape
# ddf.shape is (1453, 1175)

# Deeper Analysis
# I am not covering it now

# Cluster Evaluation
# DDX Category
# This isn't working
# Try later

# DDX 
membership = fcluster(Z, 50, criterion='maxclust')
temp_tf = ddx_tf.T
temp_tf['Membership'] = membership.tolist()
# Remove first two irrelevant columns
temp_tf = temp_tf.ix[:,2:]

ddf = temp_tf
three_most_freq_terms = []

# Find Clusteroids
# Iterate through all clusters

clusteroids_list = []
size_cluster = len(np.unique(ddf['Membership']))

for c in range(1,51):
    subset_cluster = ddf[ddf['Membership'] == c]
    
    spec_tf = [True if x in subset_cluster.index else False for x in ddx_tf.columns]
    term_lists = [ddx_w[i] for i, x in enumerate(spec_tf) if x]
    wc = {}
    for term_list in term_lists:
        wwc = term_list.split(' ')
        for term in wwc:
            if term == 'trauma' or term == 'tumor' or term == 'surgery' or term == 'inguinal_hernia':
                continue
            else:
                if term.lower() not in wc.keys():
                    wc[term.lower()] = 1
                else:
                    wc[term.lower()] = wc[term.lower()] + 1

    count_asdict = sorted(wc.items(), key = lambda x: x[1], reverse = True) 
    #print the 3 most freq terms in DDX category
#     print(str(c))
    clean_count_asdict = count_asdict
    three_most_freq_terms.append(count_asdict[:8])

# Calculate within-ratio for each cluster
for c in range(1, 51):
    subset_cluster = ddf[ddf['Membership'] == c]
    km = KMeans(n_clusters=1, max_iter=1, random_state=33)
    q = km.fit(subset_cluster.ix[:,subset_cluster.columns != "Membership"])
    print("Cluster"+str(c))
    print(q.inertia_ / (subset_cluster.shape[0] - 1))

# RadLex Category Tracker
example = []
for terms in three_most_freq_terms:
    for term in terms:
        if term[0] == "":
            example.append("abscess")
        elif term[0] == "polyp":
            example.append("lateral")
        else:
            example.append(term[0])

# Remove all underline
example = [tm.replace("_", " ") for tm in example]

# Modify some terms in a list
flag = 0
for x, tt in enumerate(example):
    if flag == 0 and tt == "fossa":
        example[x] = "fourth ventricle"
        flag = 1
        print("CHANGE")
    elif flag == 1 and tt == "fossa":
        example[x] = "fourth ventricle"
        flag = 2
        print("CHANGE one more")

# Re-define variable
radlex_term_category_list = pd.read_csv("Radlex.csv", encoding='latin-1', sep='\s*\t\s*')

term_category_dict = {}
clus_idx = 1
for x, term in enumerate(example):
    hier = ""
    terms = ""

    target = radlex_term_category_list[radlex_term_category_list["Name or Synonym"].values == term]
    hier += target["Name or Synonym"].values+" => "
    terms += target["Name or Synonym"].values+","
    # Loop through until it reaches to the root

    try:
        while target["Parent RID"].values[0] != "RID1":
            parent_rid = target["Parent RID"].values[0]
            target = radlex_term_category_list[radlex_term_category_list["RID"] == parent_rid]
#             print(target)
            hier += target["Name or Synonym"].values+" => "
            terms += target["Name or Synonym"].values+","

        terms_list = unlist(terms).strip().split(",")

        terms_list = terms_list[1:-1]

        if terms_list[-1] not in term_category_dict.keys():
            term_category_dict[terms_list[-1]] = {}
            for term in terms_list:
                if term not in term_category_dict[terms_list[-1]].keys():
                    term_category_dict[terms_list[-1]][term] = 1
                else:
                    term_category_dict[terms_list[-1]][term] += 1
        else:
            for term in terms_list:
                if term not in term_category_dict[terms_list[-1]].keys():
                    term_category_dict[terms_list[-1]][term] = 1
                else:
                    term_category_dict[terms_list[-1]][term] += 1

        if x % 8 == 0:
            print("============CLUSTER "+str(clus_idx)+"==============")
            print("===============")
            clus_idx += 1

        print(hier)
    except IndexError:
        continue

# Find Clusteroids
# Iterate through all clusters

clusteroids_list = []
size_cluster = len(np.unique(ddf['Membership']))

for c in range(1,51):
    subset_cluster = ddf[ddf['Membership'] == c]
    
    spec_tf = [True if x in subset_cluster.index else False for x in dis_tf.columns]
    term_lists = [dis_w[i] for i, x in enumerate(spec_tf) if x]
    wc = {}
    for term_list in term_lists:
        wwc = term_list.split(' ')
        for term in wwc:
            if term == 'trauma' or term == 'tumor' or term == 'surgery' or term == 'inguinal_hernia' or term == 'bones':
                continue
            else:
                if term.lower() not in wc.keys():
                    wc[term.lower()] = 1
                else:
                    wc[term.lower()] = wc[term.lower()] + 1

    count_asdict = sorted(wc.items(), key = lambda x: x[1], reverse = True) 
    #print the 3 most freq terms in DDX category
#     print(str(c))

    clean_count_asdict = count_asdict
    print(str(c)+"!++++++++++++++++++++++++++++")
    print(count_asdict)
    print("++++++++++++++++++++++++++")
    three_most_freq_terms.append(count_asdict[:8])

