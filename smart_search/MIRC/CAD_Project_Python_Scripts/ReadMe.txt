1. Files
 - RSNA_TF_ANALYSIS_ALL_CATEGORIES.ipynb : This Python script is used for all RadLex terms analysis under
 	all categories in RSNA. Below is what this code do
  (#) Extract all xml files and store all RadLex terms
  (#) Generate a TF-IDF matrix. 
  (#) Decision Tree generation to decice the number of clusters.
  (#) Create a hierarchical clustering and dendrogram. 
  (#) Cluster and sub-cluster analysis 
  (*) Note
  	- Don't modify any code. All we need to change is a 'query' variable under 'Mimic POST REQUEST' section.
  	For instance, if we're interested in specific keyword, cardiomegaly, then we can set query = 'cardiomegaly' and execute all scripts. Then we obtain a beautifulsoup object containing all teaching files related to 'cardiomegaly'. 
 - TF_WordCloud.ipynb : To create a Word Cloud from big_clusX.csv and all_X_vs_all.csv files. (X is 
 	the number of sub_clusters)
 - Radlex.csv : This file contains all Radlex terms and its id along with its parent id. Note that parent id
 	is used for searching category hierarchy. This file is mainly used for calculating coverage. 
 - TF_.csv : This file has 2320 rows and 13 attributes (Data Source, Title, Authors, Data Created, History, Findings, Diagnosis, Discussion, DDX, References, RadLex Terms, RadLex Terms URL, URL). 
  (*) Data Source : Either MIRC RSNA or MyPacs
  	  Title : Teaching File Title
  	  Authors : Teaching File Authors (It can be a name of hospital / institution)
  	  Data Created : The time Teaching File is created
  	  History, Findings, Diagnosis, Discussion, DDX : Five main categories of Teaching File
  	  References : Another attribute of Teaching File which is not used for text analysis
  	  RadLex Terms : A list of RadLex terms per teaching file
  	  RadLex Terms URL : A url link for each RadLex terms
  	  URL : Teaching File URL

 - radlex_term_Freq_ddx.csv : A Data Frame for aggregared Term Frequency of DDX category 
 - radlex_term_Freq_dia.csv : A Data Frame for aggregared Term Frequency of Diagnosis category 
 - radlex_term_Freq_dis.csv : A Data Frame for aggregared Term Frequency of Discussion category 
 - radlex_term_Freq_fin.csv : A Data Frame for aggregared Term Frequency of Findings category 
 - radlex_term_Freq_his.csv : A Data Frame for aggregared Term Frequency of History category 
(*) Those five csv files above have only two columns [term, frequency]

 - ddx_term_freq.csv : A Data Frame for Term Frequency of DDX category 
 - dia_term_freq.csv : A Data Frame for Term Frequency of Diagnosis category 
 - dis_term_freq.csv : A Data Frame for Term Frequency of Discussion category 
 - fin_term_freq.csv : A Data Frame for Term Frequency of Findings category 
 - his_term_freq.csv : A Data Frame for Term Frequency of History category 
 (*) Those five csv files above have x columns [x is the number of terms in each category]

2. Data Dictionary
 - CAD_Project_Python_Scripts / ReadMe.txt
 								TF_.csv
 								RSNA_TF_ANALYSIS_ALL_CATEGORIES.ipynb
 								TF_WordCloud.ipynb
 							  / TF_DF / ddx_term_freq.csv
 							  			dia_term_freq.csv
 							  			dis_term_freq.csv
 							  			fin_term_freq.csv
 							  			his_term_freq.csv
 							  / Aggr_TF / radlex_term_Freq_ddx.csv
 							  			  radlex_term_Freq_dia.csv
 							  			  radlex_term_Freq_dis.csv
 							  			  radlex_term_Freq_fin.csv
 							  			  radlex_term_Freq_his.csv

