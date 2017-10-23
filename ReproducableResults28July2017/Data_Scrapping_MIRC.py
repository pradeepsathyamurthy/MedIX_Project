# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:57:17 2017

@author: pradeep sathyamurthy
"""
import urllib.request
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Type any keyword in 'query' variable
query = ''
data_source = "MIRC RSNA"
payload = {'firstresult':'1', 'maxresults':'5000','orderby':'1','server':'0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:20:21:22','document':query} # Query
r2 = requests.post("http://mirc.rsna.org/query", data=payload)

def Trimmed_Text(txt):
    txt_len = len(txt)
    final_txt = ''
    for x in range(txt_len):
        tmp = txt[x].getText().strip()
        final_txt = final_txt + tmp + '\n'
        
    final_txt = final_txt.strip()
    return final_txt

# Create a html file to see how it looks like
# import webbrowser
f = open('ALLTF.html', 'w')
f.write(r2.text)
f.close()

def Find_RadLex_Terms(content):
    return_val = ''
    retrieved_info = content
    for j in range(len(retrieved_info)):
        # append terms (ex) abnormal chest becomes abnormal_chest
        if len(retrieved_info[j].string.split(' ')) > 1:
            conn_term = retrieved_info[j].string.replace(' ', '_')
            if j == len(retrieved_info)-1:
                return_val += conn_term
            else:
                return_val += conn_term+", "
        else:
            conn_term = retrieved_info[j].string
            if j == len(retrieved_info)-1: 
                return_val += conn_term
            else:
                return_val += conn_term+", "
    return return_val

def Find_RadLex_Terms_Link(content):
    return_val = ''
    retrieved_info = content
    for j in range(len(retrieved_info)):
        if j == len(retrieved_info)-1:
            return_val += str(retrieved_info[j]['href'])
        else:
            return_val += str(retrieved_info[j]['href']) + ", "
            
    return return_val

# A function to return texts from raw data of RSNA TF based on categories
def RSNA_parse2(url):
    with urllib.request.urlopen(url) as url:
        sou = url.read()
        soup = BeautifulSoup(sou)

    # Those containers are needed for preprocessing 
    ddx_container = ['Ddx', 'Differential Diagnosis', 'Differential diagnosis', 'Differential', 'Differential Dx',
                    'DDx', 'DDX']
    findings_container = ['Findings (#1)', 'Findings (#2)', 'Findings']

    temp2 = soup.find_all('div', class_ = "hide")

    # Create a dataframe to store all texts
    df = pd.DataFrame(columns = ['Data Source', 'Title', 'Authors', 'Date Created', 'History', 'Findings', 
                                 'Diagnosis', 'Discussion', 'DDX', 'References', 'RadLex_Terms', 'RadLex_Terms_Link'], index = range(0,1))
    radlex_terms = ''
    radles_terms_links = ''
    for t in range(len(temp2)):
        content = temp2[t]

        if t == 0: # Retrieve author info
            num_authors = len(content.find_all('p', class_ = 'authorname'))
            authors = ''
            if num_authors == 0: # If there is no author information
                df['Authors'] = '-'
            else:
                for a in range(num_authors):
                    authors += content.find_all('p', class_ = 'authorname')[a].text.strip()+" "
                df['Authors'] = authors
                # Handle date
                dat = content.find_all(class_ = 'center')[0].text.strip().split(' ')[2]
                if dat.isdigit(): # (1) if date format looks 20160510, convert it into 5/10/16  
                    s = datetime(year=int(dat[0:4]), month=int(dat[4:6]), day=int(dat[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif '/' in dat: # (2) if date format looks 2005/10-22
                    a = dat.replace('/', '-')
                    a = "".join(a.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif '=' in dat: # (3) if data format looks 2006-08=04
                    a = dat.replace('=', '-')
                    a = "".join(a.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif '--' in dat: # (4) if data format looks 2005--1-27
                    a = dat.replace('--', '-')
                    a = "".join(a.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif len(dat.split('-')[2]) > 4: # (5) if date format looks 2006-06-16000, convert it into 6/16/06
                    a = "".join(dat.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif len(dat.split('-')[0]) > 4: # (6) if date format looks 20005-06-16, convert it into 6/16/05. [only one]
                    a = "".join(dat.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[5:7]), day=int(a[7:9]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(2005)
                elif len(dat.split('-')[2]) == 3: # (7) if data formae looks 2005-10-121 [only one]
                    a = "".join(dat.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year)
                elif len(dat.split('-')[0]) == 3: # (8) if data format looks 205-06-01 [two exist]
                    dat = dat.replace('205', '2005')
                    a = "".join(dat.split('-'))
                    s = datetime(year=int(a[0:4]), month=int(a[4:6]), day=int(a[6:8]))
                    df["Date Created"] = str(s.month)+"/"+str(s.day)+"/"+str(s.year) 
                else:
                    df["Date Created"] = dat 
        try:
            title = content.find_all('h2')[0].getText()
            if title in ddx_container:
                radlex_terms += Find_RadLex_Terms(content.find_all('a'))
                radles_terms_links += Find_RadLex_Terms_Link(content.find_all('a'))
                df['DDX'] = Trimmed_Text(content.find_all('p'))
            elif title in findings_container:
                radlex_terms += Find_RadLex_Terms(content.find_all('a'))
                radles_terms_links += Find_RadLex_Terms_Link(content.find_all('a'))
                df['Findings'] = Trimmed_Text(content.find_all('p'))
            elif title == 'Diagnosis':
                radlex_terms += Find_RadLex_Terms(content.find_all('a'))
                radles_terms_links += Find_RadLex_Terms_Link(content.find_all('a'))
                df['Diagnosis'] = Trimmed_Text(content.find_all('p'))
            elif title == 'Discussion':
                radlex_terms += Find_RadLex_Terms(content.find_all('a'))
                radles_terms_links += Find_RadLex_Terms_Link(content.find_all('a'))
                df['Discussion'] = Trimmed_Text(content.find_all('p'))
            elif title == 'History':
                radlex_terms += Find_RadLex_Terms(content.find_all('a'))
                radles_terms_links += Find_RadLex_Terms_Link(content.find_all('a'))
                df['History'] = Trimmed_Text(content.find_all('p'))
            elif title == 'References':
                ref = ''
                for r in range(len(content.find_all('li'))):
                    ref += content.find_all('li')[r].text+"\n"
                df['References'] = ref
        except IndexError:
            continue
                
        try:
            main_title = content.find_all('h1')[0].getText().strip()
            df['Title'] = main_title
        except IndexError:
            continue
            
    df['RadLex_Terms'] = radlex_terms
    df['RadLex_Terms_Link'] = radles_terms_links
    df['Data Source'] = data_source
    
    df.index = [ix for ix in range(df.shape[0])]
    
    return df

# Convert into a beautifulsoup object
bbs = BeautifulSoup(r2.text, r"lxml")

grand_df = pd.DataFrame(columns = ['Data Source', 'Title', 'Authors', 'Date Created', 'History', 'Findings', 
                                   'Diagnosis', 'Discussion', 'DDX', 'References','RadLex_Terms', 'RadLex_Terms_Link'])
for x, xml in enumerate(bbs.find_all("a", href = True)):
        dtf = RSNA_parse2(xml['href'])
        grand_df = grand_df.append(dtf)

grand_df.index = [ix for ix in range(grand_df.shape[0])]        
        
# Remove all rows that are relevant to quiz
grand_df = grand_df[grand_df.Title.str.contains("quiz") == False]


# Figure out which rows have all missing values
x = set([x for x in range(0, grand_df.shape[0])])
y = set(grand_df.index)
diff = x.difference(y)

# # Create a csv file
grand_df.to_csv('TF_'+query+'.csv')
print('TF_'+str(query)+'.csv saved!!!')

# Create another csv file which rows should be ignored to download images

# Aux
diff_df = pd.DataFrame(list(diff), columns = ["Row_IDX_to_RM"])
diff_df.to_csv("diff_df.csv")
