# added by Prady
import os
os.chdir("D:\Courses\MedIX_Project\Reproducable_Results")


# get list of My Pacs terms
# (list of RadLex terms used in My Pacs DDX)
cat = "ddx"

# Open term frequency and clean it
term_freq = open("radlex_term_Freq_ddx.csv").readlines()

# print(term_freq)
clean_term_freq = [term_freq[i].strip() for i in range(len(term_freq))]
# Convert into dictionary form
dict_clean_term_freq = {}
for term in clean_term_freq:
    temp = term.split(',')
    dict_clean_term_freq[temp[0]] = temp[1]


# Commented by Prady
f = open(cat+"_rsna.csv") # Prady - I think this can be created from TF_.csv file by comparing the terms in column Radlex Term with each of the Category
rawTermData = f.read()
termData = rawTermData.replace("_", " ")
mpddxTerms = termData.split(",")
len(mpddxTerms)
setMPTerms = set(mpddxTerms) # Commented by Prady, used to select only the unique value, thus removing duplication
len(setMPTerms)
f.close()

# get the RadLex tree from file
# input format is each row has a term, id, parent_id
# (exclude first two rows)
radTreeLines = open("Radlex.csv").readlines()
filteredTreeLines = [ x for x in radTreeLines if not x.startswith("#") ]
rawTreeData = [ line.strip().split("\t") for line in filteredTreeLines[2:] ]

# we need to be able to look up a node and get the path from the root
# we will iterate through dictionary of nodeID->parentID
# RID1 is the root
# parent given as RID0 means this node is the root
dictTree = { line[1] : line[2] for line in rawTreeData }
dictIDToName = { line[1] : line[0] for line in rawTreeData }
dictNameToID = { line[0] : line[1] for line in rawTreeData }

# runs recursively, accumulating the path
# returns empty list when path error
def getPath(nodeID, dictTree, accum):
    print("inside the getpath function::")
    print("nodeid is >>>",nodeID)
    if nodeID == "RID1":
        return accum + [nodeID]
    if nodeID not in dictTree: return []
    return getPath(dictTree[nodeID], dictTree, accum + [nodeID]) # There is a problem in this recursive return loop

#==============================================================================
# def unlist(ls):
#     output = ''
#     for v in ls:
#         output = output + " " + v
#     return output.strip()
#==============================================================================

# create a test for if this node is a leaf
setParents = set([ line[2] for line in rawTreeData ])

# create a list of paths, one for each leaf node
pathList = []
longestPath = 0
anatomical_entity_count = 0

# top category name 
l1_name = "temporal entity"

for nodeName, nodeID, parentID in rawTreeData:
    if not nodeID in setParents:
        rawPath = getPath(nodeID, dictTree, [])
        #print(rawPath)
        # sample rawPath = ['RID6257', 'RID6256', 'RID39102', 'RID6', 'RID1']
        # For counting purpose
        try:
            L1_name = [dictIDToName[i] for i in rawPath ][-2]
            if L1_name == l1_name:
                anatomical_entity_count += 1
        except IndexError:
            continue
        if rawPath == []:
            continue
        rawPath = list(reversed(rawPath))
        pathList.append( [dictIDToName[i] for i in rawPath ] )
        if len(rawPath) > longestPath: 
            longestPath = len(rawPath)
            #print("Longest path : "+unlist([dictIDToName[i]+" -> " for i in rawPath])+" \# and its length : "+str(longestPath))

print(pathList)

# Removing all the comments done by Sungmin

print("The number of "+l1_name+" : "+str(anatomical_entity_count))

# # for each term in the mypacs term list, get its path,
# # pad and write to file
ofile = open("radtree_"+cat+"_rsna.csv", "w")

# # Count the number of anatomical entity covered by tree
num_covered = 0
freq_covered = 0

header = [ "L" + str(i) for i in range(longestPath)]
ofile.write(",".join(header) + ",size,covered\n")
print("preparing to write paths ", len(pathList))
print(pathList)
for path in pathList:
    leafName = path[-1]
    # Add exception handling so that we can avoid strange situation
    try:
        freq = dict_clean_term_freq[leafName]
        # Increment by one if path[1] == anatomical entity
        if path[1] == l1_name and int(freq) != 0:
            num_covered += 1
            freq_covered += int(freq)
    except KeyError:
        continue
    isCovered = leafName in setMPTerms
    #print(isCovered)
    #print(["0","1"][isCovered])
    # Find frequency of leafName
    try:
        freq = dict_clean_term_freq[leafName]
        ofile.write(str(freq))
    except KeyError:
        ofile.write("0")
    paddedList = path + [""]*(longestPath - len(path))
    pathText = ",".join(paddedList)
    ofile.write(pathText + ",1,")
    ofile.write(["0","1"][isCovered])
    ofile.write("\n")
ofile.close()

print("Num covered by tree : "+str(num_covered))
print("Freq : "+str(freq_covered))
print("Percentage of coverage : "+str(float(num_covered / anatomical_entity_count)))
# # Load 150 RadLex terms
#three_most_freq_terms = open("tmft.csv").readlines()
#clean_freq_term = three_most_freq_terms[0].split('\n')



