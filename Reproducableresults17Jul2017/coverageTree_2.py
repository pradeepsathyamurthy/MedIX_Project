import os
os.chdir("D:\Courses\MedIX_Project\Reproducableresults17Jul2017")

# get list of Mirc terms
# (list of RadLex terms used in MIRC DDX)
f = open("ddx_rsna.csv")
rawTermData = f.read()
termData = rawTermData.replace("_", " ")
mpddxTerms = termData.split(",")
mpddxTerms_lower = [ words.lower() for words in mpddxTerms ]
setMPTerms = set(mpddxTerms_lower)
f.close()

# get the RadLex tree from file
# input format is each row has a term, id, parent_id
# (exclude first two rows)
radTreeLines = open("Radlex.csv").readlines()
filteredTreeLines = [ x for x in radTreeLines if not x.startswith("#") ]
rawTreeData = [ line.strip().split("\t") for line in filteredTreeLines[2:] ] # List of radlex terms & SynonymID & ParentID

# we need to be able to look up a node and get the path from the root
# we will iterate through dictionary of nodeID->parentID
# RID1 is the root
# parent given as RID0 means this node is the root
dictTree = { line[1] : line[2] for line in rawTreeData } # Key = SynonymRID; Value = ParentsRID
dictIDToName = { line[1] : line[0] for line in rawTreeData } # Key = SynonymRID; Value = Radlex Terms
dictNameToID = { line[0] : line[1] for line in rawTreeData } # Key = Radlex Terms; Value = SynonymRID

# Now that we know the Radlex Term in DDX with its Synonym ID 
# And Radlex.csv having its parent RID for a particulat Synonym ID
# We need to traverse to the absolute path of the Radlex Tree Browser
# getpath() function does this functionality

# runs recursively, accumulating the path
# returns empty list when path error
def getPath(nodeID, dictTree, accum):
    if nodeID == "RID1":
        return accum + [nodeID] # We r just concatenating the nodeid with an empty list if nodeif is RID1
    if nodeID not in dictTree: return [] # If the nodeif we search is not in dicttree or if it is not one of the key in dictTree, then we return a empty list back  
    return getPath(dictTree[nodeID], dictTree, accum + [nodeID])

# create a test for if this node is a leaf
setParents = set([ line[2] for line in rawTreeData ]) # This is a dictionary with all parentId of radlex terms from RADLEX dictionary

# create a list of paths, one for each leaf node
pathList = []
longestPath = 0
# Nodeid below is a SynonymID in Radlex
for nodeName, nodeID, parentID in rawTreeData:
    if not nodeID in setParents:# checking if the synonymid/nodeid is not a part of parentid
        rawPath = getPath(nodeID, dictTree, []) # if node id is not a part of parentid then we call get path
        if rawPath == []:
            continue
        rawPath = list(reversed(rawPath))
        pathList.append( [dictIDToName[i] for i in rawPath ] )
        if len(rawPath) > longestPath: longestPath = len(rawPath)


#==============================================================================
#
# # Prady remove this later, this shows how a Rawpath gets generated from a single node id
# nodeName = 'Helium contrast'
# nodeID = 'RID11593'
# parentID = 'RID11589'
# if not nodeID in setParents:
#     rawPath = getPath(nodeID, dictTree, [])
#     rawPath = list(reversed(rawPath))
#     print(rawPath)
# 
#==============================================================================
    
# for each term in the mypacs term list, get its path,
# pad and write to file
ofile = open("radtree_Prady_19Jul.csv", "w")

header = [ "L" + str(i) for i in range(longestPath) ]
ofile.write(",".join(header) + ",size,covered\n")
print("preparing to write paths ", len(pathList))
for path in pathList:
    leafName = path[-1]
    isCovered = leafName in setMPTerms
    # print(isCovered)
    # print(["0","1"][isCovered])
    paddedList = path + [""]*(longestPath - len(path))
    pathText = ",".join(paddedList)
    ofile.write(pathText + ",1,")
    ofile.write(["0","1"][isCovered])
    ofile.write("\n")
ofile.close()
    
    
