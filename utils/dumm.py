import os, sys
import glob
import pickle


def merge_pickles(mergeFilePath):
    pickle_dir = mergeFilePath
    merged_file = mergeFilePath + ".pkl"

    graph1 = []
    graph2 = []
    ged = []
    for file in glob.glob(os.path.join(pickle_dir, "*.pickle")):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            graph1 += data[0]
            graph2 += data[1]
            ged += data[2]
            
    with open(merged_file, 'wb') as f:
        pickle.dump([graph1, graph2, ged], f)



def datasetChk(filePath):
    with open( filePath + ".pkl", "rb") as fr:
        dataset = pickle.load(fr)
    print(dataset)

    try:        
        for i in range(0, len(dataset), 3):
            print(dataset[i][0].nodes(data=True))
            print(dataset[i+1][0].nodes(data=True))
            print(dataset[i+2][0])
    except:
        print("----------")

    
    #merge 한 경우 - ged_f0_node_same_edge_1diff
    # _unnorm / ged_f0_node_same_edge_1diff



def samePair(Graphset1, Graphset2):
    idxPair = []
    for idx1, grph in enumerate (range(0, len(Graphset1), 3)):
        '''
        print(Graphset1[idx1][0].nodes(data=True))
        print(Graphset1[idx1+1][0].nodes(data=True))
        print(Graphset1[idx1+2][0])
'''
        
        Grphs1rpeList = list(Graphset1[idx1][0].nodes('rpe')) + list(Graphset1[idx1+1][0].nodes('rpe'))
        Grphs1rpeList = [t[-1].tolist() for t in Grphs1rpeList]

        Grphs1nameList = list(Graphset1[idx1][0].nodes('name')) + list(Graphset1[idx1+1][0].nodes('name'))
        Grphs1nameList = [t[-1] for t in Grphs1nameList]

        for idx2, grph2 in enumerate(range(0, len(Graphset2), 3)):       
          
            '''
            print(Graphset2[idx2][0])   
            print(Graphset2[idx2][0].nodes(data=True))
            print(Graphset2[idx2+1][0].nodes(data=True))
            print(Graphset2[idx2+2][0])
'''

            Grphs2rpeList = list(Graphset2[idx2][0].nodes('rpe')) + list(Graphset2[idx2+1][0].nodes('rpe'))
            Grphs2rpeList = [t[-1].tolist() for t in Grphs2rpeList]

            Grphs2nameList = list(Graphset2[idx2][0].nodes('name')) + list(Graphset2[idx2+1][0].nodes('name'))
            Grphs2nameList = [t[-1] for t in Grphs2nameList]


            if ((Grphs1rpeList in Grphs2rpeList == True) & (Grphs1nameList in Grphs2rpeList == True )) : #동일한 Pair의 비교군일 경우

                print("graphset1")
                print(Graphset1[idx1][0].nodes(data=True))
                print(Graphset1[idx1+1][0].nodes(data=True))
                print(Graphset1[idx1+2][0])
                print(" " )
                print("graphset2")
                print(Graphset2[idx2][0].nodes(data=True))
                print(Graphset2[idx2+1][0].nodes(data=True))
                print(Graphset2[idx1+2][0])
                print("====================")



def main():
    
    # merge, dataChk        
    filePath = "data/ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff"   
    print("----------------------- norm -----------------------" )
    merge_pickles(filePath)
    datasetChk(filePath)

    print("----------------------- unnorm -----------------------" )
    filePath = filePath+"_unnorm"   
    merge_pickles(filePath)
    datasetChk(filePath)

    ''' 
    f0GraphsPath = "data/ged_f0/ged_f0_node_diff" 
    r0GraphsPath = "data/ged_f0_rpe/ged_f0_node_diff" 


    with open( f0GraphsPath + ".pkl", "rb") as fr:
        f0Graphs = pickle.load(fr)

    
    with open( r0GraphsPath + ".pkl", "rb") as fr:
        r0Graphs = pickle.load(fr)
    

    samePair(f0Graphs, r0Graphs)
    sys.exit()
    '''





if __name__ == "__main__":
    
    #main()
    #sys.exit()
    '''
    with open('data/ged_f0/02ged_f0_node_same_edge_1diff.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print("ged_f0/02ged_f0_node_same_edge_1diff.pkl")
    '''
    with open('data/ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print("ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff.pkl")
    
    
    for idx in range(len(graphs[0])) : 
        print(graphs[0][idx].nodes(data=True))
        print(graphs[0][idx].edges(data=True))
        print("            ")
        print(graphs[1][idx].nodes(data=True))
        print(graphs[1][idx].edges(data=True))
        print("            ")
        print(graphs[2][idx])
        print("            ")
        print(" ===================== ")
