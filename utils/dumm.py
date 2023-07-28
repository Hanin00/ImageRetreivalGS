# 'data/scenegraph_merge/merge_scenegraphs_2.pkl'
import pickle
import multiprocessing

# 데이터를 읽어오는 함수
def read_data(filename, q):
    with open(filename, 'rb') as file:
        while True:
            try:
                data = pickle.load(file)
                q.put(data)  # 읽은 데이터를 Queue에 넣음
            except EOFError:
                break

# 데이터를 처리하는 함수 (예시로 간단히 출력)
def process_data(q):
    while True:
        data = q.get()  # Queue에서 데이터 가져옴
        if data is None:
            break
        print("Processed data:", data)

if __name__ == "__main__":
    filename = "data/scenegraph_merge/merge_scenegraphs_2.pkl"
    num_processes = 4

    q = multiprocessing.Queue()
    buffer_size = 10
    buffer = multiprocessing.Manager().list([None] * buffer_size)

    # 데이터 읽기 프로세스 생성
    reader_process = multiprocessing.Process(target=read_data, args=(filename, q))
    reader_process.start()

    # 데이터 처리 프로세스 생성
    process_processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=process_data, args=(q,))
        p.start()
        process_processes.append(p)

    reader_process.join()  # 데이터 읽기 프로세스가 끝날 때까지 기다림

    # Buffer에 None을 넣어서 데이터 처리 프로세스가 종료되도록 함
    for _ in range(num_processes):
        q.put(None)

    # 데이터 처리 프로세스가 끝날 때까지 기다림
    for p in process_processes:
        p.join()

    print("All processes have finished.")




# import queue
# import threading
# import pickle

# # 파일을 작은 청크(chunk)로 나누어 읽는 함수
# def read_file_in_chunks(file_path, chunk_size):
#     with open(file_path, 'rb') as file:
#         while True:
#             data = file.read(chunk_size)
#             if not data:
#                 break
#             yield data

# # 큐에 데이터를 적재하는 함수
# def enqueue_data(queue, file_path, chunk_size):
#     for chunk in read_file_in_chunks(file_path, chunk_size):
#         queue.put(chunk)
#     queue.put(None)  # 파일 읽기가 끝났음을 알리기 위해 None을 큐에 삽입

# # 메인 함수
# def main():
#     pickle_file_path = 'data/scenegraph_merge/merge_scenegraphs_2.pkl'
#     chunk_size = 4096  # 청크 크기 (원하는 크기로 변경 가능)
#     num_threads = 2  # 사용할 스레드 개수 (조정 가능)

#     data_queue = queue.Queue()

#     # 데이터를 큐에 적재하는 스레드 시작
#     producer_thread = threading.Thread(target=enqueue_data, args=(data_queue, pickle_file_path, chunk_size))
#     producer_thread.start()

#     # 큐에서 데이터를 처리하는 스레드 시작
#     def process_data(queue):


#         while True:
#             chunk = queue.get()
#             if chunk is None:
#                 break
#             # 이곳에서 읽어온 데이터를 역직렬화하여 원래 데이터 구조로 변환
#             try:
#                 data = pickle.loads(chunk)
#                 # 원하는 작업을 수행합니다.
#                 # 예를 들면, network graphs와 graph들, 벡터값들을 각각 처리하는 등의 작업을 수행할 수 있습니다.
#                 print(len(data[0]))
#                 print(len(data[1]))
#                 print(len(data[2]))
                
#             except pickle.UnpicklingError:
#                 print("Failed to unpickle the data.")

#     consumer_threads = []
#     for _ in range(num_threads):
#         consumer_thread = threading.Thread(target=process_data, args=(data_queue,))
#         consumer_threads.append(consumer_thread)
#         consumer_thread.start()

#     # 모든 스레드가 종료될 때까지 대기
#     producer_thread.join()
#     for consumer_thread in consumer_threads:
#         consumer_thread.join()

# if __name__ == '__main__':
#     main()







# def merge_pickles(mergeFilePath):
#     pickle_dir = mergeFilePath
#     merged_file = mergeFilePath + ".pkl"

#     graph1 = []
#     graph2 = []
#     ged = []
#     for file in glob.glob(os.path.join(pickle_dir, "*.pickle")):
#         with open(file, 'rb') as f:
#             data = pickle.load(f)
#             graph1 += data[0]
#             graph2 += data[1]
#             ged += data[2]
            
#     with open(merged_file, 'wb') as f:
#         pickle.dump([graph1, graph2, ged], f)



# def datasetChk(filePath):
#     with open( filePath + ".pkl", "rb") as fr:
#         dataset = pickle.load(fr)
#     print(dataset)

#     try:        
#         for i in range(0, len(dataset), 3):
#             print(dataset[i][0].nodes(data=True))
#             print(dataset[i+1][0].nodes(data=True))
#             print(dataset[i+2][0])
#     except:
#         print("----------")

    
#     #merge 한 경우 - ged_f0_node_same_edge_1diff
#     # _unnorm / ged_f0_node_same_edge_1diff



# def samePair(Graphset1, Graphset2):
#     idxPair = []
#     for idx1, grph in enumerate (range(0, len(Graphset1), 3)):
#         '''
#         print(Graphset1[idx1][0].nodes(data=True))
#         print(Graphset1[idx1+1][0].nodes(data=True))
#         print(Graphset1[idx1+2][0])
# '''
        
#         Grphs1rpeList = list(Graphset1[idx1][0].nodes('rpe')) + list(Graphset1[idx1+1][0].nodes('rpe'))
#         Grphs1rpeList = [t[-1].tolist() for t in Grphs1rpeList]

#         Grphs1nameList = list(Graphset1[idx1][0].nodes('name')) + list(Graphset1[idx1+1][0].nodes('name'))
#         Grphs1nameList = [t[-1] for t in Grphs1nameList]

#         for idx2, grph2 in enumerate(range(0, len(Graphset2), 3)):       
          
#             '''
#             print(Graphset2[idx2][0])   
#             print(Graphset2[idx2][0].nodes(data=True))
#             print(Graphset2[idx2+1][0].nodes(data=True))
#             print(Graphset2[idx2+2][0])
# '''
#             Grphs2rpeList = list(Graphset2[idx2][0].nodes('rpe')) + list(Graphset2[idx2+1][0].nodes('rpe'))
#             Grphs2rpeList = [t[-1].tolist() for t in Grphs2rpeList]

#             Grphs2nameList = list(Graphset2[idx2][0].nodes('name')) + list(Graphset2[idx2+1][0].nodes('name'))
#             Grphs2nameList = [t[-1] for t in Grphs2nameList]


#             if ((Grphs1rpeList in Grphs2rpeList == True) & (Grphs1nameList in Grphs2rpeList == True )) : #동일한 Pair의 비교군일 경우

#                 print("graphset1")
#                 print(Graphset1[idx1][0].nodes(data=True))
#                 print(Graphset1[idx1+1][0].nodes(data=True))
#                 print(Graphset1[idx1+2][0])
#                 print(" " )
#                 print("graphset2")
#                 print(Graphset2[idx2][0].nodes(data=True))
#                 print(Graphset2[idx2+1][0].nodes(data=True))
#                 print(Graphset2[idx1+2][0])
#                 print("====================")



# def main():
#     # merge, dataChk        
#     filePath = "data/ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff"   
#     print("----------------------- norm -----------------------" )
#     merge_pickles(filePath)
#     datasetChk(filePath)

#     print("----------------------- unnorm -----------------------" )
#     filePath = filePath+"_unnorm"   
#     merge_pickles(filePath)
#     datasetChk(filePath)

#     ''' 
#     f0GraphsPath = "data/ged_f0/ged_f0_node_diff" 
#     r0GraphsPath = "data/ged_f0_rpe/ged_f0_node_diff" 


#     with open( f0GraphsPath + ".pkl", "rb") as fr:
#         f0Graphs = pickle.load(fr)

    
#     with open( r0GraphsPath + ".pkl", "rb") as fr:
#         r0Graphs = pickle.load(fr)
    

#     samePair(f0Graphs, r0Graphs)
#     sys.exit()
#     '''





# if __name__ == "__main__":
    
#     with open('data/Vidor/GEDPair/walk4_step3_ged18_480_512.pkl', 'rb') as f:
#         graphs = pickle.load(f)
#     print("graphs: ",graphs)
#     print("len(graphs): ",len(graphs))

#     sys.exit()

#     #main()
#     #sys.exit()
#     '''
#     with open('data/ged_f0/02ged_f0_node_same_edge_1diff.pkl', 'rb') as f:
#         graphs = pickle.load(f)
#     print("ged_f0/02ged_f0_node_same_edge_1diff.pkl")
#     '''
#     with open('data/ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff.pkl', 'rb') as f:
#         graphs = pickle.load(f)
#     print("ged_f0_rpe/02ged_f0_rpe_node_same_edge_1diff.pkl")
    
    
#     for idx in range(len(graphs[0])) : 
#         print(graphs[0][idx].nodes(data=True))
#         print(graphs[0][idx].edges(data=True))
#         print("            ")
#         print(graphs[1][idx].nodes(data=True))
#         print(graphs[1][idx].edges(data=True))
#         print("            ")
#         print(graphs[2][idx])
#         print("            ")
#         print(" ===================== ")
