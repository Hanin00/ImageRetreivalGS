import pickle

import sys

def extract_and_save_data(input_filename, output_filename):
    # pickle 파일 읽기
    with open(input_filename, 'rb') as file:
        data = pickle.load(file)

    # data[0]만 추출하여 새로운 리스트로 저장
    extracted_data = data[0]
    
    # 새로운 pkl 파일로 저장
    with open(output_filename, 'wb') as file:
        pickle.dump(extracted_data, file)

if __name__ == "__main__":
    input_filename = "data/scenegraph_merge/merge_scenegraphs_3.pkl"
    output_filename = "data/scenegraph_merge/merge_scenegraphs_3_onlyGraphs.pkl"

    extract_and_save_data(input_filename, output_filename)
    print("Data[0] extracted and saved to", output_filename)