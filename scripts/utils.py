import json
import pickle
import tqdm

def create_id_cap_mapping(data_path, predicted_data_path, out_path):
    data = open(data_path)
    predicted_data = open(predicted_data_path)

    data = json.load(data)
    with open(predicted_data_path, 'rb') as handle:
        predicted_data = pickle.load(handle)

    id_caption = {}
    file_name_2_id = {}

    for i in data['images'] :
        file_name = i['file_name']
        id = i['id']
        file_name_2_id[file_name] = id

    dict_list = []

    for image_name,caption in predicted_data.items():
        id = file_name_2_id[image_name+'.jpg']
        tmp_dict = {'image_id': id, 'caption': caption}
        dict_list.append(tmp_dict)

    out_file = open(out_path, "w") 
    json.dump(dict_list, out_file)

    print('Done')

    return

if __name__ == '__main__':
    create_id_cap_mapping('/3d_data/datasets/coco/annotations/captions_val2014.json', '/3d_data/retreiver/base/predictions.pickle', '/3d_data/retreiver/base/predictions.json')