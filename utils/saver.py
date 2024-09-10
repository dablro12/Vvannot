import json 
def save2json(save_path, save_annot_dict):
    save_path = save_path.replace('mp4', 'json') 
    
    # dictionary to json
    with open(save_path, 'w') as json_file:
        json.dump(save_annot_dict, json_file, indent=4)
    
    print(f'#### [C] Save json : {save_path}')