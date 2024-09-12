import json
import numpy as np
import os 

def convert_numpy_types(data):
    """
    Recursively converts numpy types to native Python types for JSON serialization.
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)  # Convert numpy int to Python int
    elif isinstance(data, np.floating):
        return float(data)  # Convert numpy float to Python float
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    else:
        return data  # Return other types as is

def save2json(save_path, save_annot_dict):
    # directory
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    
    save_path = save_path.replace(save_path.split('.')[-1], 'json') 
    
    # Convert any numpy types to native Python types
    save_annot_dict_serializable = convert_numpy_types(save_annot_dict)
    
    # Save to JSON file
    with open(save_path, 'w') as json_file:
        json.dump(save_annot_dict_serializable, json_file, indent=4)

    print(f'#### [C] Save json : {save_path}')