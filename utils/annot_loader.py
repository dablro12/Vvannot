import simdjson 

def load_annot(annot_path:str):
    with open(annot_path, 'r') as f:
        annot = simdjson.load(f)
    return annot


# if __name__ == "__main__":
#     annot_path = "app/annotations/demo2.mp4.json"
#     annot = load_annot(annot_path)
#     print(annot)
