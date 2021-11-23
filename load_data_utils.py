import os

def nyu2_paired_path(nyu2_path):
    '''
    transform images to list
    in the list, the entries are pairs of pathes of images
    e.g. (path_train1, path_label1)
    '''
    paired_path = []
    pair = []
    for curDir, dirs, files in os.walk(nyu2_path):
        for file in files:
            # all file names
            if file.endswith(".jpg"): # x_train
                path = os.path.join(curDir, file)
                pair.append(path)
            else:
                path = os.path.join(curDir, file)
                pair.append(path)
                paired_path.append(pair)
                pair = []
                
    return paired_path