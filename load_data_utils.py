import os

def nyu2_paired_path(nyu2_path):
    '''
    transform images to list
    in the list, the entries are pairs of pathes of images
    e.g. (path_train1, path_label1)
    '''
    paired_path = []
    pair = []
    jpg_paths, png_paths = [], []
    for curDir, dirs, files in os.walk(nyu2_path):
        for file in files:
            file = os.path.join(curDir, file)
            if file.endswith(".jpg"):
                jpg_paths.append(file)
            else:
                png_paths.append(file)
            
    jpg_paths.sort(key=lambda x: x[:-4])
    png_paths.sort(key=lambda x: x[:-4])
    
    for jpg_path, png_path in zip(jpg_paths, png_paths):
        pair = [jpg_path, png_path]
        paired_path.append(pair)
    
    return paired_path