#from gnldataloader import GNLDataLoader
import os
import cv2
from gnldataloader import GNLDataLoader
def main(user: int = 0):
    # Create the dataloaders of our project
    if user == 0:       # Codespace
        path_data = "/workspaces/GUNILEO/data/matching/fronts" # "data/lombardgrid_front/lombardgrid/front"
        path_labels = "/workspaces/GUNILEO/data/matching/labels" # "data/lombardgrid_alignment/lombardgrid/alignment"
    elif user == 1:     # Leo
        path_data = "data/matching/fronts" # "data/lombardgrid_front/lombardgrid/front"
        path_labels = "data/matching/labels" # "data/lombardgrid_alignment/lombardgrid/alignment"


    dataLoader = GNLDataLoader(path_labels, path_data, transform=None, debug=True)

    print(dataLoader[1:3])

def remove_bad_data(delete):
    path_data = "data/matching/fronts"
    vidlist = os.listdir(path_data)
    for vid in vidlist:
        a= vid.split(".")
        js = a[0]+".json"

        pathname = os.path.join(path_data,vid)

        cap =cv2.VideoCapture(pathname)
        """ if not cap.isOpened():
            cap.release()
            os.remove("data/matching/fronts/"+vid)
            os.remove("data/matching/labels/"+js) """

        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 75:
            if delete:
                cap.release()
                os.remove("data/matching/fronts/"+vid)
                os.remove("data/matching/labels/"+js)
            else:
                print(vid)


def count_all_vids():
    path_data = "data/matching/fronts"
    for vid in os.listdir(path_data):
        print(vid)


def dataloader_scrap():
    path_data = "data/matching/fronts"
    path_labels = "data/matching/labels"

    dataset = GNLDataLoader(path_labels, path_data, transform=None, debug=False)
    batch_size = 32
    for index in range(125+35):    # First 4000
        print(f"[DEBUG] Loading of batch {index}")
        current_batch = dataset[batch_size*index : batch_size*(index+1)]




if __name__ == "__main__":
    # remove_bad_data(0)
    # dataloader_scrap()
    path_data = "data/matching/fronts"
    vidlist = os.listdir(path_data)
    print(vidlist[32+17-1], vidlist[32+18-1], vidlist[32+19-1])
