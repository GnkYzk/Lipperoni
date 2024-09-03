import os
import dlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchnlp.encoders import LabelEncoder

debug_dl = True

class GNLDataLoader(Dataset):
    """Creates a dataloader for the Lipsync Project"""
    face_detector = dlib.get_frontal_face_detector()
    landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

    alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
    encoder = LabelEncoder(alphabet, reserved_labels=['unknown'], unknown_index=0)
    CROPMARGIN = 20

    def __init__(self, labels_path: str, data_path: str, transform = None, train_test_percent: int = 75, debug: bool = False) -> None:
        """
        Creates a dataset given the path to the labels and the image directory

        Parameters:
            - `labels_path`: the path to the `csv` file containing the labels;
            - `images_dir`: the path to the directory with the images;
            - `transform`: states whether a transformation should be applied to the images or not.
        """
        super().__init__()
        self.debug: bool = debug

        if self.debug:
            print(f"[DEBUG] The data dir has{' ' if os.path.isdir(data_path) else ' not '}been recognized")
            print(f"[DEBUG] The label dir has{' ' if os.path.isdir(labels_path) else ' not '}been recognized")

        self.data_path, self.labels_path = data_path, labels_path
        self.data_dir, self.labels_dir = sorted(os.listdir(data_path)), sorted(os.listdir(labels_path))
        self.transform = transform


    def __len__(self) -> int:
        """
        Returns the length of the data/labels folder

        Returns:
            - `length` (`int`): the length of the data/labels folder
        """
        return len(self.data_dir)


    def __getitem__(self, index: int, straight: bool = False) -> tuple[torch.Tensor, list[str]]:
        """
        Get the ith item(s) in the dataset

        Parameters:
            - `index`: the index of the image that must be retrieven.

        Returns:
            - (`item`, `label`) (`tuple[torch.Tensor, torch.Tensor]`): the item in the ith position in the dataset, along with its label.
        """

        if self.debug:
            print(f"[DEBUG] Index of the dataloader: {index}")
            print(f"[DEBUG] Data folder: {self.data_dir[index]}")
            print(f"[DEBUG] Labels folder: {self.labels_dir[index]}")

        datas = [self.data_dir[index]] if type(self.data_dir[index]) != list else self.data_dir[index]
        labels = [self.labels_dir[index]] if type(self.labels_dir[index]) != list else self.labels_dir[index]

        to_return = []

        for ind, item in enumerate(datas):
            to_return.append((self.__load_video__(item), self.__load_label__(labels[ind])))


        '''return (
            [self.__load_video__(data_piece) for data_piece in datas],
            [self.__load_label__(label_piece) for label_piece in labels]
        )'''

        # print(f"{len(to_return)}")
        return tuple(to_return)


    def __load_video__(self, video_path: str) -> torch.Tensor:
        """
        Loads a video from the dataset given its path

        Parameters:
            - `video_path`: the path of the video that must be loaded

        Returns:
            - `video` (`torch.Tensor`): the video as a PyTorch's `Tensor`
        """
        label_name = video_path[:-3] + "json"
        video_path = os.path.join(self.data_path, video_path)
        cap = cv2.VideoCapture(video_path)
        if self.debug:
            #print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[DEBUG] Trying to open the video at path {video_path}")
        to_return = np.ndarray(shape =(75,100,150))

        # homog, prev_frame = True, None

        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = cap.read()
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')  # Format to 8-bit image. 'int8' doesn't seem to do the job either
            '''if self.debug:

                cv2.imshow("Frame", gframe)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite("/workspace/GUNILEO/tests/gframe001.jpg", gframe)

                prev_frame = gframe.shape if prev_frame == None else prev_frame
                homog = False if prev_frame != gframe.shape else True
                print(gframe.shape, homog)'''

            facedetect = self.face_detector(gframe)

            #HAVE A CHECK IF THE FACE IS FOUND OR NOT

            try:
                face_landmarks = self.landmark(gframe, facedetect[0])
                xleft = face_landmarks.part(48).x - self.CROPMARGIN
                xright = face_landmarks.part(54).x + self.CROPMARGIN
                ybottom = face_landmarks.part(57).y + self.CROPMARGIN
                ytop = face_landmarks.part(50).y - self.CROPMARGIN

                mouth = gframe[ytop:ybottom, xleft:xright]
                mouth = cv2.resize(mouth, (150, 100))

                mean = np.mean(mouth)
                std_dev = np.std(mouth)
                mouth = (mouth - mean) / std_dev
                to_return[i] = mouth
            except IndexError:
                # naughty_boys.add(video_path)
                # naughty_labels.add("data/matching/labels/" + label_name)
                to_return[i] = np.zeros((100, 150))

        cap.release()

        if self.debug:
            print(f"[DEBUG] Video {video_path} opened")
            print(f"[DEBUG] Shape of video: {to_return.shape}")

        to_return = np.array([to_return])

        return torch.tensor(to_return, dtype=torch.float32)


    def __load_label__(self, label_path: str) -> torch.Tensor:
        """
        Loads a label from the dataset given its path

        Parameters:
            - `label_path`: the path of the label that must be loaded;

        Returns:
            - `label` (`torch.Tensor`): the label as a PyTorch's tensor
        """

        encoding = [
            {"b":"bin","l":"lay","p":"place","s":"set"},
            {"b":"blue","g":"green","r":"red","w":"white"},
            {"a":"at","b":"by","i":"in","w":"with"},
            "letter",
            {"z":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"},
            {"a":"again","n":"now","p":"please","s":"soon"}
            ]

        code = label_path.split(".")[0].split("_")[-1]

        sentence = []
        for i, letter in enumerate(code):
            corresponding_dict = encoding[i]
            next = letter if corresponding_dict == "letter" else corresponding_dict[letter]
            sentence = sentence + [x for x in next]
     
        # Adapting the labels to be all of equal size
        

        enl = self.encoder.batch_encode(sentence)
        enl = enl.type(torch.FloatTensor)
        if self.debug: print(f"[DEBUG] Label: {enl}\n[DEBUG] Sentence: {sentence}\n[DEBUG] Length: {len(sentence)}\n")
        return enl
