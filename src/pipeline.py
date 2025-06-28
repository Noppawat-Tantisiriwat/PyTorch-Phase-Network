import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import re
from tqdm import tqdm

import os
import pickle
from glob import glob
from pathlib import Path
from datetime import datetime

from einops import reduce, repeat, rearrange

from PhaseNetworkClassFlexible import PhaseNetwork
import util as util


# ---------------- FUNCTIONS ----------------

def get_order(file):

    file_pattern = r".*?(\d+).*?"

    match = re.match(pattern=file_pattern, string=Path(file).name)

    if not match:

        return np.inf

    return int(match.groups()[0])


# ---------------- PROPROCESSING CLASSES ----------------

class SignalStandardization():
    """
    Class used for normalization of the signal
    """

    def __init__(self, mu=None, sigma=None):

        # Getting mu and sigma 
        self.mu = mu
        self.sigma = sigma

    
    def __call__(self, x, mu=None, sigma=None):

        logger = {
            "Shape" : []
        }

        if mu and sigma:

            output = (x - mu) / sigma # Get normalization of x

            logger["Shape"].append(output.shape) # Logging the shape of x output

            return output, logger

        else:

            output = (x - self.mu) / self.sigma

            logger["Shape"].append(output.shape)

            return output, logger


class RegressHeadingDirection():

    def __init__(self, regressWing=3):

        # Get regressWing parameter

        self.regressWing = regressWing
        # Calculate regression window length
        self.regressLength = regressWing * 2 + 1

    def __call__(self, x):

        logger = {
            "Shape" : []
        }

        D, W = x.shape # get the shape of inputs -> each input should have 2 dimensions

        r = np.zeros((D, W - self.regressWing * 2)) # Define "r" of zeros to store the regression outputs

        for d in range(D): # Loop through every element of first dimension

            for i in range(W - self.regressWing * 2): # Loop through each windows
                
                r[d, i] - self.regressForSlope(x[d, i:i + self.regressLength]) # Store the regression output to r corresponding to the original position "D" and "W"

        logger["Shape"].append(r.shape) # Logging the shape of output

        return r, logger

    def regressForSlope(self, y):
        """
        Essential Function derived from original implementation of "util.py"
        """

        l = len(y)

        x = np.arange(l) - (l - 1) / 2

        avgY = np.average(y)

        return np.dot(y - avgY, x) / np.dot(x, x)



class WindowSlidingWithHeadingRegression():

    def __init__(
        self,
        regressWing,
        predictionGap,
        windowSize
    ):

        # Getting parameters

        self.regressWing = regressWing
        self.predictionGap = predictionGap
        self.windowSize = windowSize

        # Get the regressor class
        self.regressor = RegressHeadingDirection(
            regressWing=regressWing
        )

    def __call__(self, x):

        logger = {
            "Shape" : [],
            "NumWindow" : []
        }

        m = x.shape[1]

        allPose = x[..., self.regressWing + self.regressWing: m]

        allHead, _ = self.regressor(x)

        # Generate all posible windows

        allPoseWindow = []
        allHeadWindow = []

        for i in range(m - 2 * self.regressWing - self.windowSize + 1):

            allPoseWindow.append(allPose[:, i: i + self.windowSize])
            allHeadWindow.append(allHead[:, i: i + self.windowSize])

        

        allPoseWindow = np.array(allPoseWindow)

        allHeadWindow = np.array(allHeadWindow)

        # Einstein operation to reshape the output in the corresponding dimension

        ops = "b c l -> c l b"
        self.allPoseWindow = rearrange(allPoseWindow, ops)
        self.allHeadWindow = rearrange(allHeadWindow, ops)

        # Concatenate "allPoseWindow" and "allHeadWindow" row-wise
        output = np.r_[self.allPoseWindow[np.newaxis, ...], self.allHeadWindow[np.newaxis, ...]]

        logger["Shape"].append(output.shape)
        logger["NumWindow"].append(output.shape[-1])


        return output, logger


class TimestepShiftPairing():

    def __init__(self, predictionGap):

        self.predictionGap = predictionGap

    def __call__(self, x):

        logger = {
            "Shape" : []
        }

        # Concatenate 1 -> n-1 with 2 -> n
        output = np.concatenate([x[np.newaxis, ..., :-self.predictionGap], x[np.newaxis, ..., self.predictionGap:]], axis=0)

        logger["Shape"].append(output.shape)

        return output, logger


class GetSessionLengthList():

    def __init__(self):

        pass

    def __call__(self, x, dim=None):

        if dim:

            return x.shape[dim]

        else:
            return x.shape[1]
        
    
        

class GetSessionSegmentMatrix():

    def __init__(self):
        
        pass

    def __call__(self, pairCount, SessionLengthList):

        """
        Args:

            XSession: all the windowed dataset

            SessionLengthList: List of the session length


        Returns:

            sessionSegmentMatrix: the matrix indicate the session-specific position by length
            desiged to be multiplied with phaseXY (each column is one session)

        """
        sessionCount = len(SessionLengthList)

        sessionSegmentMatrix = np.zeros([pairCount, sessionCount], dtype=float)

        sessionFirstIndex = 0

        for i in range(sessionCount):

            sessionSegmentMatrix[sessionFirstIndex: sessionFirstIndex + SessionLengthList[i], i] = 1

            sessionFirstIndex += SessionLengthList[i]


        return sessionSegmentMatrix
    
class GetCenterPose():

    def __call__(self, x):
        x.shape
        centerPose = np.zeros([])




# ---------------- DATASET CLASSES ----------------


## Dataset
## Dataset

class GaitPhasingDataset():


    def __init__(
            self,
            data_root: str,
            data_dir: str,
            img_dir: str,
            end_number_plus_one_path: str=None,
            end_session_plus_one_path: str=None,
            windowWing=None,
            regressWing=None,
            predictionGap=None,
            rotationMatrixList=None,
            meanForPhaseExtraction=None,
            sdForPhaseExtraction=None,
            trfm=None
    ):
        """
        Args:

            data_root: Directory for the data storage and the saved output.
        
            data_dir: Directorty of the dataset containing .pkl files of all the session.

            end_number_plus_one_path: Path to the session label.

            windowWing: Representative of window-size. This could be calculated by (WindowSize - 1) / 2.

            regressWing: Wing for the heading regression

            predictionGap: Overlapping of the window-sliding

            rotationMatrixList: projection of AzEq

            meanForPhaseExtraction: mean of the data normalization

            sdForPhaseExtraction: standard deviation of the data normalization


            trfm: Transformation of the dataset ***EXCLUDING*** 
                1. Window Sliding
                2. Future-Timestep Shifting

        """    


        # paths and directories


        self.data_root = data_root
        self.data_dir = data_dir
        self.img_dir = img_dir


        self.end_number_plus_one_path = end_number_plus_one_path
        self.end_session_plus_one_path = end_session_plus_one_path

        # Preprocessing configuration
        self.regressWing = regressWing
        self.predictionGap = predictionGap
        self.windowWing = windowWing
        self.window_size = 1 + windowWing * 2


        self.trfm = trfm


        ## color code setting
        self.colorCode = np.array(['#eee8aa','#1f77b4','#1f77b4','#ff7f0e','#2ca02c',
                      '#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f',
                      '#bcbd22','#17becf','#030303','#419388','#0efff8',
                      '#f80eff','#fff80e','#800eff','#ff805b','#cea15c',
                      '#cebe5c','#ff7f50','#ffe4b5','#00fa9a','#ffffe0',
                      '#008080','#000080','#da70d6','#ff69b4','#fff5ee',
                      '#696969','#800000','#7fffd4','#adff2f','#808000'])
        
    
        ## Default Transformations: -> Window Sliding, Future-timestep shifting (As last 2 layers)
        print("Getting Dataset Details ...")
        print("-"*100)
        self.raw_data, self.data_label, self.meanForPhaseExtraction, self.sdForPhaseExtraction = GaitPhasingDataset.get_session_label(
            data_dir=self.data_dir, 
            end_session_plus_one_path=self.end_session_plus_one_path, 
            save_session=True, 
            save_path=self.data_root,
            return_all_data=True
        )

        _, self.data_label_img, _,_ = GaitPhasingDataset.get_session_label(
            data_dir=self.img_dir, 
            end_session_plus_one_path=self.end_session_plus_one_path, 
            save_session=True, 
            save_path=self.data_root,
            return_all_data=True
        )
        print("-"*100)

        # self.EndNumberPlusOne = self.data_label["EndNumberPlusOne"]
        
        self.EndSessionPlusOne = pd.read_csv(self.end_session_plus_one_path)["EndSessionPlusOne"].to_list()
        
        self.SessionAmount = len(self.data_label)

        self.SubjectAmount = self.data_label["SubjectNo"].nunique()


        # Assign color to each phase.
        self.prePhaseColorList = []
        for iSession in range(self.SubjectAmount):

            if iSession == 0:

                Range = np.arange(1, self.EndSessionPlusOne[iSession])
            else:

                Range = np.arange(self.EndSessionPlusOne[iSession - 1], self.EndSessionPlusOne[iSession])

            
            for i in Range:

                self.prePhaseColorList.append(self.colorCode[iSession])

        # Assign color to each subject.
        self.srcColorList=[]
        
        for iColor in range(0,self.SubjectAmount):
            
            self.srcColorList.append(self.prePhaseColorList[self.EndSessionPlusOne[iColor]-2])

        self.srcColorList_subject=[]
        for iColor in range(0,self.SubjectAmount):
            self.srcColorList_subject.append(self.prePhaseColorList[self.EndSessionPlusOne[iColor]-2])
        

        # Assign list of training file names (consist of training and imaging)
        self.trainSubjectList = self.data_label["SessionFileName"]
        self.trainSubjectList_img = self.data_label_img["SessionFileName"]

        self.trainSubjectPath = self.data_label["SessionFilePath"]
        self.trainSubjectPath_img = self.data_label_img["SessionFilePath"]


        self.trainName = f"[{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}]"


        self.SessionLengthList = self.data_label["SessionLength"]

        # self.AllData = self.load_data()
        # self.AllData_img = self.load_data()

        if self.trfm:
            
            self.pipeline = {
                        "SignalStandardization" : SignalStandardization(
                            mu=self.meanForPhaseExtraction[[0, 1]],
                            sigma=self.sdForPhaseExtraction[[0, 1]],
                            ),
                        "Transformations" : self.trfm,
                        "WindowSlidingWithHeadingRegression": WindowSlidingWithHeadingRegression(
                            regressWing=self.regressWing,
                            predictionGap=self.predictionGap,
                            windowSize=self.window_size,
                        ),
                        "TimestepShiftPairing": TimestepShiftPairing(
                            predictionGap=self.predictionGap
                        )
                    }
            
        
        else:
            
            self.pipeline = {
                        "SignalStandardization" : SignalStandardization(
                            mu=self.meanForPhaseExtraction[[0, 1]],
                            sigma=self.sdForPhaseExtraction[[0, 1]],
                            ),
                        "WindowSlidingWithHeadingRegression": WindowSlidingWithHeadingRegression(
                            regressWing=self.regressWing,
                            predictionGap=self.predictionGap,
                            windowSize=self.window_size,
                        ),
                        "TimestepShiftPairing": TimestepShiftPairing(
                            predictionGap=self.predictionGap
                        )
                    }
            
        self.img_pipeline = {
                        "WindowSlidingWithHeadingRegression": WindowSlidingWithHeadingRegression(
                            regressWing=self.regressWing,
                            predictionGap=self.predictionGap,
                            windowSize=self.window_size,
                        ),
                        "TimestepShiftPairing": TimestepShiftPairing(
                            predictionGap=self.predictionGap
                        )
                    }
            
            
        print("Getting Tranining Data with preprocessing ...")
        print("-"*100)
        self.AllData = self.load_data(self.pipeline, for_image =False)
        
        print("Getting Image Visualization Data with preprocessing ...")
        self.AllData_img = self.load_data(self.img_pipeline, for_image =True)
        

    def load_data(self, pipeline,for_image=False):

        print("Preprocessing from pipeline ...")
        print("\t->", "\n\t-> ".join(pipeline.keys()))

        self.loading_logger = {
            "Shape": {},
            "NumWindow" : {}
        }

        all_data = []
        X_now = []
        X_next = []
        self.sessionLengthList = []
        self.srcColorList = []
        if for_image:
            data_loading_path = self.trainSubjectPath_img
            data_loading_length = len(self.trainSubjectList_img)
        else:
            data_loading_path = self.trainSubjectPath
            data_loading_length = len(self.trainSubjectList)
        for j, filepath in tqdm(enumerate(data_loading_path), total=data_loading_length):
            
            with open(filepath, "rb") as f:

                x = pickle.load(f)["data"].astype(float)

                if  not for_image:

                    x1 = x[0,:]

                   
                    # x2 = x [2,:]
                    x2 = x[1, :]
                    # y_axis = x[1,:]
                    y_axis = x[2, :]
                    x = np.concatenate([x1[np.newaxis, ...], x2[np.newaxis, ...]], axis=0)


                
                # print(x.shape)
                

            for process_name, process in pipeline.items():
               
                x, logger = process(x)

                

                if process_name == "TimestepShiftPairing":

                    thisSessionLength = x.shape[-1]
                    self.srcColorList+=[self.prePhaseColorList[j%len(self.prePhaseColorList)]]*thisSessionLength
                    self.sessionLengthList.append(thisSessionLength)



                for keys, log in logger.items():

                    if process_name in self.loading_logger[keys].keys():
                        self.loading_logger[keys][process_name].extend(log)

                    else:
                        self.loading_logger[keys][process_name] = log

            

            

        

                # print(process_name, x.shape, sep=": ")

            # print("-"*100)

            x_now = x[0]
            x_next = x[1]

            # print(x.shape)
            
            X_now.append(x_now)
            X_next.append(x_next)
            all_data.append(x)
        
        

        # print(np.sum(loading_logger["NumWindow"]["WindowSlidingWithHeadingRegression"]))
        # print(self.loading_logger["Shape"])
        # np.save("../data/log_NumWindow.npy", loading_logger["NumWindow"]["WindowSlidingWithHeadingRegression"])
        # print(all_data[0].shape)
        all_data = np.concatenate(all_data, axis=-1)
        X_now = np.concatenate(X_now, axis=-1)
        X_next = np.concatenate(X_next, axis=-1)

        pairCount = X_now.shape[-1]

        sessionCount = len(self.sessionLengthList)
        length_dict = {"SessionNo" : [], "WindowSession" : []}
        for i , length in enumerate(self.sessionLengthList):

            length_dict["SessionNo"].append(i + 1)
            length_dict["WindowSession"].append(length)

        windowed_label_dataframe = pd.DataFrame(length_dict)

        data_label_ = self.data_label.join(windowed_label_dataframe)


        
        
        # print(i + 1, length, sep=": ")

        self.EndNumberPlusOne = data_label_.groupby(["SubjectNo"]).sum()["WindowSession"].astype(int).cumsum().to_numpy()

        session_segment_matrix = GetSessionSegmentMatrix()(
            pairCount=pairCount,
            SessionLengthList=self.sessionLengthList
            )
        
        centerPose = X_now[0,..., self.window_size - 1, :]

        # print(X_now.shape)
        print("-"*100)
        # return X_now, X_next, session_segment_matrix, srcColorList
        return {
            "XY" : X_now,
            "XY2" : X_next,
            "centerPose": centerPose,
            "SessionSegmentMatrix" : session_segment_matrix,
            "srcColorList" : self.srcColorList, 

        }
    
      

            
        
    def __len__(self):

        return len(self.All_data)


    @staticmethod
    def get_session_label(
        data_dir: str, 
        end_session_plus_one_path: str, 
        save_session: bool=False, 
        save_path=None,
        return_all_data=False
    ):
        
        ## dataframe format -> Session -> SubjectNo.
        all_files = sorted(glob(os.path.join(data_dir, "*.pkl")), key=get_order)
        SessionCount = len(all_files)
        # print(SessionCount)
        EndSessionPlusOne = pd.read_csv(end_session_plus_one_path)["EndSessionPlusOne"].to_list()
        print(len(EndSessionPlusOne))

        session_label_dict = {

            "Session" : list(range(1, SessionCount + 1)),
            "SubjectNo": [],
            "SessionLength" : [],
            "SessionFileName" : [],
            "SessionFilePath" : [],

        }

        for subject_no, end_session in enumerate(EndSessionPlusOne):
            
            if subject_no < len(EndSessionPlusOne) - 1:

                subject_session_count = EndSessionPlusOne[subject_no + 1] - EndSessionPlusOne[subject_no]
            
                sess_count = [subject_no + 1] * subject_session_count

                session_label_dict["SubjectNo"].extend(sess_count)

                #print(subject_no + 1, len(sess_count), EndSessionPlusOne[subject_no], sep=" : ")
            
            else:

                subject_session_count = EndSessionPlusOne[-1] - EndSessionPlusOne[-2]

                sess_count = [subject_no + 1] * subject_session_count

                session_label_dict["SubjectNo"].extend(sess_count)

        session_label_dict["SubjectNo"]

        all_data = []
        for session_file_name in sorted(glob(os.path.join(data_dir, "*.pkl")), key=get_order):

            try:
                with open(session_file_name, "rb") as f:

                    data = pickle.load(f)["data"].astype(float)
                    
                    session_length = data.shape[-1]

                    session_label_dict["SessionFileName"].append(Path(session_file_name).stem)

                    session_label_dict["SessionFilePath"].append(session_file_name)

                    session_label_dict["SessionLength"].append(session_length)

                    all_data.append(data)

                    #print(Path(session_file_name).stem, data.shape[-1], sep=": ")
            
            except:

                print("Error at {}".format(Path(session_file_name).stem))

        # Get mean and std of the dataset -> export to the global

        all_data = np.hstack(all_data)

        mean = np.mean(all_data, axis=-1, keepdims=True)
        std = np.std(all_data, axis=-1, keepdims=True)

        print(mean[[0, 1]])
        print(std[[0, 1]])

        print(session_label_dict.keys())

        print("-------")

        for key, value in session_label_dict.items():

            print(value.__len__())


        session_label_df = pd.DataFrame(session_label_dict)

        session_label_df["EndNumberPlusOne"] = session_label_df["SessionLength"].cumsum()

        # session_label_df["calculated_length"] = session_label_df["cumulative_sum_length"].diff().fillna(session_label_df["cumulative_sum_length"][0]).astype(np.uint)

        if save_session:

            print("Saving ... ")
            ## saving the generated data to .csv file
            session_label_df.to_csv(os.path.join(save_path, "SessionLabel.csv"))

            print(f"File \"SessionLabel.csv\" was saved to {save_path}")

        if return_all_data:

            return all_data, session_label_df, mean, std
        
        else:
            return session_label_df, mean, std






