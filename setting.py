import os
class Config():
    def __init__(self,mode="maintenance",session_i=0):
        self.mode = mode # MODES: "maintenance" | "dynamic"
        self.session_i=session_i # session index 0 | 1
        self.data_path = "D:\zengz\phd1\EOG_deep_learning\preprocessing\subject_data_npy"  # the dir of preprocessed data
        self.fs = 100  # sample frequency
        self.segmentation_length = .25 # in second
        self.overlap_ratio = .5 # in percentage
        # self.discard_length=.25 # in second
        self.num_class = 5  # the number of classes
        self.num_sub = 50  # the number of subjects
        # base dir settings
        # self.os_dir="/home/fanjiahao/open_dataset_deep_learning_method"
        # self.data_dir = "/home/fanjiahao/TNNLS_work/raw_data_txt/PR_Set" # the dir of raw data
        # self.tb_log_dir="./tb_logs"
        # self.model_path="./models"
        self.output_path = "./outputs"
    # @property # 将方法变成属性
    # def data_path(self):
    #     return os.path.join(self.save_path,f"{self.mode}_STFT")
    # @property
    # def fold_n(self):
    #     return 6 if self.mode=="dynamic" else 2
    # @property
    # def seg_num(self):
    #     return 29 if self.mode=="dynamic" else 5

