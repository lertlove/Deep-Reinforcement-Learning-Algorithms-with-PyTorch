import json

class Experiment():
    def __init__(self, config):

        self.experiment_name = config.experiment_name
        self.results_dir = config.results_dir

        self.ctu_width = config.ctu_width
        self.ctu_height = config.ctu_height
        self.config = config

        # initialize
        self.GOPSize = 1
        self.LCUHeight = 64
        self.LCUWidth = 64
        self.frameRate = 24
        self.meta_data = ""
        self.picHeight = 144
        self.picWidth = 176
        self.targetBitrate = 1000000
        self.totalFrames = 100

        # self.file_to_save_data_results = config.file_to_save_data_results
        # self.file_to_save_results_graph = config.file_to_save_results_graph
        # self.overwrite_existing_results_file = config.overwrite_existing_results_file
        # self.save_model = config.save_model
        # self.save_and_load_meta_state = config.save_and_load_meta_state

    def start_experiment(self, message):
        print(f"start_experiment {message}")
        # start_experiment {'GOPSize': 1, 'LCUHeight': 64, 'LCUWidth': 64, 'command': 'start_experiment', 'frameRate': 24, 'meta_data': 'video_filename', 'picHeight': 144, 'picWidth': 176, 'targetBitrate': 1000000, 'totalFrames': 5}

        self.GOPSize = message["GOPSize"]
        self.LCUHeight = message["LCUHeight"]
        self.LCUWidth = message["LCUWidth"]
        self.frameRate = message["frameRate"]
        self.meta_data = message["meta_data"]
        self.picHeight = message["picHeight"]
        self.picWidth = message["picWidth"]
        self.targetBitrate = message["targetBitrate"]
        self.totalFrames = message["totalFrames"]
        # self.experiment_name = config.experiment_name
        # print(f"Experiment experiment_name: {self.experiment_name}")

        reply = f"Environment: have got start_experiment = {message}"
        return reply
