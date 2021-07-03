class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False
        
        self.load_model_file = None
        self.save_and_load_meta_state = True
        self.interval_save_result = None
        self.interval_save_policy = None
        self.use_ssd_insteadof_mse = False
        self.trials = 50
        self.training_episode_per_eval = 10
        self.experiment_name = "exp_xxx"
        self.results_dir = f"{self.experiment_name}_results"
        self.model_dir = f"{self.experiment_name}_models"
