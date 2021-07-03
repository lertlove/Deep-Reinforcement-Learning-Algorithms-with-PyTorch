import pickle
import os
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../../')))
print(sys.path)
from agents import Trainer
from utilities.data_structures.Config import Config

PICKLE_FILE = "../rc_results_bak/rc_openImageNet_300_1-64_64-round_1-ep_300.pkl" if len(sys.argv) <= 1 else sys.argv[1]
print(f"PICKLE_FILE - {PICKLE_FILE}")

config = Config()
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True

trainer = Trainer.Trainer(config,None)

# modify pickle file
results = trainer.load_obj(PICKLE_FILE)

if hasattr(results, 'keys'):
    print(f"agents {results.keys()}")
else:
    results = {"Passive_DQN":results}
    trainer.save_obj(results,PICKLE_FILE)

trainer.environment_name = "Rate Control"
trainer.visualise_preexisting_results(data_path=PICKLE_FILE)
