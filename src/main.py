import argparse 
  
import warnings
warnings.filterwarnings('ignore')

from config import run_config, opt_run_config
from utils import utils

from data.data import get_dataloaders

from training.train import train

import optuna
from optuna.trial import TrialState

def objective(trial = None):
    cfg = run_config 

    #Set optuna params
    if trial != None:
        cfg.encoder_name = trial.suggest_categorical('encoder_name',opt_run_config.encoder_names)
        cfg.model_name = trial.suggest_categorical('model_name',opt_run_config.model_names)
        #cfg.channels = trial.suggest_categorical('channels',opt_run_config.channels)

        cfg.save_name = str(trial.number)

    utils.config_run_folder(cfg)
    utils.set_seed(run_config.SEED)

    run = utils.init_experiment()
    run['cfg'] = utils.dict_from_module(cfg)

    train_dataloader, val_dataloader = get_dataloaders(cfg)

    return train(train_dataloader, val_dataloader, run, cfg, trial)


def main(args):

    if args.mode == 'opt':
        
        study = optuna.create_study(
            study_name='flood', 
            direction="maximize",
            pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=7),
            sampler=optuna.samplers.TPESampler()
            )

        study.optimize(objective, n_trials=10)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        objective()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="run",choices=['run','opt'])

    args = parser.parse_args()
    
    main(args)