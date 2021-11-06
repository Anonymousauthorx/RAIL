from marvin.trainers.IL import ILTrainer
from marvin.trainers.RL import RLTrainer
from marvin.utils.trainer_parameters import parser

if __name__ == "__main__":
    # args are the input hyperparameters and details that the user sets
    args = parser.parse_args()
    if args.rl:
        t = RLTrainer(args)
    else:
        t = ILTrainer(args)

    t.train()
