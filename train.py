import wandb
import torch
import numpy as np
import data_loader.data_loaders as module_data
import modules.loss as module_loss
import modules.metric as module_metric
import models.seq2seq as module_arch

from utils.build_utils import build_logger, build_data_loader, build_model, build_optimizer
from utils.configuration import Flags, Configuration
from trainer import Trainer
from utils import to_dict, flatten

# fix random seeds for reproducibility
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(args):
    print(args.config)

    configuration = Configuration(args.config)
    config = configuration.get_config()

    # WandB Resume Handling
    resume = False
    if args.resume is not None:
        resume = True

    base = config['base']

    # Logger
    logger = build_logger('train')

    # build_data_loader(config)
    
    model = build_model(config)
    logger.info(model)

    # build_optimizer(config)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, args['loss'])
    metrics = [getattr(module_metric, met) for met in args['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if 'module' not in args['optimizer']:
        module = torch.optim
    else:
        module = args['optimizer']['module']
    optimizer = args.init_obj('optimizer', module, trainable_params)


    lr_scheduler = args.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if not base['debug']:
        # Send Config to WandB
        wandb.init(config=flatten(to_dict(args.config)), **flatten(config['wandb']), resume=resume)

        # Send model to WandB
        wandb.watch(model)

    # Trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=args,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = Flags()
    args = parser.get_parser().parse_args()
    main(args)
