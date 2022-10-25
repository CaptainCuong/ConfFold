import os, torch, yaml, time
from collections import defaultdict
from models.conftree import ConfTreeModel

def get_model(args):
	return ConfTreeModel(args, args.num_conv_layers)

def get_optimizer_and_scheduler(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

