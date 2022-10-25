import math, os, torch, yaml
from utils.parsing import parse_train_args
from utils.dataset import construct_loader
from utils.training import train_epoch, test_epoch
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file

def train(args, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = math.inf
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        
        train_loss = train_epoch(args, model, train_loader, optimizer, args.device)
        print("Epoch {}: Training Loss {}".format(epoch, train_loss))
        
        val_loss = test_epoch(args, model, val_loader, device)
        print("Epoch {}: Validation Loss {}".format(epoch, val_loss))

        if scheduler:
            scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(args.log_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

if __name__ == '__main__':
    args = parse_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.restart_dir: # Directory to model_parameters.yml
        with open(f'{args.restart_dir}/model_parameters.yml') as f:
            args_old = Namespace(**yaml.full_load(f))

        model = get_model(args_old).to(device)
        state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu')) # Load the best model to continue
        model.load_state_dict(state_dict, strict=True)

    else:
        model = get_model(args).to(device)

    train_loader, val_loader = construct_loader(args)

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # record parameters
    yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, train_loader, val_loader)