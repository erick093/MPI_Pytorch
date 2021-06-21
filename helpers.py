import torch


def save_checkpoint(state, epoch, model_name, checkpoint_dir, best_model_dir, is_best=True):
    """ Save a train checkpoint in checkpoint_dir, save best model in best_model_dir"""
    f_path = checkpoint_dir + 'checkpoint_{}.pt'.format(model_name)
    torch.save(state, f_path)



def load_checkpoint(checkpoint_fpath, model, optimizer):
    """ load a checkpoint to resume training """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
