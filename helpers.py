import torch


def save_checkpoint(state, epoch, checkpoint_dir, best_model_dir, is_best=True):
    """ Save a train checkpoint in checkpoint_dir, save best model in best_model_dir"""
    # f_path = checkpoint_dir / 'checkpoint_{}.pt'.format(epoch)
    f_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model.pt'
        torch.save(state['state_dict'], best_fpath)
        # shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    """ load a checkpoint to resume training """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
