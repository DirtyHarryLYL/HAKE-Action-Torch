#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
import torch

def save_model(net, optimizer, scheduler, iters, ckp_path):
    # checkpoint saver
    torch.save({
                "iters": iters,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, ckp_path)

def load_model(cfg, model, optimizer, scheduler, ckp_path, mode='train'):
    # checkpoint loader
    checkpoint = torch.load(ckp_path, map_location='cpu')
    assert mode in ['train', 'test']

    if mode == 'train':
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        iters = 0
        if cfg.TRAIN.LOAD_HISTORY:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # for state in scheduler.state.values():
                #     for k, v in state.items():
                #         if isinstance(v, torch.Tensor):
                #             state[k] = v.cuda()

            if 'iters' in checkpoint:
                iters = checkpoint['iters']
                
        return model, optimizer, scheduler, iters

    else:
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model, None, None, None