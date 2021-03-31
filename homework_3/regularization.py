def loss_ReL1(model, decay = .1, loss_ft = None):
    if loss_ft is None:
        return None
    
    L1_norms = [par.norm(1).item() for name, par in model.named_parameters() if 'weight' in name]
    return loss_ft + decay*sum(L1_norms)