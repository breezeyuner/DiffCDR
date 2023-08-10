import torch


class SSCDR(torch.nn.Module):
    def __init__(self,input_dim =10 ):
        super(SSCDR,self).__init__()
        self.input_dim = input_dim
        self.mapping_layer = torch.nn.Linear(self. input_dim,self. input_dim)

    def forward(self, emb ):

        return self.mapping_layer (emb)

def sscdr_loss_fn(xu_emb_mapped,xi_pos_emb_mapped,xi_neg_emb_mapped,xu_tgt_emb,):

    lam = 1
    margin=1

    dist1 = xu_emb_mapped-xu_tgt_emb
    loss_s = torch.sum( dist1*dist1, axis=-1 )

    dist2 = xi_pos_emb_mapped-xu_tgt_emb
    dist_pos = torch.sum( dist2*dist2, axis=-1 )

    dist3 = xi_neg_emb_mapped-xu_tgt_emb
    dist_neg = torch.sum( dist3*dist3, axis=-1 )

    loss_u = torch.clamp(margin + dist_pos - dist_neg, min=0)
    
    return (loss_s + lam*loss_u).mean()

