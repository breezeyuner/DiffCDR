
import torch



class LACDR(torch.nn.Module):
    def __init__(self,input_dim =10 , hideen_dim = 4 ):
        super(LACDR,self).__init__()
        self.input_dim = input_dim
        self.hideen_dim = hideen_dim

        self.encoder_s1 = torch.nn.Linear(input_dim, hideen_dim, False)
        
        self.decoder_s1 = torch.nn.Linear(hideen_dim, input_dim, False)

        self.encoder_t1 = torch.nn.Linear(input_dim, hideen_dim, False)

        self.decoder_t1 = torch.nn.Linear(hideen_dim, input_dim, False)

    def forward(self, emb, type = 2 ):
        '''
        type = 0:  encoder_s , decoder_s 
        type = 1:  encoder_t , decoder_t 
        type = 2:  encoder_s , decoder_t 
        '''
        output = None 
        if type == 0:
            #output = self.decoder_s(self.encoder_s(emb) ) 
            output = self.encoder_s1(emb)
            output = self.decoder_s1(output)

        elif type == 1:
            #output = self.decoder_t(self.encoder_t(emb) ) 
            output = self.encoder_t1(emb)
            output = self.decoder_t1(output)

        elif type == 2:
            #output = self.decoder_t(self.encoder_s(emb) ) 
            output = self.encoder_s1(emb)
            output = self.decoder_t1(output)

        return output
    
    def get_hidden_s(self,emb):
        #return self.encoder_s(emb) 
        return self.encoder_s1(emb)
    
    def get_hidden_t(self,emb):
        #return self.encoder_t(emb) 
        return self.encoder_t1(emb)

def lacdr_loss_fn(model,emb_s,mask_s, emb_t,mask_t , alpha = 2 ):

    #loss_s_re
    u_hat_s = model(emb_s,0)
    loss_s_re = mask_s * torch.sqrt(  torch.sum(  torch.square(u_hat_s - emb_s) ,axis=-1  ))
    
    #loss_t_re
    u_hat_t = model(emb_t,1)
    loss_t_re = mask_t * torch.sqrt( torch.sum(  torch.square(u_hat_t - emb_t) ,axis=-1  ))

    #loss_align 
    loss_align = mask_s * mask_t * torch.sqrt( torch.sum( 
                                    torch.square( model.get_hidden_s(emb_s) - model.get_hidden_t(emb_t)  ) 
                                    ,axis=-1))
    return  torch.sum( loss_s_re + loss_t_re  + alpha * loss_align )
    
