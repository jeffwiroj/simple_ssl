import torch
import torch.nn as nn
import torchvision.models as models





def get_backbone():
    network = models.resnet34(False) #not pretrain
    backbone = torch.nn.Sequential(*(list(network.children())[:-1]))
    return backbone

#Adapted from Lightly:https://github.com/lightly-ai/lightly/blob/master/lightly/models/simsiam.py
def projection_mlp(in_dims: int = 512,
                    h_dims: int = 512,
                    out_dims: int =  512,
                    num_layers: int = 3) -> nn.Sequential:
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection

def prediction_mlp(in_dims: int = 512, 
                    h_dims: int = int(512/4), 
                    out_dims: int = 512) -> nn.Sequential:
    """Prediction MLP. The original paper's implementation has 2 layers, with 
    BN applied to its hidden fc layers but no BN or ReLU on the output fc layer.
    Note that the hidden dimensions should be smaller than the input/output 
    dimensions (bottleneck structure). The default implementation using a 
    ResNet50 backbone has an input dimension of 2048, hidden dimension of 512, 
    and output dimension of 2048
    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers (should be a
            bottleneck!)
        out_dims: 
            Output Dimension of the final linear layer.
    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction

class SimSiam(nn.Module):
    """Implementation of SimSiam[0] network
    Recommended loss: :py:class:`lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss`
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head. This should
            be `num_ftrs` / 4.
        out_dim:
            Dimension of the output (after the projection head).
    """

    def __init__(self,
                 backbone = None,
                 num_ftrs: int = 512,
                 proj_hidden_dim: int = 512,
                 pred_hidden_dim: int = int(512/4),
                 out_dim: int = 512,
                 num_mlp_layers: int = 3):

        super(SimSiam, self).__init__()
        
        self.backbone = backbone if backbone is not None else get_backbone()
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = \
            projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

        self.prediction_mlp = \
            prediction_mlp(out_dim, pred_hidden_dim, out_dim)
        
    def forward(self, 
                x0: torch.Tensor, 
                x1: torch.Tensor = None,
                return_features: bool = False):

        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_mlp(f0)
        p0 = self.prediction_mlp(z0)

        out0 = (z0, p0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0
        
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_mlp(f1)
        p1 = self.prediction_mlp(z1)

        out1 = (z1, p1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1

class SimSiamLoss(torch.nn.Module):
    """Implementation of the Symmetrized Loss used in the SimSiam[0] paper.
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    
    Examples:
        >>> # initialize loss function
        >>> loss_fn = SymNegCosineSimilarityLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def _neg_cosine_simililarity(self, x, y):
        v = - torch.nn.functional.cosine_similarity(x, y.detach(), dim=-1).mean()
        return v

    def forward(self, 
                out0: torch.Tensor, 
                out1: torch.Tensor):
        """Forward pass through Symmetric Loss.
            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Expects the tuple to be of the form (z0, p0), where z0 is
                    the output of the backbone and projection mlp, and p0 is the
                    output of the prediction head.
                out1:
                    Output projections of the second set of transformed images.
                    Expects the tuple to be of the form (z1, p1), where z1 is
                    the output of the backbone and projection mlp, and p1 is the
                    output of the prediction head.
 
            Returns:
                Contrastive Cross Entropy Loss value.
            Raises:
                ValueError if shape of output is not multiple of batch_size.
        """
        z0, p0 = out0
        z1, p1 = out1

        loss = self._neg_cosine_simililarity(p0, z1) / 2 + \
               self._neg_cosine_simililarity(p1, z0) / 2

        return loss
    
    
if __name__ == "__main__":
    backbone = get_backbone()
    simsiam_ = SimSiam(backbone = backbone)
    img1,img2 = torch.rand(2,3,28,28),torch.rand(2,3,28,28)
    
    
    out1,out2 = simsiam_(img1,img2)
    criterion = SimSiamLoss()
    loss = criterion(out1,out2)
    print(f"Img shape: {img1.shape}, output shape: {out1[0].shape}, loss: {loss.item()}")