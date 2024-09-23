
from medical_diffusion.models.utils.conv_blocks import save_add
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c


class LabelEmbedderRFMID(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name="ReLU"):
        super().__init__()
        self.emb_dim = emb_dim

        # Define the embedding network
        self.emb_net = nn.Sequential(
            nn.Linear(num_classes, emb_dim),
            getattr(nn, act_name)(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, condition):
        # print(condition)
        # Assuming 'condition' is a batch of binary vectors
        c = self.emb_net(condition.float())  # Transform the binary vector
        return c
    
    

class LabelEmbedderTOPCON(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name="ReLU"):
        super().__init__()
        self.emb_dim = emb_dim

        # Define the embedding network
        self.emb_net = nn.Embedding(num_classes, emb_dim)
        # self.l_r_net = nn.Embedding(2, emb_dim)

    def forward(self, condition):
        if condition is None:
            return None
        
        # Extract the target and eye_flag from the combined condition tensor
        target = condition  # First element is the target
        # eye_flag = condition[:, 1]  # Second element is the eye flag

        # Embed the target and the eye flag
        c = self.emb_net(target)  # Transform the target class vector
        # d = self.l_r_net(eye_flag)  # Transform the eye flag
        
        # Combine the embeddings
        return c
    
    
class IDRIDLabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        # Create an embedding for every possible combination of labels
        self.embedding = nn.Embedding(num_classes * num_classes, emb_dim)

    def forward(self, condition):
        # Map each pair of labels to a unique index
        unique_indices = condition[:, 0] * self.num_classes + condition[:, 1]
        return self.embedding(unique_indices)
