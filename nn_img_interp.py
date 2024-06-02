# Import Statements
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

# Siren definition
class SineLayer(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 /
                                            self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords

# Define helper functions
def get_image_tensor(height, width):
    img = Image.open("AT-Duck-1.jpeg").convert("RGB")
    transform = Compose([
        Resize((height, width)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def get_mgrid(height, width):
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    tensors = (y, x)
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, 2)
    return mgrid

# Define and instantiate datasets
class ImageFitting(Dataset):
    def __init__(self, height, width):
        super().__init__()
        img = get_image_tensor(height, width)
        self.pixels = img.permute(1, 2, 0).view(-1, 3)
        self.coords = get_mgrid(height, width)        

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        if index > 0: raise IndexError
        
        return self.coords, self.pixels
    
dataset = ImageFitting(500, 500)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

# Get training data
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

# Define training function
def train(siren, ratio_given_pix, num_epochs=1500, lr=1e-4):
    # Instantiate optimizer and define given pixel indices
    optimizer = torch.optim.Adam(siren.parameters(), lr=lr)
    idx = np.random.choice(
        model_input.shape[1], 
        size=int(ratio_given_pix * model_input.shape[1]), 
        replace=False)
    
    # Train the model
    print("")
    print("=" * 75)
    print(f" Starting Training - {ratio_given_pix * 100}% Pixels ".center(75, "="))
    print("=" * 75)
    for epoch in range(num_epochs):
        model_output = siren(model_input[:, idx, :])[0]
        loss = F.mse_loss(model_output, ground_truth[:, idx, :])

        # Penalize with L1 norm of parameter vector - encourages sparsity.
        # Since layers look like sin(Wx + b), sparsity of W, b ~ sparsity 
        # over a Fourier basis (i.e. compressed sensing)
        params = torch.nn.utils.parameters_to_vector(siren.parameters())
        loss = loss + 1e-2 * params.abs().mean()

        if not (epoch + 1) % 10:
            print(f"Step {epoch + 1}, Total loss: {loss: .6f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    del optimizer, epoch, model_output, loss
    return copy.deepcopy(siren), idx

# Run experiments
ratios = 3.0**-np.arange(5)
ratios = np.round(ratios, 2)

model_name_template = "models/model-%.2f.pt"
models = []
indices = []
for ratio in ratios:
    siren = Siren(
        in_features=2,
        out_features=3,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True
    )
    siren.cuda()

    siren, idx = train(siren, ratio)
    models.append(siren)
    indices.append(idx)

    torch.save(siren.state_dict(), model_name_template % ratio)

# Make plots
fig, ax = plt.subplots(
    nrows=len(ratios),
    ncols=4,
    figsize=(6, 8),
    width_ratios=[1, 1, 1, 0.15]
)

for i in range(len(ratios)):
    given_pixels = torch.ones(ground_truth.shape)
    given_pixels[:, indices[i], :] = ground_truth[:, indices[i], :].cpu()
    given_pixels = given_pixels.view(500, 500, 3)
    reconstruction = models[i](model_input)[0].view(500, 500, 3).cpu().detach()
    truth = ground_truth.view(500, 500, 3).cpu()
    error = ((reconstruction - truth) / truth).abs().mean(axis=2)

    ax[i, 0].imshow(given_pixels*.5 + .5)
    ax[i, 1].imshow(reconstruction*.5 + .5)
    m = ax[i, 2].imshow(error, cmap="gray")
    plt.colorbar(m, cax=ax[i, 3])

    ax[i, 0].set_ylabel(f"{ratios[i] * 100}%")

    ax[i, 0].set_yticklabels([])
    ax[i, 0].set_xticklabels([])
    ax[i, 1].set_yticklabels([])
    ax[i, 1].set_xticklabels([])
    ax[i, 2].set_yticklabels([])
    ax[i, 2].set_xticklabels([])
    ax[i, 0].set_yticks([])
    ax[i, 0].set_xticks([])
    ax[i, 1].set_yticks([])
    ax[i, 1].set_xticks([])
    ax[i, 2].set_yticks([])
    ax[i, 2].set_xticks([])

ax[0, 0].set_title("Given Data")
ax[0, 1].set_title("NN Reconstruction")
ax[0, 2].set_title("Channel-wise %MAE")

ax[0, 0].patch.set_edgecolor("red")
ax[0, 0].patch.set_linewidth(8)

fig.tight_layout()
fig.savefig("Test_Results.png")
plt.close(fig)