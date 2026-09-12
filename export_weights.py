"""
Gera weights.npz com os pesos int8 da CNN do experimento 7 (GUI-TCC), já calibrados para a NPU.

Carrega o modelo pré-treinado (models/cnn_pretrained.pth) e aplica a mesma calibração int8 do
GUI-TCC (core/nn_model.py), usando um lote do MNIST, baixado em ./data na primeira execução.
Precisa do torch e do torchvision só aqui; o Eureka em si não usa o torch:

    pip install torch torchvision
    python export_weights.py
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_model():
    import torch.nn as nn

    class Conv2D_Model(nn.Module):
        """Mesma rede do GUI-TCC (os nomes das camadas precisam bater com o state_dict do .pth)."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=0)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(13 * 13 * 4, 10)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = x.permute(0, 2, 3, 1)  # formato da NPU (channels-last)
            return self.fc(self.flatten(x))

    return Conv2D_Model()


def calibrate(model, images):
    """Quantização int8 do GUI-TCC: escalas escolhidas para os acumuladores não estourarem."""
    import torch

    model.eval()
    with torch.no_grad():
        max_y1 = model.relu(model.conv(images)).max().item()
        max_y2 = model(images).abs().max().item()

    w_conv_f = model.conv.weight.detach().numpy().reshape(4, 9)
    b_conv_f = model.conv.bias.detach().numpy()
    w_fc_f = model.fc.weight.detach().numpy()
    b_fc_f = model.fc.bias.detach().numpy()

    scale_w1_max = 127.0 / np.max(np.abs(w_conv_f))
    scale_w1_safe = (120.0 * 256.0) / (max_y1 * 127.0) if max_y1 > 0 else scale_w1_max
    scale_w1 = min(scale_w1_max, scale_w1_safe)
    w_conv_i = np.round(w_conv_f * scale_w1).astype(np.int8)
    b_conv_i = np.round(b_conv_f * 127.0 * scale_w1).astype(np.int32)

    scale_y1 = (127.0 * scale_w1) / 256.0
    scale_w2_max = 127.0 / np.max(np.abs(w_fc_f))
    scale_w2_safe = (120.0 * 256.0) / (max_y2 * scale_y1) if max_y2 > 0 else scale_w2_max
    scale_w2 = min(scale_w2_max, scale_w2_safe)
    w_fc_i = np.round(w_fc_f * scale_w2).astype(np.int8)
    b_fc_i = np.round(b_fc_f * scale_y1 * scale_w2).astype(np.int32)

    return w_conv_i, b_conv_i, w_fc_i, b_fc_i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.join(HERE, "models", "cnn_pretrained.pth"))
    parser.add_argument("--data", default=os.path.join(HERE, "data"), help="pasta do MNIST (calibração)")
    parser.add_argument("--out", default=os.path.join(HERE, "weights.npz"))
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    model = build_model()
    print(f"Carregando a rede pré-treinada ({args.model})...")
    model.load_state_dict(torch.load(args.model, weights_only=True))

    # Mesmo lote de calibração do GUI-TCC: 32 imagens do treino, com as mesmas transformações
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=15),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root=args.data, train=True, download=True, transform=transform)
    images, _ = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))

    print("Calibração int8 para o SoC RISC-V...")
    w_conv, b_conv, w_fc, b_fc = calibrate(model, images)
    np.savez(args.out, w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)
    print(f"Pesos salvos em {args.out}: w_conv {w_conv.shape}, w_fc {w_fc.shape}")


if __name__ == "__main__":
    main()
