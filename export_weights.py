"""
Gera weights.npz com os pesos int8 da CNN do experimento 7 (GUI-TCC), já calibrados para a NPU.

Usa o código e o ambiente do GUI-TCC (torch/torchvision) uma única vez; o Eureka em si não
precisa do torch. Reaproveita o modelo pré-treinado (artifacts/cnn_pretrained.pth) e o MNIST
em GUI-TCC/data para a calibração int8.

    ../GUI-TCC/.venv/bin/python export_weights.py         (Windows: ..\\GUI-TCC\\.venv\\Scripts\\python)
"""
import argparse
import os
import sys

import numpy as np


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gui-tcc", default=os.path.join(here, "..", "GUI-TCC"), help="pasta do GUI-TCC")
    parser.add_argument("--out", default=os.path.join(here, "weights.npz"))
    args = parser.parse_args()

    gui_tcc = os.path.abspath(args.gui_tcc)
    out = os.path.abspath(args.out)
    sys.path.insert(0, gui_tcc)
    os.chdir(gui_tcc)  # o carregamento do MNIST usa ./data relativo ao GUI-TCC

    from core.nn_model import carregar_ou_treinar
    w_conv, b_conv, w_fc, b_fc = carregar_ou_treinar(progress_cb=print)

    np.savez(out, w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)
    print(f"Pesos salvos em {out}: w_conv {w_conv.shape}, w_fc {w_fc.shape}")


if __name__ == "__main__":
    main()
