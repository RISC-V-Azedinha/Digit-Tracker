"""
Eureka: desenhe um dígito no ar (com o dedo indicador ou com um bastão colorido) e a NPU do
SoC RISC-V na FPGA faz a inferência, como no experimento 7 do GUI-TCC.

    python eureka.py                 # inferência na FPGA (porta detectada automaticamente)
    python eureka.py --port COM4     # FPGA em uma porta específica (Linux: --port /dev/ttyUSB1)
    python eureka.py --sim           # sem placa: NPU emulada no computador
"""
import argparse
import os
import sys

from engine import Engine, find_fpga_port
from npu import CpuNpu, FpgaNpu, load_weights
from vision import MarkerTracker

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(HERE, "calibration.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="porta serial da FPGA (padrão: detectada automaticamente)")
    parser.add_argument("--sim", action="store_true", help="sem placa: emula a NPU no computador")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão: 0)")
    parser.add_argument("--weights", default=os.path.join(HERE, "weights.npz"))
    parser.add_argument("--firmware", default=os.path.join(HERE, "..", "GUI-TCC", "artifacts", "cnn_server.bin"))
    parser.add_argument("--input", choices=("mao", "cor"), default="mao",
                        help="mao: desenha com o dedo indicador; cor: bastão colorido (padrão: mao)")
    parser.add_argument("--model", default=os.path.join(HERE, "models", "hand_landmarker.task"),
                        help="modelo do MediaPipe para o rastreio da mão")
    parser.add_argument("--idle", type=float, default=0.8, help="segundos parado até inferir (padrão: 0.8)")
    parser.add_argument("--fullscreen", action="store_true", help="abre em tela cheia (F11 alterna)")
    parser.add_argument("--scale", type=float, help="escala da interface (ex.: 1.5 em telas de alta resolução)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"Pesos não encontrados em {args.weights}. Gere com export_weights.py (veja o README).")
    weights = load_weights(args.weights)

    if args.sim:
        label = CpuNpu.label
        link = "simulação local (sem placa)"
        factory = lambda log: (log("NPU emulada no computador (aritmética int8)"), CpuNpu(weights))[1]
    else:
        port = args.port
        if not port:
            port, found = find_fpga_port()
            if not port:
                sys.exit("Nenhuma porta USB-serial encontrada. Conecte a FPGA, use --port ou rode com --sim.")
            print(f"Portas USB-serial: {', '.join(found)} -> usando {port} (outra: --port)")
        if not os.path.exists(args.firmware):
            sys.exit(f"Firmware não encontrado em {args.firmware}. Use --firmware.")
        label = f"FPGA {port}"
        link = f"{port} @ {args.baud} baud"
        factory = lambda log: FpgaNpu(port, args.baud, args.firmware, weights, log=log)

    hand = None
    if args.input == "mao":
        try:
            from hand import HandTracker
            hand = HandTracker(args.model)
        except Exception as e:  # sem MediaPipe ou sem o modelo: segue com o bastão colorido
            print(f"Rastreio da mão indisponível ({e}). Usando o rastreio por cor.")

    if args.scale:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    # Importado só aqui: o Qt não é necessário para o núcleo (engine/vision/npu)
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from gui import MainWindow

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    tracker = MarkerTracker.load(CALIBRATION_PATH) if os.path.exists(CALIBRATION_PATH) else MarkerTracker()
    engine = Engine(tracker, idle_s=args.idle, calibration_path=CALIBRATION_PATH, hand_tracker=hand)
    window = MainWindow(engine, factory, label, link, camera_index=args.camera)
    window.show_initial(args.fullscreen)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
