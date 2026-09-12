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

from paths import FROZEN, resource_path, user_path

# Executável sem console (PyInstaller windowed): sys.stdout/sys.stderr são None e qualquer print
# falharia com "'NoneType' object has no attribute 'write'".
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from engine import Engine, find_fpga_port
from npu import CpuNpu, FpgaNpu, load_weights
from vision import MarkerTracker

CALIBRATION_PATH = user_path("calibration.json")

# --selftest: usado pelo CI da release. Abre a janela com a NPU emulada, testa o MediaPipe e uma
# inferência, fecha em seguida e grava o resultado em selftest.log (o .exe não tem console).
SELFTEST = "--selftest" in sys.argv


def fail(message):
    """Encerra com uma mensagem. No executável, que não tem console, mostra uma caixa de diálogo."""
    if FROZEN and not SELFTEST:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Eureka", message)
    sys.exit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="porta serial da FPGA (padrão: detectada automaticamente)")
    parser.add_argument("--sim", action="store_true", help="sem placa: emula a NPU no computador")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão: 0)")
    parser.add_argument("--weights", default=resource_path("weights.npz"))
    parser.add_argument("--firmware", default=resource_path("firmware", "cnn_server.bin"))
    parser.add_argument("--input", choices=("mao", "cor"), default="mao",
                        help="mao: desenha com o dedo indicador; cor: bastão colorido (padrão: mao)")
    parser.add_argument("--model", default=resource_path("models", "hand_landmarker.task"),
                        help="modelo do MediaPipe para o rastreio da mão")
    parser.add_argument("--idle", type=float, default=0.8, help="segundos parado até inferir (padrão: 0.8)")
    parser.add_argument("--live-interval", type=float, default=0.2,
                        help="segundos entre inferências durante o desenho; 0 desliga (padrão: 0.2)")
    parser.add_argument("--fullscreen", action="store_true", help="abre em tela cheia (F11 alterna)")
    parser.add_argument("--scale", type=float, help="escala da interface (ex.: 1.5 em telas de alta resolução)")
    parser.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.selftest:
        args.sim = True

    if args.scale:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    # Importado só aqui: o Qt não é necessário para o núcleo (engine/vision/npu)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    from gui import MainWindow

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if not os.path.exists(args.weights):
        fail(f"Pesos não encontrados em {args.weights}. Gere com export_weights.py (veja o README).")
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
                fail("Nenhuma porta USB-serial encontrada. Conecte a FPGA, use --port ou rode com --sim"
                     + (" (Eureka-sim.bat)." if FROZEN else "."))
            print(f"Portas USB-serial: {', '.join(found)} -> usando {port} (outra: --port)")
        if not os.path.exists(args.firmware):
            fail(f"Firmware não encontrado em {args.firmware}. Use --firmware.")
        label = f"FPGA {port}"
        link = f"{port} @ {args.baud} baud"
        factory = lambda log: FpgaNpu(port, args.baud, args.firmware, weights, log=log)

    hand, hand_error = None, None
    if args.input == "mao":
        try:
            from hand import HandTracker
            hand = HandTracker(args.model)
        except Exception as e:  # sem MediaPipe ou sem o modelo: segue com o bastão colorido
            hand_error = f"Rastreio da mão indisponível ({e}). Usando o rastreio por cor."
            print(hand_error)
    if args.selftest and hand is None:
        raise RuntimeError(hand_error)

    tracker = MarkerTracker.load(CALIBRATION_PATH) if os.path.exists(CALIBRATION_PATH) else MarkerTracker()
    engine = Engine(tracker, idle_s=args.idle, calibration_path=CALIBRATION_PATH, hand_tracker=hand,
                    live_interval=args.live_interval)
    # No selftest não abre a câmera: o runner do CI não tem uma
    window = MainWindow(engine, factory, label, link, camera_index=None if args.selftest else args.camera)
    if hand_error:
        window.log(hand_error, "warn")
    window.show_initial(args.fullscreen)

    if args.selftest:
        import numpy as np
        logits = CpuNpu(weights).infer(np.zeros((28, 28), np.int8))
        _selftest_log(f"MediaPipe: {hand.name} ({args.model})")
        _selftest_log(f"NPU emulada: {logits!r}")
        _selftest_log(f"janela principal: {window.windowTitle()}")
        QTimer.singleShot(1500, window.close)

    sys.exit(app.exec_())


def _selftest_log(line):
    with open(user_path("selftest.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    if not SELFTEST:
        main()
    else:
        try:
            main()
        except SystemExit as exit_:
            _selftest_log("OK" if not exit_.code else f"FALHA: {exit_.code}")
            raise
        except BaseException:
            import traceback
            _selftest_log("FALHA:\n" + traceback.format_exc())
            os._exit(1)
