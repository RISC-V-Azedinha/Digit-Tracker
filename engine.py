"""Núcleo do Eureka, sem interface: rastreio do bastão, traços, pré-processamento 28x28 e inferência."""
import re
import time
from collections import namedtuple

import cv2
import numpy as np
from serial.tools import list_ports

from vision import StrokeCanvas, to_npu_input

PROC_WIDTH = 960  # largura usada no rastreio e no desenho (a exibição escala para a janela)
LIFT_AFTER = 3    # quadros sem ver o bastão até "levantar a caneta"
CLEAR_HOLD_S = 0.8   # tempo segurando o punho fechado para limpar o desenho
FIST_GRACE_S = 0.2   # falhas curtas do detector não zeram o progresso do gesto
UART_TX_BYTES = 1 + 784  # comando 0xFF + imagem 28x28 int8
UART_RX_BYTES = 10       # 10 logits int8

Prediction = namedtuple("Prediction", "digit probs logits")


def softmax_pct(logits, temperature=15.0):
    """Mesma calibração de confiança da aba Neural Network do GUI-TCC."""
    z = np.asarray(logits, np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum() * 100.0


def find_fpga_port():
    """
    Porta serial da FPGA: entre as portas USB-serial, usa a de maior número, porque as placas
    com FTDI duplo expõem JTAG + UART e a UART é a segunda (ex.: /dev/ttyUSB1).
    Retorna (porta ou None, lista de portas USB encontradas).
    """
    natural = lambda dev: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", dev)]
    ports = sorted((p.device for p in list_ports.comports() if p.vid is not None), key=natural)
    return (ports[-1] if ports else None), ports


class Engine:
    """Processa cada quadro da câmera e dispara a inferência quando o desenho para."""

    def __init__(self, tracker, idle_s=0.8, calibration_path=None, hand_tracker=None, live_interval=0.2):
        # Inferência ao vivo, como a lousa do experimento 7: enquanto o desenho muda, uma a cada
        # live_interval segundos (0 desliga). Ao parar de desenhar, sempre há uma inferência final.
        self.live_interval = live_interval
        self._last_live = float("-inf")
        self.final_version = -1
        self.tracker = tracker    # rastreio por cor (bastão)
        self.hand = hand_tracker  # rastreio da mão (MediaPipe), se disponível
        self.input_mode = "mão" if hand_tracker else "cor"
        self.pen_down = False     # caneta encostada: desenhando neste quadro
        self.landmarks = None     # 21 pontos da mão, para o HUD
        self.clear_progress = None  # 0..1 enquanto o punho está fechado (gesto de limpar)
        self.on_gesture_clear = None
        self._fist_since = None
        self._fist_seen = None
        self._fist_fired = False
        self.idle_s = idle_s  # tempo parado até disparar a inferência
        self.calibration_path = calibration_path
        self.backend = None   # definido quando a NPU (FPGA ou emulada) fica pronta
        self.on_inference = None  # callback(Prediction, latência em ms, ao vivo?)
        self.pen_enabled = True
        self.size = None
        self.canvas = None
        self.frame = None
        self.point = None
        self.mask = None
        self.lost_frames = 0
        self.preview_version = -1
        self.inferred_version = -1
        self.npu_input = None
        self.img28 = None
        self.prediction = None
        self.latency_ms = None
        self.inferences = 0
        self.bytes_tx = 0
        self.bytes_rx = 0
        self.error = ""

    def process(self, frame, now=None):
        """Recebe um quadro BGR (já espelhado) e atualiza rastreio, traços e inferência."""
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
        size = (PROC_WIDTH, int(round(PROC_WIDTH * h / w)))
        if (w, h) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if self.canvas is None or self.size != size:
            self.size = size
            self.canvas = StrokeCanvas(size, max_jump=130)
        self.frame = frame

        if self.input_mode == "mão":
            # Só o indicador levantado = caneta abaixada; indicador + médio levanta a caneta na hora
            self.point, touching = self.hand.detect(frame, now)
            self.mask, self.landmarks = None, self.hand.landmarks
            self._update_clear_gesture(now)
        else:
            # Bastão colorido: desenha sempre que ele aparece
            self.point, self.mask = self.tracker.detect(frame)
            touching, self.landmarks = self.point is not None, None

        self.pen_down = touching and self.pen_enabled and self.point is not None
        if self.pen_down:
            self.lost_frames = 0
            self.canvas.add_point(self.point, now)
        else:
            self.lost_frames += 1
            if self.input_mode == "mão" or self.lost_frames >= LIFT_AFTER or not self.pen_enabled:
                self.canvas.lift()

        if self.canvas.version != self.preview_version:
            converted = to_npu_input(self.canvas.strokes)
            self.npu_input, self.img28 = converted if converted else (None, None)
            self.preview_version = self.canvas.version

        if self.npu_input is None:
            return
        if (self.live_interval and self.pen_down and self.canvas.version != self.inferred_version
                and now - self._last_live >= self.live_interval):
            self._last_live = now
            self.infer(live=True)
        elif now - self.canvas.last_change >= self.idle_s and self.canvas.version != self.final_version:
            self.infer()

    def infer(self, live=False):
        if self.backend is None or self.npu_input is None:
            return
        try:
            start = time.perf_counter()
            logits = self.backend.infer(self.npu_input)
            self.latency_ms = (time.perf_counter() - start) * 1000
            self.prediction = Prediction(int(np.argmax(logits)), softmax_pct(logits), np.asarray(logits))
            self.inferences += 1
            self.bytes_tx += UART_TX_BYTES
            self.bytes_rx += UART_RX_BYTES
            self.error = ""
            if self.on_inference:
                self.on_inference(self.prediction, self.latency_ms, live)
        except Exception as e:  # erro de comunicação não deve derrubar a interface
            self.error = str(e)
        self.inferred_version = self.canvas.version
        if not live:
            self.final_version = self.canvas.version

    def _update_clear_gesture(self, now):
        """Punho fechado por CLEAR_HOLD_S limpa o desenho (uma vez; para repetir, abra a mão)."""
        if self.hand.fist:
            if self._fist_since is None:
                self._fist_since = now
            self._fist_seen = now
            held = now - self._fist_since
            self.clear_progress = min(1.0, held / CLEAR_HOLD_S)
            if held >= CLEAR_HOLD_S and not self._fist_fired:
                self._fist_fired = True
                self.clear(now)
                if self.on_gesture_clear:
                    self.on_gesture_clear()
        elif self._fist_seen is None or now - self._fist_seen > FIST_GRACE_S:
            self._fist_since, self._fist_fired, self.clear_progress = None, False, None

    def clear(self, now=None):
        if self.canvas is not None:
            self.canvas.clear(time.monotonic() if now is None else now)
        self.prediction = None
        self.latency_ms = None

    @property
    def input_label(self):
        return "MÃO · INDICADOR" if self.input_mode == "mão" else f"COR · {self.tracker.mode.upper()}"

    def cycle_input(self):
        """Alterna a entrada: mão -> cor (ponta) -> cor (centro) -> mão."""
        if self.input_mode == "mão":
            self.input_mode, self.tracker.mode = "cor", "ponta"
        elif self.tracker.mode == "ponta":
            self.tracker.mode = "centro"
        elif self.hand is not None:
            self.input_mode = "mão"
        else:
            self.tracker.mode = "ponta"
        self.tracker._point = None
        if self.canvas is not None:
            self.canvas.lift()
        return self.input_label

    @property
    def roi(self):
        """Caixa (x0, y0, x1, y1) que envolve o dígito desenhado, no espaço do quadro processado."""
        if self.canvas is None or self.canvas.empty:
            return None
        pts = np.array([p for stroke in self.canvas.strokes for p in stroke])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)

    def calibrate_at(self, x, y):
        """Amostra a cor do bastão ao redor do ponto clicado e ajusta a faixa HSV."""
        if self.frame is None:
            return None
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        patch = hsv[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        if patch.size == 0:
            return None
        result = self.tracker.calibrate_from(patch)
        if self.calibration_path:
            self.tracker.save(self.calibration_path)
        return result
