"""Rastreamento do bastão por cor, traços desenhados no ar e pré-processamento 28x28 da NPU."""
import json
import math

import cv2
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
STROKE_RATIO = 0.12  # espessura do traço relativa ao tamanho do dígito (proporção do MNIST)


class MarkerTracker:
    """
    Encontra o bastão pela cor (faixa HSV) e devolve a posição suavizada da caneta:
    - "ponta":  ponto mais alto da mancha colorida (bastão todo colorido, segurado apontando para cima);
    - "centro": centro da mancha (só a ponta é colorida: tampa, fita ou bolinha).
    """
    MODES = ("ponta", "centro")

    def __init__(self, lower=(35, 80, 80), upper=(85, 255, 255), mode="ponta", min_area=400, smoothing=0.45):
        self.lower = np.array(lower, np.uint8)
        self.upper = np.array(upper, np.uint8)
        self.mode = mode if mode in self.MODES else "ponta"
        self.min_area = min_area
        self.smoothing = smoothing  # peso da posição anterior (0 = sem suavização)
        self._point = None

    def toggle_mode(self):
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        self._point = None
        return self.mode

    def detect(self, frame_bgr):
        """Retorna (ponto ou None, máscara binária da cor)."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.dilate(mask, KERNEL, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob = max(contours, key=cv2.contourArea, default=None)
        if blob is None or cv2.contourArea(blob) < self.min_area:
            self._point = None
            return None, mask

        if self.mode == "ponta":
            # Média dos pontos do contorno perto do topo: mais estável que um único pixel
            pts = blob.reshape(-1, 2)
            top = pts[pts[:, 1] <= pts[:, 1].min() + 6]
            raw = (float(top[:, 0].mean()), float(top[:, 1].mean()))
        else:
            m = cv2.moments(blob)
            raw = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        if self._point is None:
            self._point = raw
        else:
            a = self.smoothing
            self._point = (a * self._point[0] + (1 - a) * raw[0], a * self._point[1] + (1 - a) * raw[1])
        return (int(round(self._point[0])), int(round(self._point[1]))), mask

    def calibrate_from(self, hsv_patch):
        """Ajusta a faixa HSV a partir de uma amostra da cor do bastão (clique na imagem)."""
        h, s, v = (int(np.median(hsv_patch[..., i])) for i in range(3))
        self.lower = np.array([max(0, h - 12), max(50, s - 80), max(50, v - 80)], np.uint8)
        self.upper = np.array([min(179, h + 12), 255, 255], np.uint8)
        return h, s, v

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"lower": self.lower.tolist(), "upper": self.upper.tolist(), "mode": self.mode}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(lower=data["lower"], upper=data["upper"], mode=data.get("mode", "ponta"))


class StrokeCanvas:
    """Traços desenhados no ar, como listas de pontos (a interface os desenha como linhas vetoriais)."""

    def __init__(self, size, max_jump=130, min_step=2.0):
        self.size = size
        self.max_jump = max_jump  # saltos maiores iniciam outro traço (detecção espúria ou reposicionamento)
        self.min_step = min_step  # ignora movimentos menores que isso (tremor residual)
        self.strokes = []
        self._current = None
        self.version = 0
        self.last_change = 0.0

    @property
    def empty(self):
        return not self.strokes

    def add_point(self, point, now):
        if self._current and math.dist(point, self._current[-1]) > self.max_jump:
            self.lift()
        if self._current is None:
            self._current = [point]
            self.strokes.append(self._current)
        elif math.dist(point, self._current[-1]) >= self.min_step:
            self._current.append(point)
        else:
            return
        self.version += 1
        self.last_change = now

    def lift(self):
        """Levanta a caneta: o próximo ponto começa um traço novo."""
        self._current = None

    def clear(self, now):
        self.strokes = []
        self._current = None
        self.version += 1
        self.last_change = now


def to_npu_input(strokes):
    """
    Converte os traços na entrada da NPU, como no experimento 7: recorta o dígito, redimensiona
    para caber em 20x20, centraliza em 28x28 e reduz para int8 (0..127).
    Retorna (vetor int8 de 784 posições, imagem 28x28 uint8 para visualização) ou None.
    """
    points = [p for stroke in strokes for p in stroke]
    if not points:
        return None
    pts = np.array(points)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    size = max(x1 - x0, y1 - y0, 1)
    thick = max(2, int(round(STROKE_RATIO * size)))

    # Redesenha os traços com espessura proporcional ao tamanho do dígito
    canvas = np.zeros((y1 - y0 + 1 + 2 * thick, x1 - x0 + 1 + 2 * thick), np.uint8)
    for stroke in strokes:
        shifted = np.array([(x - x0 + thick, y - y0 + thick) for x, y in stroke], np.int32)
        if len(shifted) == 1:
            cv2.circle(canvas, tuple(int(v) for v in shifted[0]), thick // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [shifted], False, 255, thick, cv2.LINE_AA)

    ys, xs = np.nonzero(canvas)
    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    ratio = 20.0 / max(h, w)
    nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    img28 = np.zeros((28, 28), np.uint8)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    img28[oy:oy + nh, ox:ox + nw] = small
    return np.clip(img28 // 2, 0, 127).astype(np.int8).reshape(-1), img28
