"""
Rastreio da mão com o MediaPipe.

Gestos:
- só o indicador levantado ....... desenha (a ponta do indicador é a caneta)
- indicador + médio levantados ... move sem riscar
- punho fechado ................... gesto de limpar (tempo de espera no engine)
"""
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")      # o MediaPipe importa o matplotlib; sem janela própria
os.environ.setdefault("GLOG_minloglevel", "2")  # silencia os logs internos do MediaPipe

import cv2

# Ligações entre os 21 pontos da mão (para desenhar o esqueleto no HUD)
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20))
WRIST, INDEX_TIP = 0, 8
PALM = (0, 5, 9, 13, 17)
# Dedo esticado, medido nos pontos 3D (metros) do MediaPipe: ângulo de dobra entre base -> articulação
# do meio e articulação -> ponta. Não depende de para onde o dedo aponta, então funciona também com o
# indicador apontado para a câmera. Histerese: estica abaixo de 45°, dobra acima de 65°.
FINGER_CHAINS = {"indicador": (5, 6, 8), "medio": (9, 10, 12), "anelar": (13, 14, 16), "minimo": (17, 18, 20)}
CURL_EXTENDED, CURL_FOLDED = 45.0, 65.0
# Alternativa 2D (se faltarem os pontos 3D): ponta mais longe do pulso que a articulação do meio.
FINGERS = {"indicador": (8, 6), "medio": (12, 10), "anelar": (16, 14), "minimo": (20, 18)}  # (ponta, articulação)
EXTEND_ON, EXTEND_OFF = 1.15, 1.0
DETECT_WIDTH = 640  # a detecção roda numa cópia menor; as coordenadas voltam ao quadro original
LOST_GRACE = 3      # quadros sem detectar a mão em que a caneta continua valendo


def bend_degrees(a, b, c):
    """Ângulo (graus) entre os segmentos a->b e b->c: 0 = dedo reto."""
    v1 = [b[i] - a[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    norm = math.hypot(*v1) * math.hypot(*v2)
    if norm == 0:
        return 0.0
    cos = sum(x * y for x, y in zip(v1, v2)) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class OneEuroFilter:
    """
    Filtro "One Euro" (Casiez et al., 2012): suaviza forte com o dedo parado (tira o tremor)
    e pouco com o dedo rápido (quase sem atraso). Padrão para cursores controlados pela mão.
    """

    def __init__(self, min_cutoff=1.5, beta=0.03, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class HandTracker:
    name = "mão"

    def __init__(self, model_path, confirm_frames=2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.confirm_frames = confirm_frames  # quadros seguidos no gesto de desenhar até abaixar a caneta
        self.extended = {name: False for name in FINGERS}
        self.curl = {name: 0.0 for name in FINGERS}  # ângulo de dobra de cada dedo (graus)
        self.gesture = None   # "desenhar", "mover", "punho" ou None (sem mão)
        self.pen_down = False
        self.fist = False
        self.palm = None      # centro da palma, para o HUD
        self.landmarks = None
        self._fx, self._fy = OneEuroFilter(), OneEuroFilter()
        self._point = None
        self._draw_frames = 0
        self._lost = 0
        self._last_ts = -1

    def detect(self, frame_bgr, now):
        """Retorna (ponto da caneta ou None, caneta abaixada?)."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (DETECT_WIDTH, round(DETECT_WIDTH * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = max(int(now * 1000), self._last_ts + 1)  # o modo VIDEO exige tempo sempre crescente
        self._last_ts = ts
        result = self._landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)

        if not result.hand_landmarks:
            self._lost += 1
            if self._lost <= LOST_GRACE and self._point is not None:
                return self._point, self.pen_down  # falha curta do detector: não interrompe o traço
            self._reset()
            return None, False
        self._lost = 0

        pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        self.landmarks = [(int(x), int(y)) for x, y in pts]
        self.palm = (int(sum(pts[i][0] for i in PALM) / len(PALM)), int(sum(pts[i][1] for i in PALM) / len(PALM)))

        world = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
        if world:
            wp = [(lm.x, lm.y, lm.z) for lm in world]
            for name, (base, pip, tip) in FINGER_CHAINS.items():
                self.curl[name] = bend_degrees(wp[base], wp[pip], wp[tip])
                self.extended[name] = self.curl[name] < (CURL_FOLDED if self.extended[name] else CURL_EXTENDED)
        else:
            wrist = pts[WRIST]
            for name, (tip, pip) in FINGERS.items():
                ratio = math.dist(wrist, pts[tip]) / max(1.0, math.dist(wrist, pts[pip]))
                self.extended[name] = ratio > (EXTEND_OFF if self.extended[name] else EXTEND_ON)
        up = self.extended
        if not any(up.values()):
            self.gesture = "punho"
        elif up["indicador"] and not up["medio"]:
            self.gesture = "desenhar"
        else:
            self.gesture = "mover"
        self.fist = self.gesture == "punho"

        self._draw_frames = self._draw_frames + 1 if self.gesture == "desenhar" else 0
        was_down = self.pen_down
        self.pen_down = self._draw_frames >= self.confirm_frames
        if self.pen_down and not was_down:
            # Ao abaixar a caneta, parte da posição real do dedo (sem o atraso do filtro)
            self._fx.reset()
            self._fy.reset()
        x, y = pts[INDEX_TIP]
        self._point = (int(round(self._fx(x, now))), int(round(self._fy(y, now))))
        return self._point, self.pen_down

    def _reset(self):
        self.landmarks, self.gesture, self.palm, self._point = None, None, None, None
        self.pen_down = self.fist = False
        self.extended = {name: False for name in FINGERS}
        self.curl = {name: 0.0 for name in FINGERS}
        self._draw_frames = 0
        self._fx.reset()
        self._fy.reset()

    def close(self):
        self._landmarker.close()
