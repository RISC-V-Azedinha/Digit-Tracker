"""
Pipeline em threads, para a interface nunca travar:

    CaptureThread ──(só o quadro mais recente)──> ProcessingThread ──(Snapshot pronto)──> interface
      lê a câmera                                   rastreio, traços, NPU                  só desenha

A captura descarta os quadros atrasados em vez de enfileirá-los, então a imagem nunca fica
"para trás" da mão. A interface fala com o processamento só por comandos (submit) e sinais Qt.
"""
import os
import queue
import threading
import time
from collections import namedtuple

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

Snapshot = namedtuple("Snapshot", (
    "frame mask size strokes point pen_down pen_enabled landmarks palm clear_progress roi img28 version "
    "input_label gesture fingers live error backend_label latency_ms inferences bytes_tx bytes_rx proc_ms fps"))


class CaptureThread(QThread):
    """Lê a câmera sem parar e guarda só o quadro mais recente."""
    opened = pyqtSignal(int, int, float)
    failed = pyqtSignal(str)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self._running = True
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def run(self):
        api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, api)
        if not cap.isOpened():
            self.failed.emit(f"Não foi possível abrir a câmera {self.index} (tente --camera 1)")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 720p a 30 FPS na maioria das webcams
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # sem fila de quadros velhos no driver
        self.opened.emit(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         float(cap.get(cv2.CAP_PROP_FPS) or 0))
        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.failed.emit("A câmera parou de enviar imagens")
                break
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()
        cap.release()

    def latest(self, last_seq, timeout):
        """Espera um quadro mais novo que last_seq. Retorna (quadro, seq) ou (None, last_seq)."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._frame, self._seq

    def stop(self):
        self._running = False
        self.wait(2000)


class ProcessingThread(QThread):
    """Roda o Engine fora da interface e entrega um Snapshot por quadro."""
    snapshot = pyqtSignal(object)
    inference = pyqtSignal(object, float, bool)  # (predição, latência em ms, ao vivo?)
    cleared = pyqtSignal()
    message = pyqtSignal(str, str)  # (texto, tipo: info | ok | warn | error)
    hsv_changed = pyqtSignal(object, object)

    def __init__(self, engine, source):
        super().__init__()
        self.engine = engine
        self.source = source
        self.commands = queue.Queue()
        self.pending = False     # a interface ainda não consumiu o último Snapshot
        self.show_mask = False
        self._running = True
        self._seq = 0
        self._fps = 0.0
        self._last_t = None
        self.live_default = engine.live_interval or 0.2
        engine.on_inference = lambda pred, ms, live: self.inference.emit(pred, ms, live)
        engine.on_gesture_clear = self.cleared.emit

    def submit(self, fn, *args):
        """Agenda fn(*args) para rodar nesta thread, antes do próximo quadro."""
        self.commands.put((fn, args))

    def run(self):
        while self._running:
            self._run_commands()
            frame, seq = self.source.latest(self._seq, 0.1)
            if frame is None:
                continue
            self._seq = seq
            start = time.perf_counter()
            if self._last_t is not None:
                dt = start - self._last_t
                self._fps = 0.9 * self._fps + 0.1 / dt if dt > 0 else self._fps
            self._last_t = start
            try:
                self.engine.process(cv2.flip(frame, 1))  # espelhado: desenhar fica natural
            except Exception as e:
                self.message.emit(f"Erro no processamento: {e}", "error")
                continue
            proc_ms = (time.perf_counter() - start) * 1000
            if not self.pending:
                self.pending = True
                self.snapshot.emit(self._make_snapshot(proc_ms))

    def _run_commands(self):
        while True:
            try:
                fn, args = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn(*args)
            except Exception as e:
                self.message.emit(f"Erro: {e}", "error")

    def _make_snapshot(self, proc_ms):
        e = self.engine
        hand = e.hand if e.input_mode == "mão" else None
        return Snapshot(
            gesture=hand.gesture if hand else None,
            fingers=tuple(hand.extended.values()) if hand and hand.landmarks else None,
            live=bool(e.live_interval),
            frame=e.frame, mask=e.mask if self.show_mask else None, size=e.size,
            strokes=[list(s) for s in e.canvas.strokes], point=e.point, pen_down=e.pen_down,
            pen_enabled=e.pen_enabled, landmarks=list(e.landmarks) if e.landmarks else None,
            palm=e.hand.palm if e.input_mode == "mão" else None, clear_progress=e.clear_progress,
            roi=e.roi, img28=e.img28, version=e.canvas.version, input_label=e.input_label, error=e.error,
            backend_label=e.backend.label if e.backend else None, latency_ms=e.latency_ms,
            inferences=e.inferences, bytes_tx=e.bytes_tx, bytes_rx=e.bytes_rx, proc_ms=proc_ms, fps=self._fps)

    # ---- comandos (rodam nesta thread via submit) ----
    def cycle_input(self):
        label = self.engine.cycle_input()
        if self.engine.input_mode == "cor" and self.engine.calibration_path:
            self.engine.tracker.save(self.engine.calibration_path)
        self.message.emit(f"Entrada: {label}", "info")

    def toggle_live(self):
        e = self.engine
        e.live_interval = 0.0 if e.live_interval else self.live_default
        self.message.emit("Inferência ao vivo: " + ("ligada" if e.live_interval else "desligada"), "info")

    def calibrate(self, x, y):
        result = self.engine.calibrate_at(x, y)
        if result:
            self.message.emit("Cor calibrada: H={} S={} V={}".format(*result), "ok")
            self.hsv_changed.emit(self.engine.tracker.lower.copy(), self.engine.tracker.upper.copy())

    def stop(self):
        self._running = False
        self.wait(3000)
