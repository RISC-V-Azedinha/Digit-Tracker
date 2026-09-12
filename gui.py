"""
Interface gráfica do Eureka (PyQt5), no mesmo estilo flat do GUI-TCC.

A interface só desenha: câmera, rastreio e NPU rodam em threads próprias (pipeline.py) e
entregam a cada quadro um Snapshot pronto.
"""
import time

import numpy as np
import qtawesome as qta
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                             QLabel, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QShortcut,
                             QSizePolicy, QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from hand import HAND_CONNECTIONS
from pipeline import CaptureThread, ProcessingThread

# ------------------------------------------------------------------ paleta (a mesma do GUI-TCC)
BG = "#12141A"
PANEL = "#0B0D12"
ELEMENT = "#1A1D24"
BORDER = "#2A2F3A"
TEXT = "#E2E8F0"
MUTED = "#8B9BB4"
DIM = "#475569"
TEAL = "#6CA1A2"
ORANGE = "#DC673E"
GREEN = "#5DB373"
MUSTARD = "#F2B845"
BLUE = "#3B82F6"
RED = "#EF4444"
FONT = "'JetBrains Mono', 'Consolas', 'DejaVu Sans Mono', monospace"
STROKE_PX = 14  # espessura do traço exibido, em pixels do quadro processado
LOG_COLORS = {"info": MUTED, "ok": GREEN, "warn": MUSTARD, "error": RED, "tx": BLUE}

STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: {BG}; color: {TEXT}; font-family: {FONT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
#Header {{ background-color: {BG}; border-bottom: 1px solid {BORDER}; }}
#AppTitle {{ color: {MUSTARD}; font-size: 17px; font-weight: 800; letter-spacing: 1px; }}
#Panel {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; }}
QLabel[class="SectionTitle"] {{ color: {MUTED}; font-weight: bold; }}
QLabel[class="Muted"] {{ color: {MUTED}; }}
QPlainTextEdit {{ background-color: {PANEL}; color: #CBD5E1; border: 1px solid {BORDER}; border-radius: 4px;
                  font-size: 12px; padding: 6px; }}
QPushButton[class="ActionBtn"] {{ background-color: {ELEMENT}; color: {TEXT}; border: 1px solid {BORDER};
                                   border-radius: 4px; padding: 6px 14px; font-weight: bold; }}
QPushButton[class="ActionBtn"]:hover {{ background-color: {BORDER}; border-color: {TEAL}; }}
QPushButton[class="PrimaryBtn"] {{ background-color: {BLUE}; color: white; border: none; border-radius: 4px;
                                    padding: 6px 16px; font-weight: bold; }}
QPushButton[class="PrimaryBtn"]:hover {{ background-color: #2563EB; }}
QPushButton[class="GhostBtn"] {{ background: transparent; color: #F3F3F3; border: none; border-radius: 4px;
                                  padding: 7px 13px; font-weight: bold; }}
QPushButton[class="GhostBtn"]:hover {{ background: {ELEMENT}; color: {MUSTARD}; }}
QTableWidget {{ background-color: transparent; border: none; gridline-color: transparent; }}
QTableWidget::item {{ border-bottom: 1px solid {ELEMENT}; padding: 0 6px; }}
QHeaderView::section {{ background-color: transparent; color: {TEAL}; border: none; border-bottom: 2px solid {TEAL};
                        padding: 6px; font-weight: bold; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; border: none; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {TEAL}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; margin: -4px 0; border-radius: 6px; }}
"""


def section_title(icon, text):
    """Título de seção no padrão do GUI-TCC: ícone flat + texto em negrito."""
    row = QHBoxLayout()
    row.setSpacing(8)
    ic = QLabel()
    ic.setPixmap(qta.icon(icon, color=MUTED).pixmap(14, 14))
    label = QLabel(text)
    label.setProperty("class", "SectionTitle")
    row.addWidget(ic)
    row.addWidget(label)
    row.addStretch()
    return row


def panel():
    frame = QFrame()
    frame.setObjectName("Panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 12, 12, 12)
    return frame, layout


def button(text, icon, cls="ActionBtn", icon_color=TEXT):
    btn = QPushButton(f" {text}")
    btn.setIcon(qta.icon(icon, color=icon_color))
    btn.setProperty("class", cls)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFocusPolicy(Qt.NoFocus)  # as teclas de atalho ficam com a janela
    return btn


class BackendThread(QThread):
    """Prepara a NPU (boot da FPGA + envio dos pesos) sem travar a interface."""
    log = pyqtSignal(str)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def run(self):
        try:
            self.ready.emit(self.factory(self.log.emit))
        except Exception as e:
            self.failed.emit(str(e))


# ------------------------------------------------------------------ componentes
class VideoView(QWidget):
    """Câmera com o traço (linhas vetoriais), o esqueleto da mão, o cursor e a ROI do dígito."""
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 300)
        self.setCursor(Qt.CrossCursor)
        self.snap = None
        self._image = None
        self._buffer = None
        self.message = "Inicializando câmera..."

    def set_snapshot(self, snap):
        src = snap.mask if snap.mask is not None else snap.frame
        h, w = src.shape[:2]
        self._buffer = src  # o QImage aponta para o array do numpy (sem cópia); mantém-no vivo
        if src.ndim == 2:
            self._image = QImage(src.data, w, h, w, QImage.Format_Grayscale8)
        else:
            self._image = QImage(src.data, w, h, 3 * w, QImage.Format_BGR888)
        self.snap = snap
        self.update()

    def _target(self):
        w, h = self.snap.size
        scale = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * scale) / 2, (self.height() - h * scale) / 2, w * scale, h * scale), scale

    def mousePressEvent(self, event):
        if self.snap is None or event.button() != Qt.LeftButton:
            return
        rect, scale = self._target()
        if rect.contains(event.pos()):
            self.clicked.emit(int((event.x() - rect.x()) / scale), int((event.y() - rect.y()) / scale))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(PANEL))
        if self._image is None:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return

        s = self.snap
        rect, scale = self._target()
        p.drawImage(rect, self._image)
        at = lambda x, y: QPointF(rect.x() + x * scale, rect.y() + y * scale)

        # Traços
        p.setPen(QPen(QColor(GREEN), max(2.0, STROKE_PX * scale), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for stroke in s.strokes:
            if len(stroke) == 1:
                p.drawPoint(at(*stroke[0]))
            else:
                path = QPainterPath(at(*stroke[0]))
                for q in stroke[1:]:
                    path.lineTo(at(*q))
                p.drawPath(path)

        # Região que vai para a NPU
        if s.roi:
            x0, y0, x1, y1 = s.roi
            pad = 20
            box = QRectF(at(x0 - pad, y0 - pad), at(x1 + pad, y1 + pad))
            p.setPen(QPen(QColor(TEAL), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(box)
            p.setPen(QColor(TEAL))
            p.drawText(QPointF(box.left(), box.top() - 6), "ROI → 28×28")

        # Esqueleto da mão
        if s.landmarks:
            joints = [at(x, y) for x, y in s.landmarks]
            bone = QColor(MUTED)
            bone.setAlpha(150)
            p.setPen(QPen(bone, 1.5))
            for a, b in HAND_CONNECTIONS:
                p.drawLine(joints[a], joints[b])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(TEXT))
            for q in joints:
                p.drawEllipse(q, 2.2, 2.2)
            p.setBrush(Qt.NoBrush)

        if s.clear_progress is not None and s.palm:
            # Gesto de limpar: anel que se completa em volta da palma
            c = at(*s.palm)
            ring = QRectF(c.x() - 36, c.y() - 36, 72, 72)
            p.setPen(QPen(QColor(BORDER), 4))
            p.drawEllipse(ring)
            done = s.clear_progress >= 1.0
            p.setPen(QPen(QColor(GREEN if done else MUSTARD), 4, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(ring, 90 * 16, -int(360 * 16 * s.clear_progress))
            p.drawText(QRectF(c.x() - 70, c.y() + 42, 140, 18), Qt.AlignCenter,
                       "limpo" if done else f"limpar {int(s.clear_progress * 100)}%")
        elif s.point:
            # Cursor: cheio enquanto desenha, vazado enquanto só move
            c = at(*s.point)
            if s.pen_down:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(GREEN))
                p.drawEllipse(c, 7, 7)
                p.setBrush(Qt.NoBrush)
                state = "desenhando"
            else:
                p.setPen(QPen(QColor(TEAL if s.pen_enabled else MUSTARD), 2))
                p.drawEllipse(c, 10, 10)
                state = "movendo" if s.pen_enabled else "pausado"
            p.setPen(QColor(TEXT))
            p.drawText(QPointF(c.x() + 16, c.y() + 5), state)

        if not s.pen_enabled:
            # Aviso de pausa no topo do vídeo
            text = "DESENHO PAUSADO  ·  [ESPAÇO] retoma"
            width = p.fontMetrics().horizontalAdvance(text) + 32
            box = QRectF(rect.center().x() - width / 2, rect.top() + 14, width, 30)
            p.setPen(QPen(QColor(MUSTARD), 1))
            p.setBrush(QColor(11, 13, 18, 225))
            p.drawRoundedRect(box, 4, 4)
            p.drawText(box, Qt.AlignCenter, text)


class NpuInputView(QWidget):
    """A imagem 28x28 como a NPU recebe; a borda acende a cada inferência."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(168, 168)
        self._image = None
        self._buffer = None
        self.flash = 0.0

    def set_image(self, img28):
        if img28 is None:
            self._image = None
        else:
            self._buffer = np.ascontiguousarray(img28)
            self._image = QImage(self._buffer.data, 28, 28, 28, QImage.Format_Grayscale8)
        self.update()

    def pulse(self):
        self.flash = 1.0
        self.update()

    def tick(self):
        if self.flash > 0:
            self.flash = max(0.0, self.flash - 0.1)
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.fillRect(r, QColor("#06070A"))
        if self._image is not None:
            p.drawImage(r, self._image)  # sem suavização: cada pixel da NPU vira um quadrado
        p.setPen(QPen(QColor(ELEMENT), 1))
        cell = r.width() / 28
        for i in range(1, 28):
            p.drawLine(QPointF(r.left() + i * cell, r.top()), QPointF(r.left() + i * cell, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + i * cell), QPointF(r.right(), r.top() + i * cell))
        p.setPen(QPen(QColor(TEAL) if self.flash > 0 else QColor(BORDER), 1 + self.flash))
        p.drawRect(r)


class ConfidenceBars(QWidget):
    """Confiança por classe, com o logit int8 que a NPU devolveu."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.target = np.zeros(10)
        self.shown = np.zeros(10)
        self.logits = None
        self.top = -1

    def set_values(self, probs, logits):
        self.target = np.asarray(probs, float)
        self.logits = logits
        self.top = int(np.argmax(logits))

    def reset(self):
        self.target = np.zeros(10)
        self.logits = None
        self.top = -1

    def tick(self):
        if np.abs(self.target - self.shown).max() > 0.05:
            self.shown += (self.target - self.shown) * 0.6  # chega ao valor novo em ~165 ms, antes da próxima inferência ao vivo
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        row = self.height() / 10
        x_bar, w_bar = 26, max(40, self.width() - 26 - 104)
        for d in range(10):
            y = d * row
            top = d == self.top
            p.setPen(QColor(GREEN if top else MUTED))
            p.drawText(QRectF(0, y, 20, row), Qt.AlignVCenter | Qt.AlignLeft, str(d))
            track = QRectF(x_bar, y + row / 2 - 3, w_bar, 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ELEMENT))
            p.drawRoundedRect(track, 3, 3)
            fill = w_bar * min(100.0, self.shown[d]) / 100.0
            if fill > 0.5:
                p.setBrush(QColor(GREEN if top else TEAL if self.shown[d] > 5 else DIM))
                p.drawRoundedRect(QRectF(track.left(), track.top(), fill, 6), 3, 3)
            p.setPen(QColor(TEXT if top else MUTED))
            p.drawText(QRectF(x_bar + w_bar + 6, y, 54, row), Qt.AlignVCenter | Qt.AlignRight, f"{self.shown[d]:5.1f}%")
            if self.logits is not None:
                p.setPen(QColor(DIM))
                p.drawText(QRectF(x_bar + w_bar + 62, y, 42, row), Qt.AlignVCenter | Qt.AlignRight,
                           f"{int(self.logits[d]):+d}")


class CalibrationPanel(QFrame):
    """Faixa HSV da cor rastreada (modo cor)."""
    changed = pyqtSignal(object, object)
    NAMES = ("H mín", "S mín", "V mín", "H máx", "S máx", "V máx")

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("Panel")
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 12)
        grid.addLayout(section_title("fa5s.sliders-h", "Calibração de cor (HSV)"), 0, 0, 1, 3)
        self.sliders = []
        for i, name in enumerate(self.NAMES):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 179 if name.startswith("H") else 255)
            slider.setFixedWidth(170)
            slider.setFocusPolicy(Qt.NoFocus)
            value = QLabel("0")
            value.setFixedWidth(30)
            slider.valueChanged.connect(lambda v, lbl=value: (lbl.setText(str(v)), self._emit()))
            name_lbl = QLabel(name)
            name_lbl.setProperty("class", "Muted")
            grid.addWidget(name_lbl, i + 1, 0)
            grid.addWidget(slider, i + 1, 1)
            grid.addWidget(value, i + 1, 2)
            self.sliders.append(slider)
        hint = QLabel("clique no bastão para calibrar sozinho")
        hint.setProperty("class", "Muted")
        grid.addWidget(hint, 7, 0, 1, 3)
        self._silent = False
        self.adjustSize()

    def set_range(self, lower, upper):
        self._silent = True
        for slider, value in zip(self.sliders, list(lower) + list(upper)):
            slider.setValue(int(value))
        self._silent = False

    def _emit(self):
        if not self._silent:
            values = [s.value() for s in self.sliders]
            self.changed.emit(values[:3], values[3:])


class PipelineStrip(QWidget):
    """Faixa que mostra onde cada etapa roda: desenho no HOST (computador) e inferência na FPGA."""
    HOST_STAGES = ("câmera", "rastreio da mão", "desenho", "imagem 28×28")

    def __init__(self):
        super().__init__()
        self.setFixedHeight(66)
        self.fpga = True
        self.tracking = self.drawing = self.has_image = False
        self.flash = 0.0      # acende a cada inferência e apaga aos poucos
        self.digit = None
        self.latency = None

    def set_backend(self, is_fpga):
        self.fpga = is_fpga
        self.update()

    def set_state(self, tracking, drawing, has_image):
        state = (tracking, drawing, has_image)
        if state != (self.tracking, self.drawing, self.has_image):
            self.tracking, self.drawing, self.has_image = state
            self.update()

    def pulse(self, digit, latency_ms):
        self.digit, self.latency, self.flash = digit, latency_ms, 1.0
        self.update()

    def tick(self):
        if self.flash > 0:
            self.flash = max(0.0, self.flash - 0.06)
            self.update()

    def _font(self, px, bold=False):
        font = self.font()
        font.setPixelSize(px)
        font.setBold(bold)
        return font

    def _zone(self, p, rect, color, title, subtitle):
        fill = QColor(color)
        fill.setAlpha(16)
        border = QColor(color)
        border.setAlpha(170)
        p.setPen(QPen(border, 1))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 4, 4)
        p.setFont(self._font(11, bold=True))
        p.setPen(color)
        p.drawText(QPointF(rect.left() + 10, rect.top() + 16), title)
        x = rect.left() + 10 + p.fontMetrics().horizontalAdvance(title) + 10
        p.setFont(self._font(11))
        p.setPen(QColor(MUTED))
        p.drawText(QPointF(x, rect.top() + 16), subtitle)

    def _stages(self, p, rect, names, active, color):
        gap = 20
        width = (rect.width() - gap * (len(names) - 1)) / len(names)
        p.setFont(self._font(11, bold=True))
        for i, (name, on) in enumerate(zip(names, active)):
            box = QRectF(rect.left() + i * (width + gap), rect.top(), width, rect.height())
            fill = QColor(color)
            fill.setAlpha(70)
            p.setPen(QPen(QColor(color) if on else QColor(BORDER), 1))
            p.setBrush(fill if on else QColor(ELEMENT))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor(TEXT) if on else QColor(MUTED))
            p.drawText(box, Qt.AlignCenter, p.fontMetrics().elidedText(name, Qt.ElideRight, int(width) - 8))
            if i < len(names) - 1:
                self._arrow(p, box.right() + 4, box.right() + gap - 4, box.center().y(), QColor(MUTED))

    def _arrow(self, p, x0, x1, y, color):
        p.setPen(QPen(color, 1.5))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        tip = 1 if x1 > x0 else -1
        p.drawLine(QPointF(x1, y), QPointF(x1 - 5 * tip, y - 4))
        p.drawLine(QPointF(x1, y), QPointF(x1 - 5 * tip, y + 4))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        link_w = 150
        host_w = int((w - link_w) * 0.58)
        fpga_x, fpga_w = host_w + link_w, w - host_w - link_w
        hw = QColor(ORANGE if self.fpga else MUSTARD)

        # HOST: tudo que roda no computador
        self._zone(p, QRectF(0.5, 0.5, host_w, h - 1), QColor(TEAL), "HOST · COMPUTADOR", "câmera, rastreio e desenho")
        self._stages(p, QRectF(10, 26, host_w - 20, 30), self.HOST_STAGES,
                     (True, self.tracking, self.drawing, self.has_image), QColor(TEAL))

        # FPGA: a inferência da CNN
        latency = f"  ·  {self.latency:.1f} ms" if self.latency is not None else ""
        if self.fpga:
            title, subtitle = "FPGA · SoC RISC-V", f"inferência da CNN na NPU{latency}"
        else:
            title, subtitle = "NPU EMULADA · COMPUTADOR", f"sem placa (--sim){latency}"
        self._zone(p, QRectF(fpga_x, 0.5, fpga_w - 1, h - 1), hw, title, subtitle)
        digit = "?" if self.digit is None else str(self.digit)
        self._stages(p, QRectF(fpga_x + 10, 26, fpga_w - 20, 30), ("NPU: Conv2D → ReLU → FC", f"logits → {digit}"),
                     (self.flash > 0, self.flash > 0), hw)

        # Link entre os dois: TX (imagem) e RX (logits), acendem a cada inferência
        on = QColor(hw)
        on.setAlphaF(0.35 + 0.65 * self.flash)
        color = on if self.flash > 0 else QColor(BORDER)
        x0, x1 = host_w + 12, fpga_x - 12
        p.setFont(self._font(10, bold=True))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(x0, 0, x1 - x0, 14), Qt.AlignCenter, "UART 921600" if self.fpga else "memória")
        tx_y, rx_y = h * 0.52, h * 0.86
        self._arrow(p, x0, x1, tx_y, color)
        self._arrow(p, x1, x0, rx_y, color)
        p.setFont(self._font(10))
        p.setPen(QColor(TEXT) if self.flash > 0 else QColor(MUTED))
        p.drawText(QRectF(x0, tx_y - 15, x1 - x0, 12), Qt.AlignCenter, "TX  785 B  →")
        p.drawText(QRectF(x0, rx_y - 15, x1 - x0, 12), Qt.AlignCenter, "←  RX  10 B")


# ------------------------------------------------------------------ janela
class MainWindow(QMainWindow):
    TELEMETRY = ("Backend", "Latência NPU", "Inferências", "UART TX", "UART RX", "Câmera", "Processamento")

    def __init__(self, engine, backend_factory, backend_label, link_text, camera_index=None, source=None):
        super().__init__()
        self.engine = engine
        self.backend_label = backend_label
        self.last = None
        self._version = -1
        self._input_label = None
        self._pen_enabled = None
        self._last_error = ""
        self.cam_text = "--"
        self._hw_where = "na NPU da FPGA"  # onde a inferência roda (muda para a emulação com --sim)

        self.setWindowTitle("Eureka — Air-Draw Neural Inference")
        self.setStyleSheet(STYLESHEET)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header(link_text))
        self.strip = PipelineStrip()
        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(24, 14, 24, 0)
        strip_row.addWidget(self.strip)
        outer.addLayout(strip_row)

        body = QHBoxLayout()
        body.setContentsMargins(24, 12, 24, 16)
        body.setSpacing(24)
        body.addLayout(self._build_left(), 1)
        body.addWidget(self._build_right())
        outer.addLayout(body, 1)

        self.calibration = CalibrationPanel(self.video)
        self.calibration.hide()
        self._install_shortcuts()

        # Pipeline: câmera -> processamento -> interface
        self.source = source
        if self.source is None and camera_index is not None:
            self.source = CaptureThread(camera_index)
        self.worker = None
        if self.source is not None:
            if isinstance(self.source, CaptureThread):
                self.source.opened.connect(self._on_camera_opened)
                self.source.failed.connect(self._on_camera_failed)
            self.worker = ProcessingThread(engine, self.source)
            self.worker.snapshot.connect(self._on_snapshot)
            self.worker.inference.connect(self._on_inference)
            self.worker.cleared.connect(lambda: self.log("Gesto: punho fechado → desenho limpo", "warn"))
            self.worker.message.connect(self.log)
            self.worker.hsv_changed.connect(self.calibration.set_range)
            self.calibration.changed.connect(lambda lo, hi: self.worker.submit(self._set_hsv, lo, hi))
            if isinstance(self.source, QThread):
                self.source.start()
            self.worker.start()

        self.backend_thread = BackendThread(backend_factory)
        self.backend_thread.log.connect(lambda msg: self.log(f"[boot] {msg}", "warn"))
        self.backend_thread.ready.connect(self._on_backend_ready)
        self.backend_thread.failed.connect(self._on_backend_failed)
        self._set_npu_status(f"NPU: INICIANDO ({backend_label})", MUSTARD)
        self.backend_thread.start()

        self.anim = QTimer(self)
        self.anim.timeout.connect(self._animate)
        self.anim.start(33)
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self._refresh_telemetry)
        self.telemetry_timer.start(250)

    # ---------------- construção
    def _build_header(self, link_text):
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(70)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(10)
        title = QLabel("Eureka · Air-Draw NPU")
        title.setObjectName("AppTitle")
        row.addWidget(title)
        row.addSpacing(24)
        plug = QLabel()
        plug.setPixmap(qta.icon("fa5s.plug", color=MUTED).pixmap(16, 16))
        row.addWidget(plug)
        # Porta + status da NPU num só rótulo, que pode encolher (corta o texto em vez de alargar a janela)
        self.link_text = link_text
        self.npu_status = QLabel()
        self.npu_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.npu_status, 1)

        self.btn_pause = button("Pausar", "fa5s.pause", "GhostBtn", MUTED)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear = button("Limpar", "fa5s.eraser", "GhostBtn", MUTED)
        self.btn_infer = button("Inferir", "fa5s.bolt", "GhostBtn", MUTED)
        self.btn_input = button("Entrada", "fa5s.hand-pointer", "GhostBtn", MUTED)
        self.btn_calib = button("Calibrar cor", "fa5s.sliders-h", "GhostBtn", MUTED)
        self.btn_full = button("Tela cheia", "fa5s.expand", "GhostBtn", MUTED)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_infer.clicked.connect(lambda: self._submit(self.engine.infer))
        self.btn_input.clicked.connect(lambda: self._submit(self.worker.cycle_input) if self.worker else None)
        self.btn_calib.clicked.connect(self.toggle_calibration)
        self.btn_full.clicked.connect(self.toggle_fullscreen)
        for b in (self.btn_pause, self.btn_clear, self.btn_infer, self.btn_input, self.btn_calib, self.btn_full):
            row.addWidget(b)
        return header

    def _build_left(self):
        left = QVBoxLayout()
        left.setSpacing(10)
        top = section_title("fa5s.video", "Câmera")
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet(f"color: {BLUE}; font-weight: bold; padding-left: 12px;")
        top.insertWidget(2, self.mode_label)
        # Gesto reconhecido e estado de cada dedo (I = indicador, M = médio, A = anelar, m = mínimo)
        self.finger_label = QLabel("")
        self.finger_label.setStyleSheet("padding-left: 16px;")
        # Ocupa a sobra da linha e corta o texto se faltar espaço, em vez de alargar a janela
        self.finger_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.insertWidget(3, self.finger_label, 1)
        self._fingers = None
        self.cam_label = QLabel("")
        self.cam_label.setProperty("class", "Muted")
        top.addWidget(self.cam_label)
        left.addLayout(top)

        frame, lay = panel()
        lay.setContentsMargins(1, 1, 1, 1)
        self.video = VideoView()
        self.video.clicked.connect(lambda x, y: self._submit(self.worker.calibrate, x, y) if self.worker else None)
        lay.addWidget(self.video)
        left.addWidget(frame, 1)

        hints = QLabel("indicador: desenha  ·  indicador + médio: move sem riscar  ·  punho (0,8 s): limpa  ·  "
                       "[ESPAÇO] pausa  [L] ao vivo  [T] mão/cor  [M] máscara  [F11] tela cheia")
        hints.setProperty("class", "Muted")
        hints.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        left.addWidget(hints)

        left.addSpacing(6)
        left.addLayout(section_title("fa5s.terminal", "Log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setFixedHeight(112)
        left.addWidget(self.log_view)
        return left

    def _build_right(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        npu_title = section_title("fa5s.microchip", "Imagem enviada à FPGA")
        self.npu_title_label = npu_title.itemAt(1).widget()
        self.npu_sub_label = QLabel("28×28 int8")
        self.npu_sub_label.setProperty("class", "Muted")
        npu_title.addWidget(self.npu_sub_label)
        col.addLayout(npu_title)
        row = QHBoxLayout()
        row.setSpacing(10)
        frame, lay = panel()
        self.npu_view = NpuInputView()
        lay.addWidget(self.npu_view, 0, Qt.AlignCenter)
        row.addWidget(frame)
        pframe, play = panel()
        self.pred_caption = QLabel("RESPOSTA DA FPGA")
        self.pred_caption.setProperty("class", "SectionTitle")
        self.pred_caption.setAlignment(Qt.AlignCenter)
        self.source_label = QLabel("")  # "calculado na NPU da FPGA · 12,4 ms"
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet(f"color: {ORANGE}; font-size: 11px; font-weight: bold;")
        self.digit = QLabel("–")
        self.digit.setAlignment(Qt.AlignCenter)
        self.digit.setStyleSheet(f"color: {DIM}; font-size: 96px; font-weight: bold;")
        self.conf = QLabel("aguardando desenho")
        self.conf.setAlignment(Qt.AlignCenter)
        self.conf.setProperty("class", "Muted")
        self.conf.setWordWrap(True)  # "confiança xx% · ao vivo" quebra a linha em vez de alargar a coluna
        play.addWidget(self.pred_caption)
        play.addWidget(self.digit, 1)
        play.addWidget(self.conf)
        play.addWidget(self.source_label)
        row.addWidget(pframe, 1)
        col.addLayout(row)

        col.addSpacing(6)
        conf_title = section_title("fa5s.chart-bar", "Confiança (logits da FPGA)")
        self.conf_title_label = conf_title.itemAt(1).widget()
        col.addLayout(conf_title)
        bframe, blay = panel()
        self.bars = ConfidenceBars()
        blay.addWidget(self.bars)
        col.addWidget(bframe)

        col.addSpacing(6)
        col.addLayout(section_title("fa5s.tachometer-alt", "Telemetria"))
        self.telemetry = QTableWidget(len(self.TELEMETRY), 2)
        self.telemetry.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.telemetry.verticalHeader().setVisible(False)
        self.telemetry.verticalHeader().setDefaultSectionSize(26)
        self.telemetry.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.telemetry.setSelectionMode(QAbstractItemView.NoSelection)
        self.telemetry.setFocusPolicy(Qt.NoFocus)
        self.telemetry.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.telemetry.verticalHeader().setMinimumSectionSize(24)
        self._fit_telemetry()
        self.telemetry.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for i, name in enumerate(self.TELEMETRY):
            k = QTableWidgetItem(name)
            k.setForeground(QColor(MUTED))
            v = QTableWidgetItem("--")
            v.setForeground(QColor(TEAL))
            self.telemetry.setItem(i, 0, k)
            self.telemetry.setItem(i, 1, v)
        col.addWidget(self.telemetry)
        col.addStretch()

        container = QWidget()
        container.setLayout(col)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(430)
        return scroll

    # ---------------- ações (comandos para a thread de processamento)
    def _submit(self, fn, *args):
        if self.worker:
            self.worker.submit(fn, *args)
        else:
            fn(*args)

    def _set_hsv(self, lower, upper):
        self.engine.tracker.lower = np.array(lower, np.uint8)
        self.engine.tracker.upper = np.array(upper, np.uint8)

    def _toggle_pen(self):
        self.engine.pen_enabled = not self.engine.pen_enabled

    def clear(self):
        self._submit(self.engine.clear)
        self._reset_prediction()

    def toggle_calibration(self):
        if self.calibration.isVisible():
            self.calibration.hide()
            if self.engine.calibration_path:
                self._submit(self.engine.tracker.save, self.engine.calibration_path)
        else:
            self.calibration.set_range(self.engine.tracker.lower, self.engine.tracker.upper)
            self._place_calibration()
            self.calibration.show()
            self.calibration.raise_()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def toggle_pause(self):
        """Pausa ou retoma o desenho (a caneta não risca enquanto pausada)."""
        self._submit(self._toggle_pen)

    def toggle_mask(self):
        if self.worker:
            self.worker.show_mask = not self.worker.show_mask

    def _install_shortcuts(self):
        """Atalhos que valem com o foco em qualquer parte da janela, inclusive no log e na tabela."""
        actions = {
            "Space": self.toggle_pause,
            "R": self.clear,
            "I": lambda: self._submit(self.engine.infer),
            "T": lambda: self._submit(self.worker.cycle_input) if self.worker else None,
            "L": lambda: self._submit(self.worker.toggle_live) if self.worker else None,
            "M": self.toggle_mask,
            "C": self.toggle_calibration,
            "F11": self.toggle_fullscreen,
            "Q": self.close,
            "Esc": self.close,
        }
        for keys, action in actions.items():
            QShortcut(QKeySequence(keys), self).activated.connect(action)

    # ---------------- eventos
    def _on_snapshot(self, snap):
        self.last = snap
        self.video.set_snapshot(snap)
        self.strip.set_state(snap.point is not None, snap.pen_down, snap.img28 is not None)
        if snap.version != self._version:
            self._version = snap.version
            self.npu_view.set_image(snap.img28)
            if not snap.strokes:
                self._reset_prediction()
        mode_state = (snap.input_label, snap.pen_enabled, snap.live)
        previous = getattr(self, "_mode_state", None)
        if mode_state != previous:
            self._mode_state = mode_state
            pen = "" if snap.pen_enabled else "  ·  PAUSADO"
            live = "  ·  AO VIVO" if snap.live else ""
            self.mode_label.setText(f"ENTRADA: {snap.input_label}{live}{pen}")
            self.btn_input.setText(f" Entrada: {'Mão' if snap.input_label.startswith('MÃO') else 'Cor'}")
            paused = not snap.pen_enabled
            self.btn_pause.setText(" Retomar" if paused else " Pausar")
            self.btn_pause.setIcon(qta.icon("fa5s.play" if paused else "fa5s.pause", color=MUSTARD if paused else MUTED))
            if previous is not None and previous[1] != snap.pen_enabled:
                self.log("Desenho pausado ([ESPAÇO] retoma)" if paused else "Desenho retomado", "warn" if paused else "info")
        if (snap.gesture, snap.fingers) != self._fingers:
            self._fingers = (snap.gesture, snap.fingers)
            if snap.fingers is None:
                self.finger_label.setText(f"<span style='color:{DIM}'>sem mão</span>" if snap.gesture is None
                                          and snap.input_label.startswith("MÃO") else "")
            else:
                color = {"desenhar": GREEN, "mover": TEAL, "punho": MUSTARD}.get(snap.gesture, MUTED)
                dots = "&nbsp;".join(f"<span style='color:{MUTED}'>{n}</span><span style='color:{GREEN if up else DIM}'>●</span>"
                                     for n, up in zip("IMAm", snap.fingers))
                self.finger_label.setText(f"<span style='color:{color}; font-weight:bold'>{snap.gesture}</span>"
                                          f"&nbsp;&nbsp;&nbsp;{dots}")
        if snap.error and snap.error != self._last_error:
            self.log(f"Erro na NPU: {snap.error}", "error")
        self._last_error = snap.error
        self.worker.pending = False

    def _on_inference(self, pred, latency_ms, live=False):
        # Ao vivo (durante o desenho) em teal; resultado final (ao parar) em verde
        self.strip.pulse(pred.digit, latency_ms)
        self.source_label.setText(f"calculado {self._hw_where} · {latency_ms:.1f} ms")
        self.digit.setText(str(pred.digit))
        self.digit.setStyleSheet(f"color: {TEAL if live else GREEN}; font-size: 96px; font-weight: bold;")
        self.conf.setText(f"confiança {pred.probs[pred.digit]:.1f}%" + ("\nao vivo" if live else ""))
        self.bars.set_values(pred.probs, pred.logits)
        self.npu_view.pulse()
        if live:
            # Registra só quando o palpite muda, para o log não encher durante o desenho
            if pred.digit != getattr(self, "_live_digit", None):
                self._live_digit = pred.digit
                self.log(f"[ao vivo] → {pred.digit}  ({pred.probs[pred.digit]:.0f}%, {latency_ms:.1f} ms)", "info")
        else:
            self._live_digit = None
            logits = " ".join(f"{int(v):+d}" for v in pred.logits)
            self.log(f"[final] TX 0xFF + 784 B → RX 10 B  [{logits}]  →  {pred.digit}   ({latency_ms:.1f} ms)", "tx")

    def _reset_prediction(self):
        self.source_label.setText("")
        self.digit.setText("–")
        self.digit.setStyleSheet(f"color: {DIM}; font-size: 96px; font-weight: bold;")
        self.conf.setText("aguardando desenho")
        self.bars.reset()

    def _set_npu_status(self, text, color):
        self.npu_status.setText(f"<span style='color:{MUTED}; font-weight:bold'>{self.link_text}</span>"
                                f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{color}'>●</span>&nbsp; "
                                f"<span style='color:{color}; font-weight:bold; font-size:11px'>{text}</span>")

    def _on_backend_ready(self, backend):
        self._submit(setattr, self.engine, "backend", backend)
        is_fpga = backend.label.startswith("FPGA")
        # Deixa explícito onde a inferência roda: na FPGA ou, com --sim, na emulação no computador
        self._hw_where = "na NPU da FPGA" if is_fpga else "na NPU emulada (computador)"
        self.strip.set_backend(is_fpga)
        self.npu_title_label.setText("Imagem enviada à FPGA" if is_fpga else "Imagem enviada à NPU")
        self.npu_sub_label.setText("28×28 int8" if is_fpga else "emulada · 28×28")
        self.pred_caption.setText("RESPOSTA DA FPGA" if is_fpga else "RESPOSTA (EMULADA)")
        self.conf_title_label.setText("Confiança (logits da FPGA)" if is_fpga else "Confiança (logits emulados)")
        self.source_label.setStyleSheet(f"color: {ORANGE if is_fpga else MUSTARD}; font-size: 11px; font-weight: bold;")
        self._set_npu_status("NPU: ONLINE (FPGA)" if is_fpga else "NPU: SIMULAÇÃO LOCAL", GREEN if is_fpga else BLUE)
        self.log(f"NPU pronta: {backend.label}", "ok")

    def _on_backend_failed(self, message):
        self._set_npu_status("NPU: OFFLINE", RED)
        self.log(f"Falha ao preparar a NPU: {message}", "error")
        self.log("No Windows, feche o GUI-TCC (porta COM exclusiva). Sem placa: --sim", "info")

    def _on_camera_opened(self, w, h, fps):
        self.cam_text = f"{w}×{h}"
        self.log(f"Câmera aberta: {w}×{h} @ {fps:.0f} fps", "ok")

    def _on_camera_failed(self, message):
        self.video.message = message
        self.video.update()
        self.log(message, "error")

    def _animate(self):
        self.strip.tick()
        self.npu_view.tick()
        self.bars.tick()

    def _refresh_telemetry(self):
        s = self.last
        if s is None:
            return
        values = (s.backend_label or self.backend_label,
                  f"{s.latency_ms:.1f} ms" if s.latency_ms is not None else "--",
                  str(s.inferences),
                  f"{s.bytes_tx:,} B".replace(",", "."),
                  f"{s.bytes_rx:,} B".replace(",", "."),
                  f"{self.cam_text} · {s.fps:.0f} fps",
                  f"{s.proc_ms:.1f} ms / quadro")
        for i, value in enumerate(values):
            self.telemetry.item(i, 1).setText(value)
        self.cam_label.setText(f"{s.fps:.0f} fps · {s.proc_ms:.0f} ms")

    def log(self, text, kind="info"):
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendHtml(f"<span style='color:{DIM}'>{stamp}</span>&nbsp; "
                                 f"<span style='color:{LOG_COLORS.get(kind, TEXT)}'>{text}</span>")

    def _place_calibration(self):
        self.calibration.move(self.video.width() - self.calibration.width() - 12, 12)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_calibration()

    def _fit_telemetry(self):
        """Altura exata da tabela (linhas + cabeçalho), medida depois de o estilo ser aplicado."""
        t = self.telemetry
        t.ensurePolished()
        t.horizontalHeader().ensurePolished()
        t.setFixedHeight(t.verticalHeader().length() + t.horizontalHeader().sizeHint().height()
                         + 2 * t.frameWidth() + 2)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_telemetry()  # o cabeçalho só tem a altura final (padding + borda teal) com o estilo aplicado

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if isinstance(self.source, QThread):
            self.source.stop()
        self.backend_thread.wait(3000)
        if self.engine.backend:
            self.engine.backend.close()
        if self.engine.hand:
            self.engine.hand.close()
        super().closeEvent(event)

    def show_initial(self, fullscreen=False):
        screen = QApplication.primaryScreen().availableGeometry()
        if fullscreen:
            self.showFullScreen()
        elif screen.width() >= 1640 and screen.height() >= 940:
            self.resize(1600, 900)
            self.show()
        else:
            self.showMaximized()
