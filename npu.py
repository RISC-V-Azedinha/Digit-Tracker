"""
Backends de inferência do Eureka.

- FpgaNpu: envia a imagem 28x28 para a NPU do SoC RISC-V, com o mesmo firmware e o mesmo
  protocolo do experimento 7 do GUI-TCC (artifacts/cnn_server.bin + core/npu_driver.py).
- CpuNpu: emula no computador a aritmética inteira da NPU, para testar sem a placa.
"""
import os
import struct
import time

import numpy as np
import serial


def load_weights(path):
    """Pesos int8 já calibrados para a NPU (gerados por export_weights.py)."""
    data = np.load(path)
    return {key: data[key] for key in ("w_conv", "b_conv", "w_fc", "b_fc")}


def pack_weights_dma(w_int8):
    """Empacota 4 pesos int8 por palavra de 32 bits, no formato que o DMA da NPU espera."""
    out_features, in_features = w_int8.shape
    packed = []
    for start in range(0, out_features, 4):
        lanes = min(4, out_features - start)
        for k in range(in_features):
            word = 0
            for lane in range(lanes):
                word |= (int(w_int8[start + lane, k]) & 0xFF) << (8 * lane)
            packed.append(word)
    return packed


class CpuNpu:
    """
    Emulação da NPU: Conv2D 3x3/stride 2 (4 filtros) + ReLU e Fully Connected (676 -> 10),
    com o pós-processamento do PPU aproximado por deslocamento de 8 bits e saturação int8.
    """
    label = "SIMULACAO (CPU)"

    def __init__(self, weights):
        self.w_conv = weights["w_conv"].astype(np.int32)  # (4, 9)
        self.b_conv = weights["b_conv"].astype(np.int32)  # (4,)
        self.w_fc = weights["w_fc"].astype(np.int32)      # (10, 676), entrada channels-last
        self.b_fc = weights["b_fc"].astype(np.int32)      # (10,)

    def infer(self, image_int8):
        x = image_int8.reshape(28, 28).astype(np.int32)
        patches = np.lib.stride_tricks.sliding_window_view(x, (3, 3))[::2, ::2].reshape(13, 13, 9)
        y1 = np.clip((patches @ self.w_conv.T + self.b_conv) >> 8, 0, 127)  # Conv2D + ReLU
        logits = np.clip((self.w_fc @ y1.reshape(-1) + self.b_fc) >> 8, -128, 127)
        return logits.astype(np.int8)

    def close(self):
        pass


class FpgaNpu:
    """Conexão com o servidor CNN rodando no SoC RISC-V (protocolo do core/npu_driver.py do GUI-TCC)."""

    def __init__(self, port, baud, firmware, weights, log=print):
        self.label = f"FPGA {port}"
        # rtscts=False: o pino RTS controla o reset/halt da placa manualmente
        self.ser = serial.Serial(port, baud, rtscts=False, dsrdtr=False, timeout=2.0)
        try:
            self.ser.reset_input_buffer()
            self._boot(firmware, log)
            self._upload_weights(weights, log)
        except Exception:
            self.ser.close()
            raise

    def _boot(self, firmware, log):
        """Reseta a FPGA para o bootloader (ROM) e envia o firmware do servidor CNN."""
        log("Reset de hardware: aguardando o bootloader...")
        self.ser.rts = False
        time.sleep(0.05)
        self.ser.write(b'\xCA\xFE\xBA\xBE')  # assume o controle da FSM de debug
        time.sleep(0.05)
        self.ser.write(b'\x09\x00\x00\x00\x00')  # PC de boot = ROM
        time.sleep(0.01)
        self.ser.write(b'\x08')  # reset
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.rts = True  # solta a placa para rodar a ROM

        buffer = ""
        start = time.time()
        while "BOOT" not in buffer:
            if time.time() - start > 4.0:
                raise TimeoutError("Timeout aguardando o bootloader da FPGA ('BOOT').")
            if self.ser.in_waiting:
                buffer += self.ser.read(1).decode("utf-8", errors="ignore")

        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.write(b'\xCA\xFE\xBA\xBE')
        ack = b''
        start = time.time()
        while time.time() - start < 2.0:
            if self.ser.in_waiting:
                ack = self.ser.read(1)
                break
        if ack != b'!':
            raise ConnectionError(f"Sem resposta de handshake do bootloader (recebido: {ack!r}).")

        size = os.path.getsize(firmware)
        log(f"Enviando {os.path.basename(firmware)} ({size} bytes)...")
        self.ser.write(struct.pack('<I', size))
        time.sleep(0.05)
        with open(firmware, "rb") as f:
            payload = f.read()
        for i in range(0, len(payload), 64):
            self.ser.write(payload[i:i + 64])
            self.ser.flush()
            time.sleep(0.002)
        time.sleep(0.5)
        self.ser.reset_input_buffer()

    def _send_block(self, command, values, fmt, ack):
        self.ser.write(struct.pack('>B', command))
        for value in values:
            self.ser.write(struct.pack(fmt, value))
        reply = self.ser.read(1)
        if reply != ack:
            raise ConnectionError(f"A NPU não confirmou o bloco 0x{command:02X} (recebido: {reply!r}).")

    def _upload_weights(self, weights, log):
        """Transfere pesos e bias da Conv2D e da camada FC para a NPU via DMA."""
        log("Enviando os pesos da CNN para a NPU...")
        self._send_block(0xAA, [v & 0xFFFFFFFF for v in pack_weights_dma(weights["w_conv"])], '>I', b'A')
        self._send_block(0xBB, [int(v) for v in weights["b_conv"]], '>i', b'B')
        self._send_block(0xCC, [v & 0xFFFFFFFF for v in pack_weights_dma(weights["w_fc"])], '>I', b'C')
        b_fc = np.pad(weights["b_fc"], (0, 12 - len(weights["b_fc"])), mode='constant')  # 3 blocos de 4
        self._send_block(0xDD, [int(v) for v in b_fc], '>i', b'D')
        log("NPU pronta para inferência.")

    def infer(self, image_int8):
        self.ser.write(struct.pack('>B', 0xFF))
        self.ser.write(image_int8.tobytes())
        reply = self.ser.read(10)
        if len(reply) != 10:
            raise TimeoutError(f"A NPU respondeu {len(reply)} de 10 bytes.")
        return np.array(struct.unpack('>10b', reply), dtype=np.int8)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
