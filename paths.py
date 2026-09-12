"""
Caminhos dos arquivos do Eureka, no código-fonte e no executável (PyInstaller).

- resource_path: arquivos distribuídos com o programa (pesos, firmware, modelo da mão). No
  executável, ficam na pasta do bundle (sys._MEIPASS).
- user_path: arquivos gravados pelo programa (calibração, selftest.log). No executável, ficam ao
  lado do .exe, porque a pasta do bundle é descartável.
"""
import os
import sys

FROZEN = getattr(sys, "frozen", False)
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    return os.path.join(sys._MEIPASS if FROZEN else SOURCE_DIR, *parts)


def user_path(*parts):
    return os.path.join(os.path.dirname(sys.executable) if FROZEN else SOURCE_DIR, *parts)
