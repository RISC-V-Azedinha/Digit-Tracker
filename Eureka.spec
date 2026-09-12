# Eureka.spec - build do executável com PyInstaller
#   pyinstaller Eureka.spec --noconfirm --workpath build/pyinstaller --distpath dist
# Gera dist/Eureka/Eureka.exe (modo onedir: abre rápido, sem extrair o MediaPipe a cada execução).
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ('weights.npz', '.'),
    ('firmware/cnn_server.bin', 'firmware'),
    ('models/hand_landmarker.task', 'models'),
]
# O MediaPipe carrega a própria biblioteca nativa (mediapipe/tasks/c/libmediapipe.dll) por
# ctypes, fora da análise de imports do PyInstaller: ela precisa ir junto, na mesma subpasta.
datas += collect_data_files('mediapipe', excludes=['**/test/**'])
binaries = collect_dynamic_libs('mediapipe')

a = Analysis(
    ['eureka.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    excludes=['tkinter', 'torch', 'torchvision'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Eureka',
    console=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name='Eureka',
)
