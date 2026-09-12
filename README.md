# Eureka: dígitos desenhados no ar, reconhecidos pela NPU na FPGA

Protótipo derivado do experimento 7 do [GUI-TCC](https://github.com/RISC-V-Azedinha/GUI-TCC). Em vez de desenhar com o mouse, você desenha um dígito **no ar** com o dedo indicador (ou com um bastão colorido). A câmera rastreia o movimento e o traço aparece sobre a imagem. O computador converte esse traço em uma imagem 28×28 e a envia à **NPU do SoC RISC-V na FPGA**, que faz a inferência da CNN e devolve a previsão.

```
câmera (até 1280x720) → mão (MediaPipe) ou cor (HSV) → traços → recorte + 20x20 + centralização em 28x28 → int8
      → UART (921600 baud) → NPU: Conv2D 3x3 + ReLU → Fully Connected → 10 logits → previsão
```

Para o funcionamento interno em detalhes (threads, rastreio, pré-processamento, protocolo serial e desempenho), veja [`FUNCIONAMENTO.md`](FUNCIONAMENTO.md).

A placa usa o mesmo firmware, o mesmo protocolo serial e os mesmos pesos int8 do experimento 7. O firmware (`firmware/cnn_server.bin`), os pesos (`weights.npz`) e os modelos (`models/`) vêm neste repositório, então o Eureka não depende da pasta do GUI-TCC. Por padrão, a inferência roda **na FPGA**. Com `--sim`, a NPU é emulada no computador com a mesma aritmética inteira, para testar sem a placa.

## Gestos

| Gesto | Ação |
|---|---|
| **só o indicador levantado** | desenha (a ponta do indicador é a caneta) |
| **indicador + médio levantados** | move sem riscar (entre traços, como no "4" ou no "7" com corte) |
| **punho fechado por 0,8 s** | limpa o desenho. Um anel em volta da palma mostra o progresso. Para limpar de novo, abra a mão e feche outra vez |

A inferência roda sozinha quando o desenho fica parado por 0,8 s (ajuste com `--idle`). O cursor fica verde e cheio enquanto desenha, e vazado enquanto só move.

Para saber se um dedo está levantado, o Eureka mede o **ângulo de dobra do dedo nos pontos 3D** que o MediaPipe estima (em metros). Assim, o gesto funciona para qualquer lado que o dedo aponte, inclusive com o indicador **apontado para a câmera**. Ao lado do rótulo da câmera aparecem o gesto reconhecido e o estado de cada dedo (I = indicador, M = médio, A = anelar, m = mínimo), útil para entender por que a caneta não abaixou.

O rastreio usa o filtro **One Euro**, que segura o tremor com o dedo parado e quase não atrasa com o dedo rápido. Falhas curtas do detector, de até 3 quadros, não interrompem o traço.

Com **T**, dá para trocar para o rastreio por cor (bastão). Nesse modo, o ideal é que só a ponta do bastão seja colorida (modo *centro*).

## Desempenho

O programa roda em três threads, para a interface nunca travar:

```
captura da câmera ──(só o quadro mais recente)──> processamento ──(quadro pronto)──> interface
                                                   rastreio, traços, NPU               só desenha
```

- **Captura:** descarta os quadros atrasados em vez de enfileirá-los, então a imagem nunca fica "para trás" da mão.
- **Processamento:** o MediaPipe, o pré-processamento e a comunicação com a FPGA rodam fora da thread da interface.
- **Interface:** desenha o traço como linhas vetoriais e passa o quadro da câmera direto para a tela, sem cópia e sem conversão de cor.

A telemetria mostra o fps e o tempo de processamento por quadro.

## Windows (executável)

Cada versão publicada traz o `Eureka-windows-x64.zip` na página de [Releases](https://github.com/RISC-V-Azedinha/Digit-Tracker/releases). Não é preciso instalar o Python. Basta extrair o zip e abrir:

- `Eureka.exe`: inferência na FPGA, com a porta detectada automaticamente;
- `Eureka-sim.bat`: sem placa, com a NPU emulada (o mesmo que `--sim`).

As outras opções da seção [Uso](#uso) também valem para o `.exe`, por exemplo num atalho ou no terminal: `Eureka.exe --port COM4`. A calibração de cor (`calibration.json`) é salva ao lado do `.exe`.

O executável é gerado pelo GitHub Actions (`.github/workflows/release.yml`), com o PyInstaller (`Eureka.spec`). Antes de publicar, o workflow abre o `.exe` com `--selftest`: esse modo carrega o MediaPipe, faz uma inferência na NPU emulada, abre a janela e fecha. Para publicar uma versão:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

Rodar o workflow manualmente (aba Actions → *Run workflow*) só gera o zip como artifact da execução, sem criar uma release.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps mediapipe==1.0.1
```

O MediaPipe é instalado com `--no-deps` de propósito. Ele declara como dependências o `opencv-contrib-python`, que traz plugins Qt próprios e quebra o PyQt5, e o `sounddevice`, que não é usado. As dependências de que ele realmente precisa já estão no `requirements.txt`. Pelo mesmo motivo, o OpenCV instalado é o `opencv-python-headless`.

O modelo de rastreio da mão (`models/hand_landmarker.task`, 7,8 MB) já vem na pasta. Sem ele, ou sem o MediaPipe, o Eureka avisa no log e usa o rastreio por cor.

### Pesos da NPU (`weights.npz`)

O arquivo `weights.npz` já vem gerado. Para regenerá-lo a partir da rede pré-treinada (`models/cnn_pretrained.pth`), por exemplo depois de treinar a rede de novo no GUI-TCC e copiar o `.pth` para cá, rode o script abaixo. Ele precisa do torch, que o Eureka em si não usa, e baixa o MNIST em `data/` para a calibração int8:

```bash
pip install torch torchvision
python export_weights.py
```

### Executável local

Para gerar o executável na própria máquina, como o workflow faz (o resultado fica em `dist/Eureka/`):

```bash
pip install -r requirements-windows.txt
pip install --no-deps mediapipe==1.0.1
pyinstaller Eureka.spec --noconfirm --workpath build/pyinstaller --distpath dist
```

## Uso

```bash
python eureka.py                      # NPU na FPGA, com a porta detectada automaticamente
python eureka.py --port COM4          # FPGA em uma porta específica (Linux: --port /dev/ttyUSB1)
python eureka.py --sim                # sem placa: NPU emulada no computador
python eureka.py --input cor          # começa no rastreio por cor (bastão)
python eureka.py --camera 1           # outra câmera
python eureka.py --fullscreen         # tela cheia (F11 alterna)
python eureka.py --scale 1.5          # interface maior em telas de alta resolução
```

Sem `--port`, o Eureka usa a porta USB-serial de maior número. Nas placas com FTDI duplo (JTAG + UART), essa é a UART. O boot da FPGA roda em segundo plano e aparece no **Log**. No Windows, feche o GUI-TCC antes, porque a porta COM só pode ser aberta por um programa de cada vez.

### Teclas

Os botões do cabeçalho (Limpar, Inferir, Entrada, Calibrar cor, Tela cheia) fazem o mesmo que as teclas.

| Tecla | Função |
|---|---|
| **r** | limpa o desenho |
| **i** | força a inferência agora |
| **t** | alterna a entrada: mão → cor (ponta) → cor (centro) |
| **espaço** | pausa ou retoma a caneta |
| **c** | painel de calibração HSV (modo cor). No modo cor, clicar no bastão calibra a cor sozinho |
| **m** | mostra a máscara da cor rastreada (modo cor) |
| **F11** | tela cheia |
| **q** / **Esc** | sai |

### Dicas

- **Mão:** a mão deve estar inteira na imagem e bem iluminada, de frente para a câmera.
- **Movimento:** desenhe com movimentos do braço, não só do pulso, e faça dígitos grandes. O tamanho não importa para a NPU, porque a espessura do traço é normalizada.
- **Gestos:** deixe os dedos que não estão sendo usados bem dobrados. Um médio meio esticado pode ser lido como "mover".

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `eureka.py` | ponto de entrada: argumentos, escolha da NPU (FPGA ou emulada), abertura da janela e `--selftest` |
| `paths.py` | caminhos dos arquivos no código-fonte e no executável |
| `gui.py` | interface PyQt5 (estilo do GUI-TCC): vídeo com o traço, entrada 28×28, predição, confiança, telemetria e log |
| `pipeline.py` | threads de captura e de processamento; a interface recebe um `Snapshot` pronto por quadro |
| `engine.py` | núcleo sem interface: processa cada quadro, controla a caneta e os gestos, dispara a inferência |
| `hand.py` | rastreio da mão com o MediaPipe (`HandTracker`): gestos por dedos levantados e filtro One Euro |
| `vision.py` | rastreio por cor (`MarkerTracker`), traços (`StrokeCanvas`) e pré-processamento 28×28 (`to_npu_input`) |
| `npu.py` | `FpgaNpu` (placa, protocolo do `core/npu_driver.py`) e `CpuNpu` (emulação) |
| `export_weights.py` | gera `weights.npz` a partir de `models/cnn_pretrained.pth` (calibração int8 do GUI-TCC) |
| `weights.npz` | pesos int8 da NPU |
| `firmware/cnn_server.bin` | firmware do servidor CNN do SoC RISC-V (cópia do `artifacts/cnn_server.bin` do GUI-TCC) |
| `models/hand_landmarker.task` | modelo do MediaPipe para detectar a mão |
| `models/cnn_pretrained.pth` | rede em ponto flutuante do experimento 7, usada só pelo `export_weights.py` |
| `Eureka.spec`, `requirements-windows.txt` | build do executável com o PyInstaller |
| `.github/workflows/release.yml` | gera o `.exe` para Windows e publica a release a cada tag `v*` |
