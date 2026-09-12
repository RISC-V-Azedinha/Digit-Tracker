# Como o Eureka funciona

Este documento descreve o funcionamento interno do Eureka: da câmera até a previsão devolvida pela NPU na FPGA. Para instalar e usar, veja o [README](README.md).

## 1. Visão geral

O Eureka reconhece dígitos desenhados **no ar**. A câmera acompanha a ponta do dedo indicador (ou de um bastão colorido). Os pontos percorridos formam traços, que são convertidos em uma imagem 28×28 no formato do MNIST e enviados pela UART à NPU do SoC RISC-V na FPGA. A NPU executa a CNN e devolve 10 valores (logits), um por dígito. O maior deles é a previsão.

```
 câmera ──> rastreio ──> traços ──> imagem 28x28 int8 ──UART──> NPU (FPGA) ──> 10 logits ──> previsão
 1280x720    mão ou cor   pontos     recorte + escala           Conv2D + ReLU               softmax
                                     + centralização            + Fully Connected           (confiança)
```

Todo o lado da FPGA é o mesmo do experimento 7 do GUI-TCC: o firmware `cnn_server.bin`, o protocolo serial e os pesos int8. O que o Eureka acrescenta é a forma de entrada (a câmera, em vez do mouse) e a interface.

## 2. Arquitetura em threads

O programa roda em quatro threads, para que a janela nunca espere pela câmera, pelo rastreio ou pela FPGA:

```
 ┌──────────────────┐   quadro mais recente   ┌─────────────────────────┐   Snapshot   ┌───────────────┐
 │ CaptureThread    │ ──────────────────────> │ ProcessingThread        │ ───────────> │ Interface     │
 │ lê a câmera      │   (sem fila)            │ Engine: rastreio,       │  (sinal Qt)  │ (thread       │
 │ (pipeline.py)    │                         │ traços, pré-proc., NPU  │ <─────────── │ principal)    │
 └──────────────────┘                         └─────────────────────────┘   comandos   └───────────────┘
                                                          ▲                 (submit)
                                   NPU pronta (FPGA ou    │
                                   emulada) ──────────────┘
                                          BackendThread (boot da FPGA + envio dos pesos, só no início)
```

| Thread | Arquivo | Responsabilidade |
|---|---|---|
| **CaptureThread** | `pipeline.py` | Abre a câmera (MJPG, 1280×720, buffer do driver com 1 quadro) e lê quadros sem parar. Guarda **só o mais recente**, com um número de sequência, protegido por um `threading.Condition`. |
| **ProcessingThread** | `pipeline.py` | Espera um quadro novo, espelha-o, passa-o ao `Engine` e emite um `Snapshot` com tudo o que a interface precisa desenhar. |
| **Interface** | `gui.py` | Só desenha o `Snapshot` recebido e envia comandos (limpar, pausar, trocar a entrada etc.). |
| **BackendThread** | `gui.py` | Prepara a NPU no início: boot da FPGA e envio dos pesos, ou a criação da emulação. Terminado isso, entrega o backend ao processamento. |

**Por que a captura descarta quadros.** Se o processamento de um quadro demorar mais que o intervalo da câmera (33 ms a 30 fps), uma fila faria a imagem ficar cada vez mais atrasada em relação à mão. Guardando só o último quadro, o atraso máximo fica limitado a um quadro, e os intermediários são simplesmente pulados.

**Comunicação entre threads.**
- A interface **nunca** altera o `Engine` diretamente. Ela chama `ProcessingThread.submit(função, argumentos)`, que coloca a chamada numa fila (`queue.Queue`). A thread de processamento executa essa fila antes de cada quadro, então o `Engine` só é alterado por uma thread.
- O processamento avisa a interface por sinais Qt, entregues na thread principal: `snapshot` (um por quadro), `inference` (a cada inferência), `cleared` (gesto de limpar), `message` (linhas do log) e `hsv_changed` (calibração de cor).
- A flag `pending` evita acumular `Snapshot`s: um novo só é emitido depois de a interface consumir o anterior. Se a interface atrasar, quadros são processados mas não exibidos, sem fila.

## 3. O ciclo de um quadro

Cada quadro passa pelas etapas abaixo, em `Engine.process()` (`engine.py`):

1. **Espelhamento:** `cv2.flip(quadro, 1)`, para que mover a mão para a direita mova o traço para a direita na tela.
2. **Redimensionamento** para 960 px de largura (`PROC_WIDTH`), mantendo a proporção. Com a câmera em 1280×720, o quadro processado fica em 960×540. Todas as coordenadas internas (traços, cursor, ROI) usam esse tamanho.
3. **Rastreio:** o rastreador da mão (seção 4) ou o de cor (seção 5) devolve o ponto da caneta e se ela está abaixada.
4. **Caneta:** a caneta está abaixada quando o rastreador indica isso **e** o desenho não está pausado (tecla espaço). Com a caneta abaixada, o ponto entra no traço atual. Com ela levantada, o traço atual é encerrado e o próximo ponto começa outro.
5. **Pré-processamento:** sempre que os traços mudam, a imagem 28×28 é recalculada (seção 6).
6. **Inferência** (seção 8):
   - **ao vivo:** enquanto a caneta está abaixada e o desenho mudou, uma inferência a cada 0,2 s;
   - **final:** 0,8 s depois do último ponto desenhado, uma única vez por versão do desenho.
7. **Snapshot:** a thread de processamento copia o estado (quadro, traços, cursor, esqueleto da mão, imagem 28×28, previsão e telemetria) e o envia à interface.

## 4. Rastreio da mão (`hand.py`)

### 4.1 Detecção

O rastreio usa o **MediaPipe Hand Landmarker** (API de Tasks, modelo `models/hand_landmarker.task`):

- **Modo `VIDEO`:** aproveita o quadro anterior para seguir a mão em vez de detectá-la do zero a cada quadro. Esse modo exige carimbos de tempo sempre crescentes, garantidos em `detect()`.
- **Configuração:** uma mão por vez, com confianças mínimas de 0,6 (detecção), 0,5 (presença) e 0,5 (rastreio).
- **Tamanho:** a detecção roda numa cópia do quadro com 640 px de largura (`DETECT_WIDTH`). Os pontos voltam normalizados (0 a 1), então valem para o quadro de 960 px.
- **Saída:** o detector devolve **21 pontos** da mão em duas versões:
  - `hand_landmarks`: posição na imagem, usada para o cursor e para desenhar o esqueleto;
  - `hand_world_landmarks`: posição **3D em metros**, estimada pela rede, usada para saber quais dedos estão levantados.

```
        8   12  16  20        pontas:        4 polegar, 8 indicador, 12 médio, 16 anelar, 20 mínimo
        |   |   |   |         articulações:  6, 10, 14, 18 (do meio de cada dedo)
        6   10  14  18        bases:         5, 9, 13, 17
        |   |   |   |         pulso:         0
   4    5---9---13--17
    \   |          /
     2--1----0----            (numeração do MediaPipe)
```

### 4.2 Quais dedos estão levantados

Para cada dedo (exceto o polegar), o Eureka mede o **ângulo de dobra** nos pontos 3D: o ângulo entre o segmento base → articulação do meio e o segmento articulação do meio → ponta.

- Um dedo reto tem dobra perto de 0°, **para qualquer lado que aponte**. Um dedo dobrado passa de 90°.
- **Histerese:** o dedo passa a contar como levantado quando a dobra fica abaixo de **45°**, e como dobrado quando passa de **65°**. Entre os dois limites ele mantém o estado anterior, então um dedo meio dobrado não fica piscando.

O ângulo 3D resolve o caso do **dedo apontado para a câmera**. Nessa pose, a ponta, a articulação e a base se sobrepõem na imagem 2D, e uma regra baseada em distâncias na imagem acha que o dedo está dobrado. Se o MediaPipe não devolver os pontos 3D, o Eureka usa essa regra 2D como alternativa: o dedo conta como levantado quando a distância pulso → ponta passa de 1,15× a distância pulso → articulação (e volta a contar como dobrado abaixo de 1,0×).

### 4.3 Gestos

| Dedos levantados | Gesto | Efeito |
|---|---|---|
| nenhum | **punho** | não risca; segurado por 0,8 s, limpa o desenho |
| indicador, sem o médio | **desenhar** | a ponta do indicador risca |
| qualquer outra combinação (ex.: indicador + médio, mão aberta) | **mover** | move o cursor sem riscar |

A regra é avaliada nessa ordem. O anelar e o mínimo não influenciam o "desenhar", então indicador + mínimo também desenha.

- **Confirmação:** a caneta só abaixa depois de **2 quadros seguidos** no gesto "desenhar", o que filtra leituras isoladas erradas do detector.
- **Início do traço:** no instante em que a caneta abaixa, o filtro de suavização é reiniciado na posição real do dedo, para o traço começar exatamente onde o dedo está, sem um "rabinho" vindo de onde a mão pairava.
- **Falhas do detector:** se a mão some por até **3 quadros** (`LOST_GRACE`), o último ponto e o estado da caneta são mantidos, e o traço não é interrompido. Depois disso, o rastreio é reiniciado.

**Gesto de limpar** (`Engine._update_clear_gesture`):
- Com o punho fechado, o progresso vai de 0 a 1 em **0,8 s**, e a interface mostra um anel em volta da palma.
- Ao completar, limpa **uma vez**. Para limpar de novo, é preciso abrir a mão e fechar outra vez.
- Interrupções de até **0,2 s** no punho, por falhas do detector, não zeram o progresso.

### 4.4 Suavização: filtro One Euro

A posição da ponta do indicador passa por um **filtro One Euro** (Casiez et al., 2012), um em cada eixo. Ele é um passa-baixas cuja frequência de corte cresce com a velocidade:

```
α(fc, dt) = 1 / (1 + τ/dt),  com τ = 1 / (2π·fc)
velocidade filtrada:  dx̂ = α(d_cutoff)·(x − x̂_anterior)/dt + (1 − α(d_cutoff))·dx̂_anterior
corte adaptativo:     fc = min_cutoff + beta·|dx̂|
posição filtrada:     x̂ = α(fc)·x + (1 − α(fc))·x̂_anterior
```

Os parâmetros são `min_cutoff = 1,5 Hz`, `beta = 0,03` e `d_cutoff = 1,0 Hz`:
- **Dedo parado:** a frequência de corte fica perto de 1,5 Hz, uma suavização forte que tira o tremor.
- **Dedo rápido:** a 300 px/s o corte sobe para ~10 Hz, e o atraso fica pequeno.

Num teste com uma mão sintética, o tremor com o dedo parado caiu de 2,0 px para 1,0 px, e o atraso a 300 px/s ficou em ~3,5 px (~12 ms).

## 5. Rastreio por cor (`vision.py`, `MarkerTracker`)

Alternativa ao rastreio da mão, para um bastão colorido (tecla **T**). A cada quadro:

1. **Filtro de cor:** desfoque gaussiano 7×7, conversão para HSV e máscara com a faixa de cor (padrão `H 35–85, S 80–255, V 80–255`, um verde). A máscara é limpa com abertura morfológica e dilatação (elipse 5×5).
2. **Mancha:** é escolhida a maior mancha (contorno externo). Com área abaixo de 400 px², considera-se que o bastão não está na imagem.
3. **Ponto da caneta**, conforme o modo:
   - **ponta:** média dos pontos do contorno até 6 px abaixo do ponto mais alto. É o modo para um bastão todo colorido, segurado apontando para cima.
   - **centro:** centroide da mancha (momentos da imagem). É o modo para um bastão só com a ponta colorida.
4. **Suavização:** média móvel exponencial, com peso 0,45 para a posição anterior.

No modo cor, a caneta fica abaixada sempre que o bastão aparece, e levanta depois de 3 quadros sem vê-lo (`LIFT_AFTER`).

**Calibração:** um clique no bastão amostra a mediana HSV de um quadrado de 13×13 px em volta do ponto e define a faixa como `H ± 12`, `S ≥ max(50, S − 80)` e `V ≥ max(50, V − 80)`. A faixa também pode ser ajustada à mão no painel HSV (tecla **C**) e fica salva em `calibration.json`.

## 6. Traços e imagem 28×28 (`vision.py`)

### 6.1 Traços (`StrokeCanvas`)

Os traços são listas de pontos no quadro de 960 px. Não existe imagem rasterizada do desenho; a interface desenha os traços como linhas vetoriais.

- Um ponto novo só entra se estiver a pelo menos **2 px** do anterior (`min_step`), para não acumular tremor residual.
- Um salto maior que **130 px** (`max_jump`) começa um traço novo, em vez de riscar uma linha longa, o que costuma indicar uma detecção errada.
- Cada mudança incrementa `version`. É por ela que o resto do sistema sabe quando recalcular a imagem 28×28 e quando inferir de novo.

### 6.2 Conversão para a entrada da NPU (`to_npu_input`)

A conversão segue o experimento 7 e o formato do MNIST:

1. **Caixa do dígito:** calcula a caixa que envolve todos os pontos; o **tamanho** do dígito é o maior lado dessa caixa.
2. **Espessura:** redesenha os traços num quadro próprio, com linhas suavizadas de espessura `max(2, round(0,12 × tamanho))`. No MNIST, o traço ocupa cerca de 12% da altura do dígito. Recalcular a espessura assim torna o resultado independente do tamanho com que se desenha no ar.
3. **Recorte:** recorta a região com tinta.
4. **Escala:** redimensiona (`INTER_AREA`) para caber em **20×20**, mantendo a proporção.
5. **Centralização:** cola no centro de uma imagem **28×28**, pela caixa do dígito, como no experimento 7.
6. **int8:** divide cada pixel (0–255) por 2, gerando valores de 0 a 127. É esse vetor de 784 bytes que vai para a NPU.

## 7. NPU (`npu.py`)

### 7.1 A rede

A CNN é a do experimento 7 do GUI-TCC:

```
entrada 28×28×1 (int8)
  → Conv2D: 4 filtros 3×3, stride 2, sem padding   → 13×13×4
  → ReLU
  → reorganização "channels-last" e achatamento     → 676 valores
  → Fully Connected 676 → 10                        → 10 logits (int8)
```

Os pesos em ponto flutuante (`models/cnn_pretrained.pth`, cópia do `artifacts/cnn_pretrained.pth` do GUI-TCC) são quantizados para int8 com a mesma calibração da função `carregar_ou_treinar` do GUI-TCC. O script `export_weights.py` reproduz essa calibração, com um lote de 32 imagens do MNIST, e salva o resultado em `weights.npz`:
- `w_conv`: 4×9, int8;
- `b_conv`: 4, int32;
- `w_fc`: 10×676, int8;
- `b_fc`: 10, int32.

Assim, o Eureka não precisa do PyTorch.

### 7.2 Na FPGA (`FpgaNpu`)

A comunicação reproduz byte a byte o `core/npu_driver.py` do GUI-TCC. A equivalência foi verificada: numa porta serial simulada, os dois produziram a mesma sequência de 2175 eventos (escritas, leituras, RTS e limpezas de buffer) e os mesmos 15.051 bytes.

**Porta:** 921600 baud, sem controle de fluxo por hardware (`rtscts=False`, `dsrdtr=False`), porque o pino RTS é controlado manualmente e comanda o reset da placa. Tempo limite de leitura: 2 s.

**Boot do firmware:**

| Passo | Enviado / esperado | Significado |
|---|---|---|
| 1 | `RTS = 0` | segura a placa |
| 2 | `CA FE BA BE` | assume o controle da máquina de debug |
| 3 | `09 00 00 00 00` | PC de boot = ROM (`0x00000000`) |
| 4 | `08` | reset |
| 5 | limpa a entrada, `RTS = 1` | solta a placa para rodar o bootloader da ROM |
| 6 | espera `BOOT` (até 4 s) | o bootloader anunciou que está pronto |
| 7 | `CA FE BA BE` → espera `!` (até 2 s) | handshake com o bootloader |
| 8 | tamanho do firmware (uint32, little-endian) | |
| 9 | `cnn_server.bin` em blocos de 64 bytes, 2 ms entre blocos | upload para a RAM; o firmware começa a rodar |

**Envio dos pesos:** cada bloco começa com um byte de comando, e o firmware confirma com uma letra.

| Comando | Conteúdo | Confirmação |
|---|---|---|
| `AA` | `w_conv` empacotado: 9 palavras de 32 bits (big-endian) | `A` |
| `BB` | `b_conv`: 4 × int32 (big-endian) | `B` |
| `CC` | `w_fc` empacotado: 3 grupos × 676 = 2028 palavras | `C` |
| `DD` | `b_fc` completado com zeros até 12 × int32 | `D` |

**Empacotamento para o DMA** (`pack_weights_dma`): cada palavra de 32 bits leva o mesmo peso de entrada para **4 saídas consecutivas**, com a saída *k* do grupo nos bits `8k` a `8k+7`. As saídas são percorridas em grupos de 4, completando com zeros o último grupo incompleto. Por isso os 10 neurônios da camada FC viram 3 grupos, e os bias da FC são completados até 12.

**Inferência:**

```
TX: FF + 784 bytes (imagem 28×28 int8)      RX: 10 bytes (logits int8, com sinal)
```

São 785 bytes enviados e 10 recebidos por inferência. Com 10 bits por byte na UART (início + 8 dados + fim), a transmissão leva cerca de 8,5 ms a 921600 baud, antes do tempo de cálculo da NPU.

**Porta automática:** sem `--port`, o Eureka usa a porta USB-serial de maior número, com ordenação numérica (`COM10` vem depois de `COM3`). Nas placas com FTDI duplo, a primeira porta é o JTAG e a segunda é a UART.

### 7.3 Emulação no computador (`CpuNpu`, `--sim`)

Reproduz a aritmética inteira da NPU:

```
conv  = janelas 3×3 com stride 2 da imagem · w_conv + b_conv          (inteiros de 32 bits)
y1    = satura(conv >> 8, 0, 127)                                     (ReLU + saturação int8)
fc    = w_fc · achata(y1) + b_fc
logit = satura(fc >> 8, −128, 127)
```

O deslocamento de 8 bits aproxima o pós-processamento (PPU) da NPU. Nas 2000 primeiras imagens de teste do MNIST, a emulação acertou 96,5%, contra 96,7% do modelo original em ponto flutuante, e concordou com ele em 99,2% das previsões.

## 8. Inferência e confiança (`engine.py`)

### 8.1 Quando inferir

- **Ao vivo:** enquanto a caneta está abaixada, se o desenho mudou desde a última inferência e já se passaram `live_interval` segundos (padrão 0,2 s), a imagem é enviada à NPU. Na interface, o palpite aparece em teal com "ao vivo". A tecla **L** liga e desliga esse modo, e `--live-interval 0` o desativa.
- **Final:** quando o desenho fica parado por `idle` segundos (padrão 0,8 s), acontece uma inferência final. Ela roda **sempre**, uma vez por versão do desenho, mesmo que a última inferência ao vivo já tenha visto o traço completo. O resultado aparece em verde.

As inferências rodam na thread de processamento, então a janela não trava enquanto a FPGA responde.

### 8.2 Confiança

A previsão é o índice do maior logit. A confiança exibida é um softmax com **temperatura 15** sobre os logits int8, a mesma calibração da aba Neural Network do GUI-TCC:

```
confiança_d = exp(logit_d / 15) / Σ exp(logit_i / 15)
```

A temperatura só muda o quanto as porcentagens ficam "espalhadas"; a previsão não muda.

## 9. Interface (`gui.py`)

O visual segue o GUI-TCC: mesma paleta, fonte monoespaçada, botões fantasma no cabeçalho, títulos de seção com ícones do QtAwesome e tabela no estilo do banco de registradores.

| Área | Conteúdo |
|---|---|
| **Cabeçalho** | porta e estado da NPU (iniciando, online, simulação ou offline) e os botões Pausar/Retomar, Limpar, Inferir, Entrada, Calibrar cor e Tela cheia |
| **Câmera** | vídeo com os traços, a caixa ROI → 28×28, o esqueleto da mão, o cursor e o anel do gesto de limpar. Acima dele, o modo de entrada, o gesto reconhecido e o estado de cada dedo (I, M, A, m) |
| **Entrada da NPU** | a imagem 28×28 exatamente como a NPU a recebe, pixel a pixel. A borda acende a cada inferência |
| **Predição e Confiança por classe** | o dígito previsto e as barras de cada classe, com o logit int8 recebido |
| **Telemetria** | backend, latência da última inferência, número de inferências, bytes enviados e recebidos pela UART, fps e tempo de processamento por quadro |
| **Log** | boot da FPGA, mudanças de palpite ao vivo, resultado final (com os 10 logits), pausas, gestos e erros |

Detalhes de implementação:
- **Vídeo sem cópia:** o quadro é mostrado sem cópia nem conversão de cor. O `QImage` aponta direto para o array do OpenCV, no formato `Format_BGR888`.
- **Traços vetoriais:** são linhas (`QPainterPath`) com pontas arredondadas e espessura de 14 px do quadro processado, escalada para o tamanho da janela.
- **Entrada da NPU:** a imagem 28×28 é desenhada ampliada sem suavização, com a grade por cima.
- **Barras:** são animadas até o valor novo (fator 0,6 a cada 33 ms). Chegam a menos de 1 ponto percentual do valor final em ~165 ms, antes da próxima inferência ao vivo.
- **Telemetria:** é atualizada a cada 250 ms, e não a cada quadro, para evitar recálculos de layout desnecessários.
- **Atalhos de teclado:** são `QShortcut`s da janela e funcionam com o foco em qualquer parte dela, inclusive no log e na tabela.

**Pausa:** a barra de espaço (ou o botão Pausar/Retomar) desliga a caneta. O rastreio continua, e o cursor aparece como "pausado". Um aviso no topo do vídeo e uma linha no log indicam a pausa. O desenho existente é mantido, e a inferência final dele acontece normalmente.

## 10. Desempenho

Medições feitas com vídeo sintético em tempo real (30 fps), com o MediaPipe rodando de verdade em cada quadro:

| Etapa | Antes (tudo na thread da janela) | Agora |
|---|---|---|
| Thread da interface, por quadro | ~24 ms | ~0,04 ms (+ ~3 ms para pintar o vídeo) |
| Thread de processamento, por quadro | — | ~20 ms (mediana); o MediaPipe responde por ~10 ms |
| Com inferência ao vivo e FPGA de 12 ms | — | ~20 ms (mediana), 33 ms (p95) |
| Quadros exibidos | — | 167 de 168 (148 de 149 com inferência ao vivo) |

Os ganhos vieram de três mudanças:
- **Threads:** o trabalho pesado saiu da thread da interface.
- **Composição:** a composição do traço em numpy foi trocada por linhas vetoriais do Qt.
- **Captura:** a fila de quadros atrasados foi eliminada.

Numa máquina mais lenta que a de teste, um quadro pode ser pulado de vez em quando, mas a imagem não fica atrasada em relação à mão.

## 11. Configuração

| Opção | Padrão | Efeito |
|---|---|---|
| `--port` | detecção automática | porta serial da FPGA |
| `--sim` | desligado | NPU emulada no computador, sem placa |
| `--baud` | 921600 | velocidade da UART |
| `--camera` | 0 | índice da câmera |
| `--input` | `mao` | `mao` (dedo) ou `cor` (bastão) |
| `--model` | `models/hand_landmarker.task` | modelo do MediaPipe |
| `--weights` | `weights.npz` | pesos int8 da NPU |
| `--firmware` | `firmware/cnn_server.bin` | firmware do servidor CNN |
| `--idle` | 0,8 s | tempo parado até a inferência final |
| `--live-interval` | 0,2 s | intervalo da inferência ao vivo (0 desliga) |
| `--fullscreen` | desligado | abre em tela cheia |
| `--scale` | — | escala da interface (ex.: 1,5 em telas de alta resolução) |

Arquivos de dados:
- `weights.npz`: pesos da NPU;
- `firmware/cnn_server.bin`: firmware enviado à FPGA no boot;
- `models/hand_landmarker.task`: modelo da mão;
- `calibration.json`: faixa de cor e modo do rastreio por cor. É criado ao calibrar e fica fora do git.

Os caminhos vêm de `paths.py`. Os arquivos distribuídos (pesos, firmware, modelo) são lidos da pasta do código ou, no executável do PyInstaller, da pasta do pacote (`_internal`). O `calibration.json` é gravado na pasta do código ou ao lado do `.exe`.

## 12. Mapa do código

| Arquivo | Principais elementos |
|---|---|
| `eureka.py` | `main()`: argumentos, escolha do backend, criação do rastreador da mão, abertura da janela, `--selftest` |
| `paths.py` | `resource_path` (arquivos distribuídos), `user_path` (arquivos gravados) |
| `pipeline.py` | `CaptureThread`, `ProcessingThread`, `Snapshot` |
| `engine.py` | `Engine` (processamento de quadros, caneta, gestos, inferência), `softmax_pct`, `find_fpga_port` |
| `hand.py` | `HandTracker` (MediaPipe, gestos), `OneEuroFilter`, `bend_degrees` |
| `vision.py` | `MarkerTracker` (cor), `StrokeCanvas` (traços), `to_npu_input` (28×28) |
| `npu.py` | `FpgaNpu` (protocolo serial), `CpuNpu` (emulação), `pack_weights_dma`, `load_weights` |
| `gui.py` | `MainWindow`, `VideoView`, `NpuInputView`, `ConfidenceBars`, `CalibrationPanel`, `BackendThread` |
| `export_weights.py` | `calibrate`: gera `weights.npz` a partir de `models/cnn_pretrained.pth` |

## 13. Limitações conhecidas

- **Traços parados:** num dígito de dois traços, se a mão ficar parada mais de 0,8 s entre um traço e outro, sai uma inferência final do traço incompleto antes da verdadeira. Aumentar `--idle` (ex.: 1,2) evita isso.
- **Gesto de desenhar:** só depende do indicador e do médio. Indicador + mínimo também desenha.
- **Detecção da mão:** depende de boa iluminação e da mão inteira na imagem. O ângulo 3D do MediaPipe é uma estimativa da rede, não uma medida de profundidade.
- **Porta COM no Windows:** é exclusiva. O Eureka e o GUI-TCC não podem usar a placa ao mesmo tempo.
- **MediaPipe:** precisa ser instalado com `--no-deps`, pelo conflito do `opencv-contrib-python` com o PyQt5 (veja o README).
- **Testes:** os testes automatizados usaram mãos e vídeos sintéticos e uma porta serial simulada. O comportamento com a câmera e a placa reais é verificado manualmente.
