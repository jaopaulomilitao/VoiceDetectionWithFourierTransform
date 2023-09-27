# Aluno : João Paulo de Abreu Militão
# Matrícula: 494959

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import distance

# Carregar os sinais de áudios para o treinamento
audiosMatData = scipy.io.loadmat('./InputDataTrain.mat')

# Pegar a matriz de dados dos sinais de áudio
audiosMatrix = audiosMatData['InputDataTrain']

audioNaoData = audiosMatrix[:, :5]
audioSimData = audiosMatrix[:, 5:]

# Definir valores do eixo X
x = np.arange(0, audiosMatrix.shape[0])

# Cores para os gráficos
cores = ['black', 'orange', 'blue', 'green', 'orangered']

# ________________________________Questão 01________________________________________
# Questão 01: Visualização dos Sinais de Áudio
# Nesta questão, os sinais de áudio "NÃO" e "SIM" são visualizados no domínio do tempo. 
# São criados dois subplots em uma única linha, um para "NÃO" e outro para "SIM", 
# e os sinais de áudio são plotados em cada subplot.

# Crie uma figura com duas subtramas em uma única linha
plt.figure(figsize=(12, 6))

# Primeira subtrama para 'NÃO'
plt.subplot(1, 2, 1)
for i in range(5):
    plt.plot(x, audioNaoData[:, i], label=f'audio0{i+1}', color=cores[i], linewidth=0.6)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinais do áudio "NÃO"')
plt.legend()

# Segunda subtrama para 'SIM'
plt.subplot(1, 2, 2)
for i in range(5):
    plt.plot(x, audioSimData[:, i], label=f'audio0{i+6}', color=cores[i], linewidth=0.6)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinais do áudio "SIM"')
plt.legend()

# Ajuste a disposição dos subplots
plt.tight_layout()

# Exibir a figura
plt.show()

# ________________________________Questão 02________________________________________
# Questão 02: Divisão dos Sinais de Áudio em Blocos
# Os sinais de áudio "NÃO" e "SIM" são divididos em 80 blocos de tamanho igual. 
# Em seguida, a energia de cada bloco é calculada para os 10 sinais de áudio.


# Dividir os sinais de áudio 'NÃO' e 'SIM' em 80 blocos de N/80 amostras
numeroDivisoes = 80
audioNaoDivided = np.array_split(audiosMatrix[:, :5], numeroDivisoes, axis=0)
audioSimDivided = np.array_split(audiosMatrix[:, 5:], numeroDivisoes, axis=0)

# Calcular a energia de cada bloco nos 10 sinais de áudio
audioNaoEnergies = np.array([np.sum(np.square(block), axis=0) for block in audioNaoDivided])
audioSimEnergies = np.array([np.sum(np.square(block), axis=0) for block in audioSimDivided])

# Criar uma figura com dois subplots (2 linhas, 1 coluna)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Definir valores do eixo X
x = np.arange(0, numeroDivisoes)

# Gráficos de energia dos áudios 'NÃO'
for i in range(5):
    axs[0].plot(x, audioNaoEnergies[:, i], label=f'audio0{i+1} (NÃO)', color=cores[i])

# Gráficos de energia dos áudios 'SIM'
for i in range(5, 10):
    axs[1].plot(x, audioSimEnergies[:, i-5], label=f'audio0{i+1} (SIM)', color=cores[i-5])

# Adicionar rótulos aos eixos e títulos aos subplots
axs[0].set_xlabel('Bloco')
axs[0].set_ylabel('Energia')
axs[0].set_title('Energia dos sinais de áudio "NÃO"')
axs[0].legend()

axs[1].set_xlabel('Bloco')
axs[1].set_ylabel('Energia')
axs[1].set_title('Energia dos sinais de áudio "SIM"')
axs[1].legend()

# Ajustar a posição dos subplots
plt.tight_layout()

# Exibir os gráficos
plt.show()
# ________________________________Questão 03________________________________________
# Questão 03: Transformada de Fourier dos Sinais de Áudio
# A transformada de Fourier é aplicada aos sinais de áudio "NÃO" e "SIM" 
# para obter informações sobre suas componentes de frequência. 
# Os gráficos mostram o módulo ao quadrado da transformada de Fourier no domínio de frequência.

# Calcular o módulo ao quadrado da transformada de Fourier dos áudios 'NÃO'
audioNaoFft = np.abs(np.fft.fftshift(np.fft.fft(audiosMatrix[:, :5], axis=0)))**2

# Calcular o módulo ao quadrado da transformada de Fourier dos áudios 'SIM'
audioSimFft = np.abs(np.fft.fftshift(np.fft.fft(audiosMatrix[:, 5:], axis=0)))**2

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, audiosMatrix.shape[0])

# Criar uma figura com subplots (2 linhas, 1 coluna)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Gráficos da transformada de Fourier dos áudios 'NÃO'
for i in range(5):
    axs[0].plot(x, audioNaoFft[:, i], label=f'audio0{i+1} (NÃO)', color=cores[i])

# Adicionar rótulos aos eixos e um título ao subplot 'NÃO'
axs[0].set_xlabel('Frequência')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Transformada de Fourier dos sinais de áudio "NÃO"')
axs[0].legend()

# Gráficos da transformada de Fourier dos áudios 'SIM'
for i in range(5, 10):
    axs[1].plot(x, audioSimFft[:, i-5], label=f'audio0{i+1} (SIM)', color=cores[i-5])

# Adicionar rótulos aos eixos e um título ao subplot 'SIM'
axs[1].set_xlabel('Frequência')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Transformada de Fourier dos sinais de áudio "SIM"')
axs[1].legend()

# Ajustar a posição dos subplots
plt.tight_layout()

# Exibir os gráficos
plt.show()


# ________________________________Questão 04________________________________________
# Questão 04: Filtragem das Baixas Frequências na Transformada de Fourier
# Os gráficos da transformada de Fourier são filtrados para manter apenas as frequências no intervalo de 0 a π/2. 
# Isso é feito para enfocar as frequências mais baixas, que podem conter informações importantes sobre os sinais.


# Definir os índices das frequências no intervalo de 0 a pi/2
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

# Filtrar os sinais FT para as baixas frequências (0 a pi/2)
audioNaoFftFiltered = audioNaoFft[x_filtered, :]
audioSimFftFiltered = audioSimFft[x_filtered, :]

# Criar uma figura com subplots (2 linhas, 1 coluna)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Gráficos da transformada de Fourier dos áudios 'NÃO' (Filtrada)
for i in range(5):
    axs[0].plot(x_filtered, audioNaoFftFiltered[:, i], label=f'audio0{i+1} (NÃO)', color=cores[i])

# Adicionar rótulos aos eixos e um título ao subplot 'NÃO'
axs[0].set_xlabel('Frequência')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Transformada de Fourier dos sinais de áudio "NÃO" (Filtrada)')
axs[0].legend()

# Gráficos da transformada de Fourier dos áudios 'SIM' (Filtrada)
for i in range(5, 10):
    axs[1].plot(x_filtered, audioSimFftFiltered[:, i-5], label=f'audio0{i+1} (SIM)', color=cores[i-5])

# Adicionar rótulos aos eixos e um título ao subplot 'SIM'
axs[1].set_xlabel('Frequência')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Transformada de Fourier dos sinais de áudio "SIM" (Filtrada)')
axs[1].legend()

# Ajustar a posição dos subplots
plt.tight_layout()

# Exibir os gráficos
plt.show()


# ________________________________Questão 05________________________________________
# Questão 05: Energia de Blocos dos Sinais de Áudio após Filtragem
# As energias dos blocos dos sinais de áudio são calculadas após a filtragem das baixas frequências. 
# Os gráficos mostram como a energia varia ao longo do tempo para os sinais "SIM" e "NÃO".


numeroDivisoes = 80

# Definir uma função para calcular as energias dos blocos de áudio
def calcular_energias(audio_fft_filtered, numeroDivisoes):
    audio_fft_divided = np.array_split(audio_fft_filtered, numeroDivisoes)
    audio_filtered_energies = [np.sum(block) for block in audio_fft_divided]
    return audio_filtered_energies

# Definir os índices das frequências no intervalo de 0 a pi/2 (como na Questão 04)
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

# Filtrar os sinais FT para as baixas frequências (0 a pi/2) - como na Questão 04
audioNaoFftFiltered = audioNaoFft[x_filtered, :]
audioSimFftFiltered = audioSimFft[x_filtered, :]

# Dividir os sinais de áudio em blocos e calcular as energias usando a função
numeroDivisoes = 80
audioFTNaoEnergies = []
audioFTSimEnergies = []

for i in range(5):
    audioFTNaoEnergies.append(calcular_energias(audioNaoFftFiltered[:, i], numeroDivisoes))
    audioFTSimEnergies.append(calcular_energias(audioSimFftFiltered[:, i], numeroDivisoes))

# Definir valores do eixo X
x = np.arange(0, numeroDivisoes)

# Criar uma figura para os subplots
plt.figure(figsize=(12, 6))  # Tamanho total da figura

# Subplot 1 - Energia da Transformada de Fourier dos áudios 'NÃO'
plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, primeiro subplot
for i in range(5):
    plt.plot(x, audioFTNaoEnergies[i], label=f'audio0{i+1}', color='C'+str(i))
plt.xlabel('Bloco')
plt.ylabel('Energia')
plt.title('Energia da Transformada de Fourier dos áudios "NÃO"')
plt.legend()

# Subplot 2 - Energia da Transformada de Fourier dos áudios 'SIM'
plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, segundo subplot
for i in range(5):
    plt.plot(x, audioFTSimEnergies[i], label=f'audio0{i+6}', color='C'+str(i))
plt.xlabel('Bloco')
plt.ylabel('Energia')
plt.title('Energia da Transformada de Fourier dos áudios "SIM"')
plt.legend()

# Ajustar o layout para evitar sobreposições
plt.tight_layout()

# Exibir o gráfico
plt.show()



# ________________________________Questão 06________________________________________
# Questão 06: Transformada de Fourier de Tempo Curto (STFT)
# Os sinais de áudio "NÃO" e "SIM" são divididos em blocos e, em seguida, a STFT é aplicada a esses blocos. 
# Os gráficos mostram como a amplitude da STFT varia ao longo do tempo.

# Separar os sinais de áudio 'NÃO' e 'SIM'
audiosNaoData = audiosMatrix[:, :5]  # Matriz para áudios 'NÃO'
audiosSimData = audiosMatrix[:, 5:]  # Matriz para áudios 'SIM'

# Dividir os sinais de áudio 'NÃO' e 'SIM' em 10 blocos de N/10 amostras
numeroDivisoes = 10
audiosNaoDivided = [np.array_split(audiosNaoData[:, i], numeroDivisoes) for i in range(5)]
audiosSimDivided = [np.array_split(audiosSimData[:, i], numeroDivisoes) for i in range(5)]


# Calcular o módulo ao quadrado da transformada de Fourier de cada bloco dos sinais de áudio
# Transformada de Fourier de tempo curto (short-time Fourier transform – STFT)
audio01_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosNaoDivided[0]))))
audio02_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosNaoDivided[1]))))
audio03_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosNaoDivided[2]))))
audio04_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosNaoDivided[3]))))
audio05_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosNaoDivided[4]))))
audio06_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosSimDivided[0]))))
audio07_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosSimDivided[1]))))
audio08_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosSimDivided[2]))))
audio09_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosSimDivided[3]))))
audio10_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audiosSimDivided[4]))))



x = np.linspace(-np.pi, np.pi, int(audiosMatrix.shape[0]/numeroDivisoes))
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]
#Definir os índices dos blocos da STFT
N_blocs = np.arange(audio01_STFT.shape[0])

# Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) 
audio01_STFT_filtrado = audio01_STFT[N_blocs[:, np.newaxis], x_filtered]
audio02_STFT_filtrado = audio02_STFT[N_blocs[:, np.newaxis], x_filtered]
audio03_STFT_filtrado = audio03_STFT[N_blocs[:, np.newaxis], x_filtered]
audio04_STFT_filtrado = audio04_STFT[N_blocs[:, np.newaxis], x_filtered]
audio05_STFT_filtrado = audio05_STFT[N_blocs[:, np.newaxis], x_filtered]
audio06_STFT_filtrado = audio06_STFT[N_blocs[:, np.newaxis], x_filtered]
audio07_STFT_filtrado = audio07_STFT[N_blocs[:, np.newaxis], x_filtered]
audio08_STFT_filtrado = audio08_STFT[N_blocs[:, np.newaxis], x_filtered]
audio09_STFT_filtrado = audio09_STFT[N_blocs[:, np.newaxis], x_filtered]
audio10_STFT_filtrado = audio10_STFT[N_blocs[:, np.newaxis], x_filtered]

import random  # Importe a biblioteca random para embaralhar as cores

# Definir cores para as linhas dos gráficos de STFT
lineColors = ['brown', 'green', 'orange', 'purple', 'pink', 'cyan', 'red', 'gray', 'blue', 'olive']

# Sinônimos para os títulos dos gráficos
titles = ['STFT do sinal de áudio "NÃO"', 'STFT do sinal de áudio "SIM"']

# Embaralhar as cores para que elas sejam diferentes em cada gráfico
random.shuffle(lineColors)

# Criar uma figura com subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plotar fft dos sinais de áudio 'NÃO' nos subplots
for i in range(numeroDivisoes):
    color = lineColors[i % len(lineColors)]
    axs[0].plot(x_filtered, audio01_STFT_filtrado[i], label=f'bloco {i+1}', color=color)

axs[0].set_xlabel('Amostra')
axs[0].set_ylabel('Amplitude')
axs[0].set_title(titles[0])  # Usar o novo título aqui
axs[0].legend()

# Embaralhar novamente as cores para o segundo gráfico
random.shuffle(lineColors)

for i in range(numeroDivisoes):
    color = lineColors[i % len(lineColors)]
    axs[1].plot(x_filtered, audio06_STFT_filtrado[i], label=f'bloco {i+1}', color=color)

axs[1].set_xlabel('Amostra')
axs[1].set_ylabel('Amplitude')
axs[1].set_title(titles[1])  # Usar o novo título aqui
axs[1].legend()

plt.tight_layout()

plt.show()

# ________________________________Questão 07________________________________________
# # Questão 07: Divisão dos Sinais de Áudio em Blocos de Tamanho Variável
# # Os sinais de áudio "NÃO" e "SIM" são divididos em 8 blocos de tamanho N/320, 
# # onde N é o número de amostras nos sinais de áudio.
# # Dividir os sinais de áudio em 8 blocos de tamanho N/320

# Dividir as STFT dos sinais de áudio 'SIM' e 'NÃO' em 8 blocos de N/320 amostras
numeroDivisoes = 8

# Instânciar vetores para armazenar de cada bloco da STFT dividido por 8  
STFT01Bloco = []
STFT02Bloco = []
STFT03Bloco = []
STFT04Bloco = []
STFT05Bloco = []
STFT06Bloco = []
STFT07Bloco = []
STFT08Bloco = []
STFT09Bloco = []
STFT10Bloco = []

# Armazenar cada bloco da STFT dividido por 8 (10x8x730)
for i in range(10):
    STFT01Bloco.append(np.array_split(audio01_STFT[i], numeroDivisoes))
    STFT02Bloco.append(np.array_split(audio02_STFT[i], numeroDivisoes))
    STFT03Bloco.append(np.array_split(audio03_STFT[i], numeroDivisoes))
    STFT04Bloco.append(np.array_split(audio04_STFT[i], numeroDivisoes))
    STFT05Bloco.append(np.array_split(audio05_STFT[i], numeroDivisoes))
    STFT06Bloco.append(np.array_split(audio06_STFT[i], numeroDivisoes))
    STFT07Bloco.append(np.array_split(audio07_STFT[i], numeroDivisoes))
    STFT08Bloco.append(np.array_split(audio08_STFT[i], numeroDivisoes))
    STFT09Bloco.append(np.array_split(audio09_STFT[i], numeroDivisoes))
    STFT10Bloco.append(np.array_split(audio10_STFT[i], numeroDivisoes))


# Instânciar vetores para armazenar as energias de cada bloco (N/320 amostras)
# Energias: 8 energias para cada uma das 10 STFTs
STFT01EnergiaDoBloco = []
STFT02EnergiaDoBloco = []
STFT03EnergiaDoBloco = []
STFT04EnergiaDoBloco = []
STFT05EnergiaDoBloco = []
STFT06EnergiaDoBloco = []
STFT07EnergiaDoBloco = []
STFT08EnergiaDoBloco = []
STFT09EnergiaDoBloco = []
STFT10EnergiaDoBloco = []

# Calcular as 80 energias: 8 energias para cada uma das 10 partes dos STFT
for i in range(10):
    for j in range(8):
        STFT01EnergiaDoBloco.append(np.sum(STFT01Bloco[i][j]))
        STFT02EnergiaDoBloco.append(np.sum(STFT02Bloco[i][j]))
        STFT03EnergiaDoBloco.append(np.sum(STFT03Bloco[i][j]))
        STFT04EnergiaDoBloco.append(np.sum(STFT04Bloco[i][j]))
        STFT05EnergiaDoBloco.append(np.sum(STFT05Bloco[i][j]))
        STFT06EnergiaDoBloco.append(np.sum(STFT06Bloco[i][j]))
        STFT07EnergiaDoBloco.append(np.sum(STFT07Bloco[i][j]))
        STFT08EnergiaDoBloco.append(np.sum(STFT08Bloco[i][j]))
        STFT09EnergiaDoBloco.append(np.sum(STFT09Bloco[i][j]))
        STFT10EnergiaDoBloco.append(np.sum(STFT10Bloco[i][j]))
  
    
    
# ________________________________Questão 08________________________________________
# # Questão 08: Cálculo das Médias das Energias nos Domínios de Tempo, TF e STFT
# # As médias das energias nos domínios de tempo, frequência (TF) e STFT são calculadas para os sinais "SIM" e "NÃO". 
# # Isso permitirá a comparação entre os domínios e ajudará na classificação.
    
# Calcular média das energias do áudio "NAO" E "SIM" para cada dominio
mediaEnergiaNoTempo_NAO = np.mean(np.array([audioNaoEnergies[:, 0], audioNaoEnergies[:, 1], audioNaoEnergies[:, 2], audioNaoEnergies[:, 3], audioNaoEnergies[:, 4]]), axis=0)
mediaEnergiaNoTempo_SIM = np.mean(np.array([audioSimEnergies[:, 0], audioSimEnergies[:, 1], audioSimEnergies[:, 2], audioSimEnergies[:, 3], audioSimEnergies[:, 4]]), axis=0)

mediaEnergiaNaFrequencia_NAO = np.mean(np.array([audioFTNaoEnergies[0], audioFTNaoEnergies[1], audioFTNaoEnergies[2], audioFTNaoEnergies[3], audioFTNaoEnergies[4]]), axis=0)
mediaEnergiaNaFrequencia_SIM = np.mean(np.array([audioFTSimEnergies[0], audioFTSimEnergies[1], audioFTSimEnergies[2], audioFTSimEnergies[3], audioFTSimEnergies[4]]), axis=0)

mediaEnergiaNaSTFT_NAO = np.mean(np.array([STFT01EnergiaDoBloco, STFT02EnergiaDoBloco, STFT03EnergiaDoBloco, STFT04EnergiaDoBloco, STFT05EnergiaDoBloco]), axis=0)
mediaEnergiaNaSTFT_SIM = np.mean(np.array([STFT06EnergiaDoBloco, STFT07EnergiaDoBloco, STFT08EnergiaDoBloco, STFT09EnergiaDoBloco, STFT10EnergiaDoBloco]), axis=0)


# ________________________________Questão 09________________________________________
# # Questão 09: Energia dos Sinais de Áudio de Teste
# # Os sinais de áudio de teste são divididos em blocos e a energia de cada bloco é calculada nos domínios de tempo, 
# # frequência (TF) e STFT.

# Carregar os sinais de áudios para o teste
audiosMatDataTest = scipy.io.loadmat('./InputDataTest.mat')

# Pegar a matriz de dados dos sinais de áudio
audiosTestMatrix = audiosMatDataTest['InputDataTest']

# Separar os sinais de áudio 'NÃO'
audio01DataTest = audiosTestMatrix[:, 0]
audio02DataTest = audiosTestMatrix[:, 1]
audio03DataTest = audiosTestMatrix[:, 2]

# Separar os sinais de áudio 'SIM'
audio04DataTest = audiosTestMatrix[:, 3]
audio05DataTest = audiosTestMatrix[:, 4]
audio06DataTest = audiosTestMatrix[:, 5]
audio07DataTest = audiosTestMatrix[:, 6]

#Cálculo de energias para o domínio do tempo :

numeroDivisoes = 80
audio01TestDivided = np.array_split(audio01DataTest, numeroDivisoes)
audio02TestDivided = np.array_split(audio02DataTest, numeroDivisoes)
audio03TestDivided = np.array_split(audio03DataTest, numeroDivisoes)
audio04TestDivided = np.array_split(audio04DataTest, numeroDivisoes)
audio05TestDivided = np.array_split(audio05DataTest, numeroDivisoes)
audio06TestDivided = np.array_split(audio06DataTest, numeroDivisoes)
audio07TestDivided = np.array_split(audio07DataTest, numeroDivisoes)

audio01TestEnergies = []
audio02TestEnergies = []
audio03TestEnergies = []
audio04TestEnergies = []
audio05TestEnergies = []
audio06TestEnergies = []
audio07TestEnergies = []

for i in range(numeroDivisoes):
    audio01TestEnergies.append(np.sum(np.square(audio01TestDivided[i])))
    audio02TestEnergies.append(np.sum(np.square(audio02TestDivided[i])))
    audio03TestEnergies.append(np.sum(np.square(audio03TestDivided[i])))
    audio04TestEnergies.append(np.sum(np.square(audio04TestDivided[i])))
    audio05TestEnergies.append(np.sum(np.square(audio05TestDivided[i])))
    audio06TestEnergies.append(np.sum(np.square(audio06TestDivided[i])))
    audio07TestEnergies.append(np.sum(np.square(audio07TestDivided[i])))



#Cálculo de energias para o domínio de TF :
     
# Calcular o módulo ao quadrado da transformada de Fourier de cada sinal de teste
audio01fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01DataTest))))
audio02fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02DataTest))))
audio03fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03DataTest))))
audio04fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04DataTest))))
audio05fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05DataTest))))
audio06fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06DataTest))))
audio07fft = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07DataTest))))

# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, audiosTestMatrix.shape[0])
 
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]

x_freqCutStart = x_filtered[0]
x_freqCutEnd = x_filtered[len(x_filtered)-1] + 1

# Filtrando os sinais FT para as baixas frequências (0 a pi/2) 
audio01Testfft_filtered = audio01fft[x_freqCutStart:x_freqCutEnd]
audio02Testfft_filtered = audio02fft[x_freqCutStart:x_freqCutEnd]
audio03Testfft_filtered = audio03fft[x_freqCutStart:x_freqCutEnd]
audio04Testfft_filtered = audio04fft[x_freqCutStart:x_freqCutEnd]
audio05Testfft_filtered = audio05fft[x_freqCutStart:x_freqCutEnd]
audio06Testfft_filtered = audio06fft[x_freqCutStart:x_freqCutEnd]
audio07Testfft_filtered = audio07fft[x_freqCutStart:x_freqCutEnd]

# Dividir os sinais da TF dos áudios de teste 'SIM' e 'NÃO' em 80 blocos
numeroDivisoes = 80
audio01TestfftDivided = np.array_split(audio01Testfft_filtered, numeroDivisoes)
audio02TestfftDivided = np.array_split(audio02Testfft_filtered, numeroDivisoes)
audio03TestfftDivided = np.array_split(audio03Testfft_filtered, numeroDivisoes)
audio04TestfftDivided = np.array_split(audio04Testfft_filtered, numeroDivisoes)
audio05TestfftDivided = np.array_split(audio05Testfft_filtered, numeroDivisoes)
audio06TestfftDivided = np.array_split(audio06Testfft_filtered, numeroDivisoes)
audio07TestfftDivided = np.array_split(audio07Testfft_filtered, numeroDivisoes)

# Instânciar vetores para armazenar as energias dos blocos de sinais
audio01Testfft_filteredEnergies = []
audio02Testfft_filteredEnergies = []
audio03Testfft_filteredEnergies = []
audio04Testfft_filteredEnergies = []
audio05Testfft_filteredEnergies = []
audio06Testfft_filteredEnergies = []
audio07Testfft_filteredEnergies = []


for i in range(numeroDivisoes):
    audio01Testfft_filteredEnergies.append(np.sum(audio01TestfftDivided[i]))
    audio02Testfft_filteredEnergies.append(np.sum(audio02TestfftDivided[i]))
    audio03Testfft_filteredEnergies.append(np.sum(audio03TestfftDivided[i]))
    audio04Testfft_filteredEnergies.append(np.sum(audio04TestfftDivided[i]))
    audio05Testfft_filteredEnergies.append(np.sum(audio05TestfftDivided[i]))
    audio06Testfft_filteredEnergies.append(np.sum(audio06TestfftDivided[i]))
    audio07Testfft_filteredEnergies.append(np.sum(audio07TestfftDivided[i]))
    
    
#Cálculo de energias para o domínio de STFT:
    
# Dividir os sinais de teste 'SIM' e 'NÃO' em 10 blocos de N/10 amostras
numeroDivisoes = 10
audio01TestDivided = np.array_split(audio01DataTest, numeroDivisoes)
audio02TestDivided = np.array_split(audio02DataTest, numeroDivisoes)
audio03TestDivided = np.array_split(audio03DataTest, numeroDivisoes)
audio04TestDivided = np.array_split(audio04DataTest, numeroDivisoes)
audio05TestDivided = np.array_split(audio05DataTest, numeroDivisoes)
audio06TestDivided = np.array_split(audio06DataTest, numeroDivisoes)
audio07TestDivided = np.array_split(audio07DataTest, numeroDivisoes)


# Calcular o módulo ao quadrado da transformada de Fourier de cada bloco dos sinais de teste
# Transformada de Fourier de tempo curto (short-time Fourier transform – STFT)
audio01Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio01TestDivided))))
audio02Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio02TestDivided))))
audio03Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio03TestDivided))))
audio04Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio04TestDivided))))
audio05Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio05TestDivided))))
audio06Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio06TestDivided))))
audio07Test_STFT = np.square(np.abs(np.fft.fftshift(np.fft.fft(audio07TestDivided))))


# Definir valores do eixo X
x = np.linspace(-np.pi, np.pi, int(audiosTestMatrix.shape[0]/numeroDivisoes))
x_filtered = np.where((x >= 0) & (x <= np.pi/2))[0]
N_blocs = np.arange(audio01Test_STFT.shape[0])

# Filtrando os sinais da STFT para as baixas frequências (0 a pi/2 ) 
audio01_STFT_filtrado = audio01Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio02_STFT_filtrado = audio02Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio03_STFT_filtrado = audio03Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio04_STFT_filtrado = audio04Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio05_STFT_filtrado = audio05Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio06_STFT_filtrado = audio06Test_STFT[N_blocs[:, np.newaxis], x_filtered]
audio07_STFT_filtrado = audio07Test_STFT[N_blocs[:, np.newaxis], x_filtered]

# Dividir as STFT dos sinais de teste 'SIM' e 'NÃO' em 8 blocos de N/320 amostras
numeroDivisoes = 8

STFT01TestBloco = []
STFT02TestBloco = []
STFT03TestBloco = []
STFT04TestBloco = []
STFT05TestBloco = []
STFT06TestBloco = []
STFT07TestBloco = []

for i in range(10):
    STFT01TestBloco.append(np.array_split(audio01_STFT_filtrado, numeroDivisoes))
    STFT02TestBloco.append(np.array_split(audio02_STFT_filtrado, numeroDivisoes))
    STFT03TestBloco.append(np.array_split(audio03_STFT_filtrado, numeroDivisoes))
    STFT04TestBloco.append(np.array_split(audio04_STFT_filtrado, numeroDivisoes))
    STFT05TestBloco.append(np.array_split(audio05_STFT_filtrado, numeroDivisoes))
    STFT06TestBloco.append(np.array_split(audio06_STFT_filtrado, numeroDivisoes))
    STFT07TestBloco.append(np.array_split(audio07_STFT_filtrado, numeroDivisoes))


STFT01TestEnergiaDoBloco = []
STFT02TestEnergiaDoBloco = []
STFT03TestEnergiaDoBloco = []
STFT04TestEnergiaDoBloco = []
STFT05TestEnergiaDoBloco = []
STFT06TestEnergiaDoBloco = []
STFT07TestEnergiaDoBloco = []

for i in range(10):
    for j in range(8):
        STFT01TestEnergiaDoBloco.append(np.sum(STFT01TestBloco[i][j]))
        STFT02TestEnergiaDoBloco.append(np.sum(STFT02TestBloco[i][j]))
        STFT03TestEnergiaDoBloco.append(np.sum(STFT03TestBloco[i][j]))
        STFT04TestEnergiaDoBloco.append(np.sum(STFT04TestBloco[i][j]))
        STFT05TestEnergiaDoBloco.append(np.sum(STFT05TestBloco[i][j]))
        STFT06TestEnergiaDoBloco.append(np.sum(STFT06TestBloco[i][j]))
        STFT07TestEnergiaDoBloco.append(np.sum(STFT07TestBloco[i][j]))

# ________________________________Questão 10________________________________________
# # Questão 10: Classificação dos Sinais de Áudio de Teste
# # Nesta questão, as distâncias euclidianas entre as energias calculadas para os sinais de áudio de teste 
# # e as médias das energias calculadas para os sinais "SIM" e "NÃO" são calculadas nos domínios de tempo, 
# # frequência (TF) e STFT. Com base nessas distâncias, os sinais de áudio de teste são classificados como "SIM" 
# # ou "NÃO", dependendo de quais médias de energia estão mais próximas.


euclideanDist_audio01TesteNoTempoNAO = distance.euclidean(audio01TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio02TesteNoTempoNAO = distance.euclidean(audio02TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio03TesteNoTempoNAO = distance.euclidean(audio03TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio04TesteNoTempoNAO = distance.euclidean(audio04TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio05TesteNoTempoNAO = distance.euclidean(audio05TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio06TesteNoTempoNAO = distance.euclidean(audio06TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_audio07TesteNoTempoNAO = distance.euclidean(audio07TestEnergies,mediaEnergiaNoTempo_NAO)
euclideanDist_Time_untilNO = [euclideanDist_audio01TesteNoTempoNAO,
                              euclideanDist_audio02TesteNoTempoNAO,
                              euclideanDist_audio03TesteNoTempoNAO,
                              euclideanDist_audio04TesteNoTempoNAO,
                              euclideanDist_audio05TesteNoTempoNAO,
                              euclideanDist_audio06TesteNoTempoNAO,
                              euclideanDist_audio07TesteNoTempoNAO]


euclideanDist_audio01TesteNoTempoSIM = distance.euclidean(audio01TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio02TesteNoTempoSIM = distance.euclidean(audio02TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio03TesteNoTempoSIM = distance.euclidean(audio03TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio04TesteNoTempoSIM = distance.euclidean(audio04TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio05TesteNoTempoSIM = distance.euclidean(audio05TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio06TesteNoTempoSIM = distance.euclidean(audio06TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_audio07TesteNoTempoSIM = distance.euclidean(audio07TestEnergies,mediaEnergiaNoTempo_SIM)
euclideanDist_Time_untilYes = [euclideanDist_audio01TesteNoTempoSIM,
                               euclideanDist_audio02TesteNoTempoSIM,
                               euclideanDist_audio03TesteNoTempoSIM,
                               euclideanDist_audio04TesteNoTempoSIM,
                               euclideanDist_audio05TesteNoTempoSIM,
                               euclideanDist_audio06TesteNoTempoSIM,
                               euclideanDist_audio07TesteNoTempoSIM]


euclideanDist_audio01TesteNaFrequenciaNAO = distance.euclidean(audio01Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio02TesteNaFrequenciaNAO = distance.euclidean(audio02Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio03TesteNaFrequenciaNAO = distance.euclidean(audio03Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio04TesteNaFrequenciaNAO = distance.euclidean(audio04Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio05TesteNaFrequenciaNAO = distance.euclidean(audio05Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio06TesteNaFrequenciaNAO = distance.euclidean(audio06Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_audio07TesteNaFrequenciaNAO = distance.euclidean(audio07Testfft_filteredEnergies,mediaEnergiaNaFrequencia_NAO)
euclideanDist_FT_untilNO = [euclideanDist_audio01TesteNaFrequenciaNAO,
                            euclideanDist_audio02TesteNaFrequenciaNAO,
                            euclideanDist_audio03TesteNaFrequenciaNAO,
                            euclideanDist_audio04TesteNaFrequenciaNAO,
                            euclideanDist_audio05TesteNaFrequenciaNAO,
                            euclideanDist_audio06TesteNaFrequenciaNAO,
                            euclideanDist_audio07TesteNaFrequenciaNAO]


euclideanDist_audio01TesteNaFrequenciaSIM = distance.euclidean(audio01Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio02TesteNaFrequenciaSIM = distance.euclidean(audio02Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio03TesteNaFrequenciaSIM = distance.euclidean(audio03Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio04TesteNaFrequenciaSIM = distance.euclidean(audio04Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio05TesteNaFrequenciaSIM = distance.euclidean(audio05Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio06TesteNaFrequenciaSIM = distance.euclidean(audio06Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_audio07TesteNaFrequenciaSIM = distance.euclidean(audio07Testfft_filteredEnergies,mediaEnergiaNaFrequencia_SIM)
euclideanDist_FT_untilYes = [euclideanDist_audio01TesteNaFrequenciaSIM,
                             euclideanDist_audio02TesteNaFrequenciaSIM,
                             euclideanDist_audio03TesteNaFrequenciaSIM,
                             euclideanDist_audio04TesteNaFrequenciaSIM,
                             euclideanDist_audio05TesteNaFrequenciaSIM,
                             euclideanDist_audio06TesteNaFrequenciaSIM,
                             euclideanDist_audio07TesteNaFrequenciaSIM]


euclideanDist_audio01TesteNaSTFTNAO = distance.euclidean(STFT01TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio02TesteNaSTFTNAO = distance.euclidean(STFT02TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio03TesteNaSTFTNAO = distance.euclidean(STFT03TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio04TesteNaSTFTNAO = distance.euclidean(STFT04TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio05TesteNaSTFTNAO = distance.euclidean(STFT05TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio06TesteNaSTFTNAO = distance.euclidean(STFT06TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_audio07TesteNaSTFTNAO = distance.euclidean(STFT07TestEnergiaDoBloco,mediaEnergiaNaSTFT_NAO)
euclideanDist_STFT_untilNO = [euclideanDist_audio01TesteNaSTFTNAO,
                               euclideanDist_audio02TesteNaSTFTNAO,
                               euclideanDist_audio03TesteNaSTFTNAO,
                               euclideanDist_audio04TesteNaSTFTNAO,
                               euclideanDist_audio05TesteNaSTFTNAO,
                               euclideanDist_audio06TesteNaSTFTNAO,
                               euclideanDist_audio07TesteNaSTFTNAO]



euclideanDist_audio01TesteNaSTFTSIM = distance.euclidean(STFT01TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio02TesteNaSTFTSIM = distance.euclidean(STFT02TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio03TesteNaSTFTSIM = distance.euclidean(STFT03TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio04TesteNaSTFTSIM = distance.euclidean(STFT04TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio05TesteNaSTFTSIM = distance.euclidean(STFT05TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio06TesteNaSTFTSIM = distance.euclidean(STFT06TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_audio07TesteNaSTFTSIM = distance.euclidean(STFT07TestEnergiaDoBloco,mediaEnergiaNaSTFT_SIM)
euclideanDist_STFT_untilYes = [euclideanDist_audio01TesteNaSTFTSIM,
                              euclideanDist_audio02TesteNaSTFTSIM,
                              euclideanDist_audio03TesteNaSTFTSIM,
                              euclideanDist_audio04TesteNaSTFTSIM,
                              euclideanDist_audio05TesteNaSTFTSIM,
                              euclideanDist_audio06TesteNaSTFTSIM,
                              euclideanDist_audio07TesteNaSTFTSIM]



AcertosTempo = 0
AcertosFrequencia = 0
AcertosSTFT = 0

for i in range(3):
    if euclideanDist_Time_untilNO[i] < euclideanDist_Time_untilYes[i]:
        AcertosTempo += 1
    if euclideanDist_FT_untilNO[i] < euclideanDist_FT_untilYes[i]:
        AcertosFrequencia += 1
    if euclideanDist_STFT_untilNO[i] < euclideanDist_STFT_untilYes[i]:
        AcertosSTFT += 1

for i in range(3, 7):
    if euclideanDist_Time_untilNO[i] > euclideanDist_Time_untilYes[i]:
        AcertosTempo += 1
    if euclideanDist_FT_untilNO[i] > euclideanDist_FT_untilYes[i]:
        AcertosFrequencia += 1
    if euclideanDist_STFT_untilNO[i] > euclideanDist_STFT_untilYes[i]:
        AcertosSTFT += 1


print(
    f'''
        Domínio do Tempo teve {AcertosTempo}/7, domínio da frequencia (FT) teve {AcertosFrequencia}/7 e domínio de STFT:  {AcertosSTFT}/7

        Ao analisar as taxas de precisão em diferentes áreas, nota-se que o domínio da Transformada de Fourier (FT) alcançou a maior precisão. Em contrapartida, o domínio de Tempo registrou a menor precisão.O domínio temporal se revelou menos preciso, pois não representa claramente o espectro de frequências presentes no áudio. Em outras palavras, para identificar os padrões que caracterizam as palavras "SIM" e "NÃO" nos áudios, é necessário determinar em quais frequências as amostras dos sinais são predominantes, e o domínio de tempo é a escolha menos adequada para essa análise.

        O segundo domínio menos preciso foi o da Short Time Fourier Transform (STFT). Normalmente utilizado para sinais não estacionários, esse domínio permite analisar as variações espectrais ao longo do tempo, levando em consideração a evolução temporal das características do áudio. No entanto, para este experimento específico, não apresentou uma precisão satisfatória.Por outro lado, o domínio da Transformada de Fourier revelou a maior precisão, pois possibilita a análise precisa das características espectrais dos sinais, incluindo as frequências dominantes em cada classe de áudio. Ou seja, ele permite o reconhecimento dos padrões de frequência nos sinais de fala.'''
)
