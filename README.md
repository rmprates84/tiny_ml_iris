# TinyML – Classificação do Dataset Iris no Raspberry Pi Pico W  
### Prática com Rede Neural Artificial (RNA) para Microcontroladores

Este projeto implementa uma **Rede Neural Artificial (RNA)**, Perceptron Multicamadas (MLP), embarcada no **Raspberry Pi Pico W**, utilizando a biblioteca **TensorFlow Lite Micro (TFLM)** para executar inferência diretamente no microcontrolador — abordagem típica de **TinyML**.

Este código faz parte de um projeto que demonstra como treinar, converter e executar um modelo real de Machine Learning em um dispositivo de recursos extremamente limitados. Como conteúdo complementar, o modelo foi treinado usando o google colab, o link do código está disponível em: https://colab.research.google.com/drive/1MnmXluBn_oCctJ-MPaiS2RxqsRbwg4Fk?usp=sharing 

---

## 📌 Objetivos

- Demonstrar o fluxo completo de TinyML:  
  **Criação do modelo → Treinamento → Conversão → Deploy → Inferência embarcada**
- Normalizar dados embarcados de forma idêntica ao treinamento.
- Executar inferências usando TFLM. Biblioteca disponível em: https://github.com/raspberrypi/pico-tflmicro.git
- Construir e imprimir a **matriz de confusão** 3×3.
- Calcular a acurácia final diretamente no microcontrolador.
- Integrar código C/C++ ao TensorFlow Lite Micro via wrapper.

---

## 🧠 Visão geral

A aplicação embarcada no Pico W:

1. Carrega um modelo **MLP (rede neural multicamadas)** treinado com o dataset Iris.
2. Aplica normalização padrão (média e desvio).
3. Executa inferência amostra por amostra (150).
4. Constrói a **matriz de confusão 3×3** (real × predito).
5. Calcula a acurácia final da rede.
6. Exibe tudo via USB/serial.

Essa prática permite que estudantes compreendam como modelos inteligentes podem ser executados em **microcontroladores**, base fundamental para aplicações TinyML e Edge AI.

---

## 📁 Organização dos arquivos

### `tiny_ml_02.c`
Aplicação principal em C.  
Responsável por:

- Inicializar o Pico W e o ambiente TFLM.  
- Normalizar cada amostra com `iris_means` e `iris_stds`.  
- Realizar inferências via `tflm_infer()`.  
- Construir a matriz de confusão.  
- Calcular a acurácia e imprimir os resultados.

---

### `tflm_wrapper.h` / `tflm_wrapper.cpp`
Wrapper em C/C++ para o TensorFlow Lite Micro.

- Configura a arena de tensores.  
- Carrega o modelo embarcado (`iris_mlp_float_tflite`).  
- Registra operações necessárias (Dense, ReLU, Softmax).  
- Expõe:
  - `tflm_init_model()`  
  - `tflm_infer(float input[4], float output[3])`

---

### `iris_mlp_float.h`
Modelo TFLite convertido para array C (`unsigned char[]`), contendo a rede neural MLP treinada previamente em Python.

---

### `iris_dataset.h`
Dataset Iris embarcado no firmware:

- `iris_features[150][4]`  
- `iris_labels[150]`

---

### `iris_normalization.h`
Estatísticas de normalização utilizadas:

- `iris_means[4]`  
- `iris_stds[4]`

Esses valores replicam exatamente o StandardScaler do treinamento, garantindo consistência na inferência.

---

### `CMakeLists.txt`
Arquivo de build usando pico-sdk + TFLM:

- Configuração do projeto
- Inclusão do TensorFlow Lite Micro
- Compilação dos arquivos `.c` e `.cpp`
- Links com bibliotecas padrão do Pico

---

## 🔧 Como compilar o projeto

### 1. Instale o Pico SDK
Disponível em:  
https://github.com/raspberrypi/pico-sdk

---

### 2. Configure e compile
```bash
mkdir build
cd build
cmake ..
make -j4
