#include <cstdio>
#include "pico/stdlib.h"

// -------------------------------------------------------------------
// TensorFlow Lite Micro (via pico-tflmicro)
// -------------------------------------------------------------------
// Biblioteca disponível em: git clone https://github.com/raspberrypi/pico-tflmicro.git
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelo convertido em array C (gerado via xxd -i no Colab)
#include "iris_mlp_float.h"

// API em C que será chamada pelo main.c
#include "tflm_wrapper.h"

// -------------------------------------------------------------------
// Objetos estáticos do TFLM
// -------------------------------------------------------------------
namespace {

// tamanho da arena de tensores (ajuste se der erro de memória)
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// logger de erros
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// modelo e intérprete
static const tflite::Model* model = nullptr;

// registrador de operações (número de ops que vamos registrar)
static tflite::MicroMutableOpResolver<4> resolver;

// intérprete e tensores de entrada/saída
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

}  // namespace

// -------------------------------------------------------------------
// Inicializa o modelo TFLM
// -------------------------------------------------------------------
int tflm_init_model(void) {
    // Aponta para o modelo dentro do array iris_mlp_float_tflite
    model = tflite::GetModel(iris_mlp_float_tflite);
    if (model == nullptr) {
        printf("Erro: modelo nulo.\n");
        return -1;
    }

    // Registrar apenas as operações usadas pelo MLP do Iris
    // (FullyConnected + Relu + Softmax + Reshape)
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();

    // Cria o intérprete estático usando a arena
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        nullptr,  // resource variables
        nullptr,  // profiler
        false     // use_recording_allocator
    );

    interpreter = &static_interpreter;

    // Aloca os tensores
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors falhou.\n");
        return -2;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        printf("Erro ao obter tensores de entrada/saida.\n");
        return -3;
    }

    printf("TFLM inicializado com sucesso.\n");
    printf("Dimensoes input: ");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        printf("%d ", input_tensor->dims->data[i]);
    }
    printf("\n");

    printf("Dimensoes output: ");
    for (int i = 0; i < output_tensor->dims->size; i++) {
        printf("%d ", output_tensor->dims->data[i]);
    }
    printf("\n");

    return 0;
}

// -------------------------------------------------------------------
// Executa uma inferência no modelo Iris
// -------------------------------------------------------------------
int tflm_infer(const float in_features[4], float out_scores[3]) {
    if (!interpreter || !input_tensor || !output_tensor) {
        return -1;
    }

    // Copia os 4 atributos de entrada para o tensor de input
    for (int i = 0; i < 4; i++) {
        input_tensor->data.f[i] = in_features[i];
    }

    // Executa o modelo
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke falhou.\n");
        return -2;
    }

    // Copia as 3 saídas (uma por classe)
    for (int i = 0; i < 3; i++) {
        out_scores[i] = output_tensor->data.f[i];
    }

    return 0;
}
