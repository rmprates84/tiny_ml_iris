#ifndef TFLM_WRAPPER_H_
#define TFLM_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

// Inicializa o modelo (aloca tensores etc.)
// Retorna 0 em sucesso, <0 em erro.
int tflm_init_model(void);

// Executa uma inferência.
// in_features: 4 entradas do Iris (normalizadas da mesma forma que no Python)
// out_scores: 3 saídas (probabilidades ou scores para cada classe)
// Retorna 0 em sucesso, <0 em erro.
int tflm_infer(const float in_features[4], float out_scores[3]);

#ifdef __cplusplus
}
#endif

#endif  // TFLM_WRAPPER_H_
