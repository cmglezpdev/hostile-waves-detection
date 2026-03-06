# Análisis del código original de entrenamiento (notebook)

## Qué hace exactamente

1. **Carga CIFAR-10** (no señales electromagnéticas reales) y lo divide en:
   - Entrenamiento: 45,000 imágenes.
   - Validación: 5,000 imágenes.
   - Test: 10,000 imágenes.
2. **Aplica preprocesamiento de ImageNet** (resize a 224x224 y normalización ImageNet) para poder usar ResNet18 preentrenada.
3. Selecciona **GPU si hay CUDA**; si no, CPU.
4. Define función `accuracy` para medir precisión de clasificación.
5. Carga `resnet18(pretrained=True)` y crea `model_aux` quitando la última capa (`fc`).
6. **Congela** los parámetros del backbone (`requires_grad=False`).
7. Construye un modelo final:
   - `model_aux`
   - `Flatten`
   - `Linear(512 -> 10)`
8. Entrena con **Adam** y `cross_entropy` por 5 épocas.
9. Evalúa precisión final en test.

## En términos de transferencia de aprendizaje

Sí, el código usa **transfer learning**:
- Aprovecha características ya aprendidas en ImageNet (bordes, texturas, patrones visuales).
- Reentrena solo la cabeza para CIFAR-10.

## Limitaciones para tu caso (ondas electromagnéticas)

- El dataset actual es **CIFAR-10**, no espectrogramas de señales hostiles.
- El pipeline no integra captura SDR real (`rtl_tcp`, I/Q, FFT/STFT).
- No hay UI operativa ni archivado automático por umbral.

## Cómo adaptarlo a tu objetivo

1. Sustituir CIFAR-10 por un dataset tipo `ImageFolder` con espectrogramas por clase.
2. Mantener la lógica de transferencia (ResNet18 congelada + capa final nueva).
3. Añadir módulo de inferencia en tiempo real y top-3.
4. Aplicar umbral de energía previo (filtro de ruido).
5. Implementar auto-archivo:
   - Si `conf >= umbral_conf`: guardar en carpeta de clase.
   - Si no: guardar en `NO_IDENTIFICADOS`.
6. Registrar eventos con timestamp.

Este repositorio ya incluye un primer prototipo de UI con esos bloques funcionales para acelerar la integración final.
