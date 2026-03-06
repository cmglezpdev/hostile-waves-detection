# Sistema de clasificación automática de señales hostiles (prototipo)

Este repositorio contiene un prototipo funcional en **Streamlit + PyTorch** para:

1. Entrenar un clasificador por transferencia (ResNet18).
2. Cargar/gestionar una base de datos por carpetas (una carpeta por clase).
3. Realizar predicción con **top-3** y umbral de confianza.
4. Auto-archivar capturas clasificadas o mover a `NO_IDENTIFICADOS`.
5. Mostrar un panel tipo operativo similar al diseño solicitado.

## Estructura esperada del dataset

```text
Base_Datos/
  ├── KY-57-58/
  ├── P-3-P-8/
  ├── KYV-5/
  ├── MIL-STD-188-181-C/
  └── NO_IDENTIFICADOS/
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
./install_deps.sh
```

> `install_deps.sh` instala primero PyTorch/TorchVision en versión **CPU** para evitar descargas CUDA muy pesadas.

## Ejecución

```bash
python -m streamlit run app.py
```

## Qué hace el entrenamiento

- Usa `torchvision.models.resnet18` preentrenada en ImageNet.
- Congela el backbone convolucional.
- Reemplaza la cabeza final por una capa lineal con `N` clases del dataset.
- Entrena solo la cabeza con `cross_entropy`.
- Reporta precisión de validación.

> Nota: este prototipo está preparado para imágenes (p.ej. espectrogramas). La captura real desde SDR (`rtl_tcp`, HDSDR, etc.) se deja como integración posterior por hardware.
