# Sistema de clasificación automática de señales hostiles (prototipo)

Este repositorio contiene un prototipo funcional en **Streamlit + PyTorch** para:

1. Entrenar un clasificador por transferencia (ResNet18).
2. Cargar/gestionar una base de datos por carpetas (una carpeta por clase).
3. Guardar el modelo entrenado en `artifacts/model_checkpoint.pt` y recargarlo al reiniciar la app.
4. Realizar predicción con **top-3** y umbral de confianza.
5. Auto-archivar capturas clasificadas o mover a `NO_IDENTIFICADOS`.
6. Mostrar un panel tipo operativo similar al diseño solicitado.
7. Consultar una guia completa de uso en `DOCUMENTACION_ES.md`.
8. Probar la UI con escenarios concretos en `EXAMPLES.md`.

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

## Uso con tu dataset

Por defecto la UI usa `./Base_Datos` como carpeta del dataset.

Si prefieres, puedes cambiarla por la ruta de tu dataset real en `Carpeta base de datos`, idealmente en una carpeta externa al repo.

Despues pulsa `Reentrenar modelo`. El checkpoint se guardara en `artifacts/model_checkpoint.pt`.

Las predicciones archivadas por la UI se guardan aparte en `artifacts/archived_predictions/` para no contaminar automaticamente el dataset de entrenamiento.

## Como entrenar un modelo

1. Abre la app con `python -m streamlit run app.py`.
2. Revisa el campo `Carpeta base de datos real (usa una ruta externa al repo)`.
3. Deja `./Base_Datos` o escribe la ruta de tu dataset real.
4. Verifica que dentro de esa carpeta haya una subcarpeta por clase.
5. Si hace falta, usa `Anadir nuevo medio` para cargar imagenes nuevas.
6. Pulsa `Reentrenar modelo`.
7. Espera a que aparezca el mensaje de exito.
8. Comprueba que se haya creado `artifacts/model_checkpoint.pt`.

## Como probar el modelo despues de entrenarlo

1. Con la app abierta, sube una imagen en `Cargar espectrograma/imagen`.
2. Mira el bloque `Resultados de clasificacion`.
3. Revisa la clase principal, el porcentaje y el top-3.
4. Si `Auto-archivar` esta activo, revisa `artifacts/archived_predictions/`.
5. Cierra y vuelve a abrir la app para confirmar que el checkpoint se recarga automaticamente.
6. Vuelve a subir otra imagen para verificar que no necesitas reentrenar.

## Qué hace el entrenamiento

- Usa `torchvision.models.resnet18` preentrenada en ImageNet.
- Congela el backbone convolucional.
- Reemplaza la cabeza final por una capa lineal con `N` clases del dataset.
- Entrena solo la cabeza con `cross_entropy`.
- Reporta precisión de validación.
- Guarda un checkpoint reutilizable y lo recarga automaticamente al reiniciar Streamlit.

> Nota: este prototipo está preparado para imágenes (p.ej. espectrogramas). La captura real desde SDR (`rtl_tcp`, HDSDR, etc.) se deja como integración posterior por hardware.

## Documentacion ampliada

Consulta `DOCUMENTACION_ES.md` para ver el flujo completo, limitaciones actuales, mejoras recomendadas y pruebas sugeridas.

Consulta `EXAMPLES.md` para ver escenarios rapidos de prueba de la UI y validar si el flujo principal esta funcionando bien.
