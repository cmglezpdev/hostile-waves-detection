# Documentacion del proyecto

## 1. Objetivo del proyecto

Este proyecto es un prototipo de clasificacion de imagenes de espectro o waterfall para identificar senales hostiles a partir de capturas visuales. La idea base es:

1. Organizar imagenes por clase.
2. Entrenar un modelo con esas imagenes.
3. Subir una nueva imagen desde la UI.
4. Obtener una prediccion con nivel de confianza.
5. Guardar el modelo entrenado para reutilizarlo sin volver a entrenar en cada inicio.

Hoy el sistema funciona como un **clasificador de imagenes**. Todavia no es una integracion completa con SDR en tiempo real.

## 2. Que hace hoy la aplicacion

La app actual permite:

- Cargar una imagen manualmente desde Streamlit.
- Administrar un dataset por carpetas, una carpeta por clase.
- Anadir imagenes nuevas desde la UI.
- Reentrenar el modelo con transfer learning usando ResNet18.
- Guardar el checkpoint del modelo en `artifacts/model_checkpoint.pt`.
- Recargar el checkpoint automaticamente al reiniciar la app.
- Mostrar top-3 de predicciones y un umbral de confianza.
- Auto-archivar la imagen clasificada en `artifacts/archived_predictions/<clase>` o en `artifacts/archived_predictions/NO_IDENTIFICADOS`.

## 3. Arquitectura actual

### `app.py`

Es la interfaz Streamlit. Desde aqui se hace todo el flujo operativo:

- carga de imagen para inferencia,
- alta de nuevas clases,
- entrenamiento,
- carga del modelo guardado,
- visualizacion del estado del checkpoint,
- archivado automatico,
- registro de eventos.

### `model_utils.py`

Contiene la logica de machine learning:

- transformaciones de imagen,
- carga del dataset con `ImageFolder`,
- construccion del modelo ResNet18,
- entrenamiento,
- validacion,
- guardado del checkpoint,
- carga del checkpoint,
- prediccion top-k.

### `README.md`

Explica el arranque rapido.

## 4. Como funciona el dataset

El dataset debe tener una carpeta por clase. Ejemplo:

```text
Base_Datos/
  KY-57 58/
  KYV-5/
  MIL-STD-188-181-C/
  P-3 P-8/
```

Cada carpeta contiene las imagenes de entrenamiento de esa clase. `torchvision.datasets.ImageFolder` usa el nombre de la carpeta como etiqueta.

### Importante

- Debe haber al menos 2 clases.
- Debe haber suficientes imagenes para que el entrenamiento tenga sentido.
- Si cambias las carpetas del dataset despues de entrenar, conviene volver a entrenar.

## 5. Flujo completo de trabajo

### Paso 1: Elegir carpeta del dataset

En la UI, el campo `Carpeta base de datos` define donde se guardan y se leen las imagenes del dataset.

Por defecto la app arranca con `./Base_Datos`.

Esa ruta puede mantenerse si quieres trabajar localmente dentro del proyecto, o puede apuntar a tu dataset real organizado por carpetas de clase, idealmente fuera de este repo.

### Paso 2: Anadir imagenes o clases

Puedes crear una clase nueva desde la UI:

1. Escribes el nombre de la clase.
2. Subes varias imagenes.
3. La app crea la carpeta correspondiente.
4. La app guarda las imagenes en esa carpeta.

### Paso 3: Entrenar

Cuando pulsas `Reentrenar modelo`:

1. Se carga el dataset con `ImageFolder`.
2. Se divide en entrenamiento y validacion.
3. Se construye una ResNet18.
4. Se congela el backbone preentrenado.
5. Se reemplaza la capa final para el numero de clases actual.
6. Se entrena la cabeza final.
7. Se calcula la precision de validacion.
8. Se guarda el checkpoint en `artifacts/model_checkpoint.pt`.

### Paso 4: Guardar y reutilizar el modelo

El checkpoint guarda:

- pesos del modelo (`state_dict`),
- clases,
- dataset usado,
- precision de validacion,
- fecha de entrenamiento,
- version de PyTorch y TorchVision,
- numero de epocas,
- arquitectura.

Al volver a abrir Streamlit, la app intenta cargar automaticamente ese checkpoint.

### Paso 5: Probar una imagen nueva

1. Subes una imagen en `Cargar espectrograma/imagen`.
2. La app ejecuta inferencia.
3. Muestra la mejor clase y el top-3.
4. Si `Auto-archivar` esta activo:
   - si la confianza supera el umbral, la imagen se guarda en `artifacts/archived_predictions/<clase>`;
   - si no supera el umbral, se guarda en `artifacts/archived_predictions/NO_IDENTIFICADOS`.
   - estas capturas archivadas no entran automaticamente al dataset de entrenamiento; primero deberian revisarse.

## 6. Como ejecutar el proyecto

### Requisitos

- Python 3.10 o superior recomendado.
- Dependencias de `requirements.txt`.
- PyTorch y TorchVision.

### Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

En Linux/macOS puedes seguir el flujo equivalente del `README.md`.

### Ejecucion

```bash
python -m streamlit run app.py
```

## 7. Como entrenar un modelo paso a paso

Esta es la secuencia completa dentro de la UI:

1. Arranca la app.
2. Ve al campo `Carpeta base de datos real (usa una ruta externa al repo)`.
3. Deja `./Base_Datos` o escribe la ruta donde esta tu dataset.
4. Comprueba que en esa carpeta tengas una subcarpeta por clase.
5. Si te faltan imagenes, usa `Anadir nuevo medio` para cargar mas muestras.
6. Ajusta `Epocas` segun la prueba que quieras hacer.
7. Pulsa `Reentrenar modelo`.
8. Espera el mensaje de exito.
9. Revisa el resumen de `Ultimo entrenamiento` y `Precision validacion`.
10. Comprueba que existe `artifacts/model_checkpoint.pt`.

### Que significa que el entrenamiento salio bien

Si todo fue correcto, deberias ver:

- mensaje de exito en pantalla,
- clases cargadas en la columna derecha,
- precision de validacion visible,
- checkpoint creado en `artifacts/model_checkpoint.pt`.

## 8. Como probar el modelo despues de entrenarlo

Cuando el entrenamiento termina, puedes probarlo asi:

1. Sube una imagen en `Cargar espectrograma/imagen`.
2. Mira la seccion `Resultados de clasificacion`.
3. Verifica la clase principal y el porcentaje de confianza.
4. Revisa tambien el top-3 para ver alternativas.
5. Si `Auto-archivar` esta activo, comprueba que aparezca una copia en `artifacts/archived_predictions/`.
6. Cierra la app y vuelvela a abrir.
7. Verifica que el modelo se cargue automaticamente desde disco.
8. Sube otra imagen para confirmar que el modelo sigue funcionando sin reentrenar.

### Que significa que la prueba salio bien

La prueba posterior al entrenamiento esta bien si:

- la app no pide reentrenar otra vez,
- el checkpoint se carga solo al iniciar,
- puedes subir imagenes y obtener top-3,
- las predicciones archivadas aparecen en `artifacts/archived_predictions/`.

## 9. Como probarlo

Si quieres una lista corta de escenarios listos para ejecutar desde la UI, consulta tambien `EXAMPLES.md`.

### Prueba rapida con tu dataset

1. Ejecuta Streamlit.
2. Cambia `Carpeta base de datos` a la ruta de tu dataset real.
3. Pulsa `Reentrenar modelo`.
4. Verifica que se cree `artifacts/model_checkpoint.pt`.
5. Sube una imagen de prueba o una nueva captura similar.
6. Revisa la prediccion top-3.
7. Cierra Streamlit y vuelve a abrirlo.
8. Comprueba que el modelo se carga automaticamente.

### Casos que conviene validar

#### Caso A: sin dataset

- Define una carpeta vacia.
- Pulsa entrenar.
- La app debe mostrar error claro.

#### Caso B: menos de 2 clases

- Deja una sola carpeta de clase.
- Pulsa entrenar.
- La app debe impedir el entrenamiento.

#### Caso C: checkpoint corrupto o ausente

- Borra o renombra `artifacts/model_checkpoint.pt`.
- Reinicia la app.
- La app debe seguir abriendo y avisar que no hay checkpoint disponible.

#### Caso D: persistencia

- Entrena una vez.
- Reinicia la app.
- Comprueba que no hace falta reentrenar para clasificar.

#### Caso E: dataset cambiado despues de entrenar

- Anade o quita carpetas de clases.
- La app debe mostrar una advertencia para reentrenar.

## 10. Como funciona el entrenamiento

El modelo actual usa transfer learning con ResNet18:

- entrada redimensionada a `224x224`,
- normalizacion tipo ImageNet,
- backbone congelado,
- cabeza final nueva,
- optimizador Adam,
- perdida `cross_entropy`,
- split de entrenamiento y validacion.

Esto es un buen punto de partida para un prototipo, especialmente si tus datos son capturas consistentes del waterfall.

## 11. Limitaciones actuales

Estas son las limitaciones mas importantes:

1. **No hay captura SDR real**. El panel SDR es solo una simulacion visual.
2. **Es un clasificador de imagenes, no de I/Q**. El modelo no procesa muestras RF crudas.
3. **No hay set de test independiente**. Solo hay validacion.
4. **No hay matriz de confusion ni metricas por clase**.
5. **Puede aprender ruido visual de la pantalla**. Si en la imagen aparece la UI de HDSDR, escalas, colores fijos o barras del sistema, el modelo puede memorizar eso.
6. **No hay recorte de ROI**. Se entrena con la captura completa.
7. **No hay aumento de datos**.
8. **No hay API backend separada**. Todo corre dentro de Streamlit.

## 12. Como mejorarlo

### Mejora 1: recortar solo la zona util del waterfall

Es una de las mejoras mas importantes. Si entrenas con la captura completa, el modelo puede usar como pista la barra superior, la escala o el color del programa en vez del patron RF.

### Mejora 2: separar `train`, `val` y `test`

Ahora solo hay entrenamiento y validacion. Para medir rendimiento real necesitas un conjunto de test independiente.

### Mejora 3: agregar metricas mejores

Conviene anadir:

- accuracy por clase,
- precision,
- recall,
- F1,
- matriz de confusion.

### Mejora 4: agregar aumentos de datos

Ejemplos utiles:

- pequenos cambios de brillo,
- ruido moderado,
- recortes controlados,
- desplazamientos leves.

No conviene hacer aumentos que destruyan la estructura temporal/frecuencial de la senal.

### Mejora 5: guardar tambien un resumen JSON

El checkpoint actual ya guarda metadata util. Como mejora futura, puedes generar ademas un JSON facil de leer sin cargar PyTorch.

### Mejora 6: integrar un backend real

Si mas adelante quieres separar UI y logica:

- Streamlit o frontend web para la interfaz,
- FastAPI o Flask para entrenamiento e inferencia,
- cola de trabajos para entrenamiento largo,
- almacenamiento de modelos versionados.

### Mejora 7: integrar captura SDR real

Para llegar a un sistema operativo real faltaria:

- captura desde hardware SDR,
- generacion automatica de waterfall o espectrograma,
- deteccion de actividad,
- segmentacion de ROI,
- clasificacion final.

### Mejora 8: versionado del modelo

Hoy solo se guarda un checkpoint principal. En el futuro puedes guardar varios modelos por fecha o por dataset.

## 13. Recomendaciones para el dataset

- Usa capturas con formato consistente.
- Mantiene el mismo rango de frecuencia, zoom y estilo visual siempre que sea posible.
- Evita mezclar imagenes con interfaces muy distintas si no forman parte del objetivo.
- Revisa que las etiquetas sean correctas.
- Intenta tener mas muestras por clase.
- Si una carpeta tiene muy pocas imagenes, el modelo se vuelve inestable.

## 14. Solucion de problemas

### "No existe la carpeta"

La ruta del dataset esta mal o la carpeta aun no fue creada.

### "Se necesitan al menos 2 clases"

Falta una segunda carpeta de clase.

### "Dataset muy pequeno"

Necesitas mas imagenes antes de entrenar.

### El modelo se entreno pero no se guardo

Revisa permisos de escritura y la carpeta `artifacts`.

### La app abre pero no carga el modelo

Puede faltar el checkpoint o estar corrupto. Reentrena y vuelve a probar.

## 15. Resumen final

El proyecto ya resuelve un flujo util para prototipado:

- dataset por carpetas,
- entrenamiento desde UI,
- persistencia del modelo,
- recarga automatica del checkpoint,
- prediccion top-3,
- archivado automatico.

El siguiente salto de calidad no esta tanto en cambiar de red, sino en mejorar el dato:

1. recortar mejor la senal,
2. medir con test real,
3. limpiar sesgos visuales,
4. integrar captura SDR verdadera.
