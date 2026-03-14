# Ejemplos de prueba de la UI

Este archivo contiene escenarios practicos para probar la interfaz Streamlit y verificar si el flujo principal funciona bien o si hay fallos visibles.

## Antes de empezar

- Arranca la app con `python -m streamlit run app.py`.
- Prepara un dataset real fuera del repo.
- La estructura debe ser una carpeta por clase.
- La UI arranca por defecto con `./Base_Datos`.

Ejemplo:

```text
C:/RF-Datasets/SenalesHostiles/Base_Datos/
  KY-57 58/
  KYV-5/
  MIL-STD-188-181-C/
  P-3 P-8/
```

## Como entrenar un modelo desde la UI

1. Abre la app.
2. Revisa el campo `Carpeta base de datos real (usa una ruta externa al repo)`.
3. Deja `./Base_Datos` o escribe la ruta de tu dataset.
4. Confirma que tienes una carpeta por clase.
5. Si hace falta, usa `Anadir nuevo medio` para cargar imagenes nuevas.
6. Ajusta `Epocas`.
7. Pulsa `Reentrenar modelo`.
8. Espera el mensaje de exito.
9. Verifica que existe `artifacts/model_checkpoint.pt`.

## Como probar el modelo despues

1. Con el modelo ya entrenado, sube una imagen en `Cargar espectrograma/imagen`.
2. Revisa la clase principal y el top-3.
3. Si `Auto-archivar` esta activo, verifica `artifacts/archived_predictions/`.
4. Cierra y abre la app otra vez.
5. Comprueba que el checkpoint se carga solo.
6. Sube otra imagen y confirma que el modelo responde sin reentrenar.

## Ejemplo 1: verificar el aviso del path por defecto

### Pasos

1. Abre la app.
2. No cambies la ruta por defecto `./Base_Datos` del campo `Carpeta base de datos real (usa una ruta externa al repo)`.
3. Mira el bloque de ayuda bajo ese campo.

### Resultado esperado

- La UI debe advertir que esa ruta esta dentro del repo.
- Debe quedar claro que la ruta por defecto sirve para trabajo local, pero no es la ubicacion recomendada para un dataset operativo.

## Ejemplo 2: entrenamiento basico correcto

### Pasos

1. En `Carpeta base de datos real`, escribe la ruta completa de tu dataset.
2. Verifica que la UI muestre `Dataset actual:` con esa ruta.
3. Pulsa `Reentrenar modelo`.

### Resultado esperado

- La UI debe mostrar el spinner de entrenamiento.
- Debe aparecer mensaje de exito al terminar.
- Debe crearse `artifacts/model_checkpoint.pt`.
- En el panel derecho deben verse las clases detectadas y la precision de validacion.

## Ejemplo 3: comprobar persistencia del modelo

### Pasos

1. Entrena una vez con exito.
2. Cierra Streamlit.
3. Vuelve a ejecutar `python -m streamlit run app.py`.

### Resultado esperado

- La app debe cargar automaticamente el checkpoint.
- Debe verse un mensaje indicando que hay un modelo disponible desde disco.
- No deberias necesitar reentrenar para hacer una prediccion.

## Ejemplo 4: probar inferencia de una imagen

### Pasos

1. Con un modelo ya entrenado o cargado, sube una imagen en `Cargar espectrograma/imagen`.
2. Mira el panel `Resultados de clasificacion`.

### Resultado esperado

- Debe aparecer una clase principal con porcentaje.
- Debe mostrarse el top-3.
- Si `Auto-archivar` esta activado, debe guardarse una copia en `artifacts/archived_predictions/<clase>` o en `artifacts/archived_predictions/NO_IDENTIFICADOS`.

## Ejemplo 5: alta de una clase nueva desde la UI

### Pasos

1. Configura una ruta real de dataset.
2. En `Anadir nuevo medio hostil`, escribe un nombre de clase nuevo.
3. Sube varias imagenes de esa clase.
4. Pulsa `Anadir nuevo medio`.

### Resultado esperado

- La app debe crear la carpeta de la clase dentro del dataset.
- Debe guardar las imagenes subidas en esa carpeta.
- Debe aparecer un mensaje indicando que reentrenes el modelo.

## Ejemplo 6: detectar error por dataset inexistente

### Pasos

1. Escribe una ruta que no exista en `Carpeta base de datos real`.
2. Pulsa `Reentrenar modelo`.

### Resultado esperado

- La app debe fallar con un error claro de carpeta no encontrada.
- No debe bloquearse ni cerrarse.

## Ejemplo 7: detectar error por pocas clases

### Pasos

1. Usa un dataset que tenga solo una carpeta de clase.
2. Pulsa `Reentrenar modelo`.

### Resultado esperado

- La app debe avisar que se necesitan al menos 2 clases para entrenar.

## Ejemplo 8: detectar error por dataset demasiado pequeno

### Pasos

1. Usa un dataset con muy pocas imagenes en total.
2. Pulsa `Reentrenar modelo`.

### Resultado esperado

- La app debe avisar que el dataset es muy pequeno.
- No debe guardar un checkpoint invalido.

## Ejemplo 9: checkpoint ausente o borrado

### Pasos

1. Cierra la app.
2. Borra `artifacts/model_checkpoint.pt`.
3. Vuelve a abrir la app.

### Resultado esperado

- La UI debe indicar que no hay modelo cargado.
- Debe seguir siendo posible entrenar uno nuevo.

## Ejemplo 10: dataset dentro del repo

### Pasos

1. Escribe una ruta que apunte a una carpeta dentro de este repo.
2. Mira el bloque de ayuda bajo el campo de dataset.

### Resultado esperado

- La UI debe mostrar una advertencia indicando que se recomienda usar un dataset fuera del repo.
- El aviso debe dejar claro que no es la ubicacion recomendada.

## Checklist rapido

Si quieres una validacion corta, revisa esto:

- La app exige una ruta real de dataset.
- La app avisa si el dataset esta dentro del repo.
- El entrenamiento termina y guarda checkpoint.
- El modelo se recarga al reiniciar.
- La inferencia muestra top-3.
- El archivado guarda una copia fuera del dataset de entrenamiento.
- Los errores de ruta o dataset pequeno aparecen en pantalla sin romper la UI.
