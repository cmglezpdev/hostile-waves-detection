from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from model_utils import predict_topk, train_transfer

st.set_page_config(page_title="Clasificación de Señales Hostiles", layout="wide")


def log_event(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {msg}")


for key, default in {
    "logs": [],
    "trained_model": None,
    "classes": [],
    "last_val_acc": None,
    "last_train_at": None,
}.items():
    st.session_state.setdefault(key, default)

st.title("Sistema de Clasificación Automática de Señales Hostiles - FAR")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Visualizador de espectro")
    image_file = st.file_uploader("Cargar espectrograma/imagen", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file, use_column_width=True)

with col2:
    st.subheader("Resultados de clasificación")
    confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    auto_archive = st.checkbox("Auto-archivar", value=True)

    if st.session_state.trained_model is not None and image_file is not None:
        temp = Path(".tmp_predict.jpg")
        temp.write_bytes(image_file.getvalue())
        top3 = predict_topk(st.session_state.trained_model, temp, st.session_state.classes, topk=3)
        best_label, best_prob = top3[0]
        st.markdown(f"### Medio detectado: **{best_label} ({best_prob * 100:.1f}%)**")
        st.progress(min(1.0, float(best_prob)))
        st.write("Top-3 candidatos:")
        for idx, (name, p) in enumerate(top3, start=1):
            st.write(f"{idx}. {name}: {p * 100:.1f}%")

        dataset_dir = Path(st.session_state.dataset_dir)
        target_class = best_label if best_prob >= confidence_threshold else "NO_IDENTIFICADOS"
        if auto_archive:
            target_dir = dataset_dir / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{target_class}_conf{int(best_prob * 100)}.jpg"
            dest = target_dir / filename
            dest.write_bytes(temp.read_bytes())
            log_event(f"Detección {best_label} ({best_prob * 100:.1f}%). Guardado en {target_class}.")

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("Control SDR (simulado)")
    st.number_input("Frecuencia (MHz)", value=250.0, step=0.1)
    st.number_input("Ganancia (dB)", value=30.0, step=1.0)
    st.number_input("Sample rate (MHz)", value=2.4, step=0.1)
    st.radio("Modo de acceso", ["Exclusivo", "Compartido (rtl_tcp)"], horizontal=True)

with right:
    st.subheader("Captura automática")
    st.checkbox("Activar monitoreo continuo cada 1 s", value=True)
    st.checkbox("Guardar solo cuando hay señal", value=True)
    st.number_input("Umbral de energía", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    dataset_dir = st.text_input("Carpeta base de datos", value="./Base_Datos")
    st.session_state.dataset_dir = dataset_dir

st.divider()

m1, m2 = st.columns([1, 1])
with m1:
    st.subheader("Gestión de base de datos y modelo")

    new_class = st.text_input("Añadir nuevo medio hostil (nombre de clase)")
    new_files = st.file_uploader("Imágenes del nuevo medio", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("➕ Añadir nuevo medio"):
        if new_class and new_files:
            class_dir = Path(st.session_state.dataset_dir) / new_class
            class_dir.mkdir(parents=True, exist_ok=True)
            for f in new_files:
                (class_dir / f.name).write_bytes(f.getvalue())
            log_event(f"Nuevo medio '{new_class}' añadido con {len(new_files)} imágenes.")
            st.success("Medio añadido correctamente.")
        else:
            st.warning("Define nombre de clase y sube imágenes.")

    epochs = st.number_input("Épocas", min_value=1, max_value=100, value=5, step=1)
    if st.button("🔄 Reentrenar modelo"):
        try:
            with st.spinner("Entrenando modelo por transferencia..."):
                result = train_transfer(Path(st.session_state.dataset_dir), epochs=int(epochs))
            st.session_state.trained_model = result.model
            st.session_state.classes = result.classes
            st.session_state.last_val_acc = result.val_accuracy
            st.session_state.last_train_at = datetime.now().strftime("%d/%m/%Y %H:%M")
            log_event(
                f"Reentrenamiento completado ({epochs} épocas). Precisión validación: {result.val_accuracy * 100:.1f}%"
            )
            st.success("Modelo entrenado correctamente.")
        except Exception as exc:
            log_event(f"Error de entrenamiento: {exc}")
            st.error(f"No se pudo entrenar el modelo: {exc}")

with m2:
    st.subheader("Clases disponibles")
    base = Path(st.session_state.dataset_dir)
    if base.exists():
        classes = [d for d in base.iterdir() if d.is_dir()]
        for c in sorted(classes):
            count = len([p for p in c.iterdir() if p.is_file()])
            st.write(f"• {c.name} ({count} imágenes)")
    else:
        st.info("Aún no existe la carpeta base.")

    if st.session_state.last_train_at:
        st.markdown(
            f"**Último entrenamiento:** {st.session_state.last_train_at}  \\n**Precisión validación:** {st.session_state.last_val_acc * 100:.1f}%"
        )

st.divider()
st.subheader("Registro de eventos")
for line in st.session_state.logs[:20]:
    st.code(line)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶️ INICIAR CAPTURA"):
        log_event("Iniciado monitoreo continuo cada 1 s.")
with c2:
    if st.button("⏹️ DETENER"):
        log_event("Captura detenida.")
with c3:
    if st.button("💾 GUARDAR CONFIGURACIÓN"):
        cfg = Path("config_runtime.txt")
        cfg.write_text(
            f"dataset_dir={st.session_state.dataset_dir}\n"
            f"confidence_threshold={confidence_threshold}\n"
            f"auto_archive={auto_archive}\n",
            encoding="utf-8",
        )
        log_event("Configuración guardada en config_runtime.txt")
        st.success("Configuración guardada")
