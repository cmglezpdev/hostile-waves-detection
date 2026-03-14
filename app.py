from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

from model_utils import (
    checkpoint_exists,
    load_checkpoint,
    predict_topk_from_bytes,
    save_checkpoint,
    train_transfer,
)

DEFAULT_DATASET_DIR = "./Base_Datos"
DEFAULT_CHECKPOINT_PATH = Path("artifacts/model_checkpoint.pt")
ARCHIVE_ROOT = Path("artifacts/archived_predictions")
UTILITY_FOLDERS = {"NO_IDENTIFICADOS"}
WORKSPACE_ROOT = Path(__file__).resolve().parent
DATASET_PATH_EXAMPLES = [
    "C:/RF-Datasets/SenalesHostiles/Base_Datos",
    "D:/Operaciones/Waterfalls/Base_Datos",
]

st.set_page_config(page_title="Clasificacion de Senales Hostiles", layout="wide")


def log_event(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {msg}")


def normalize_class_name(raw_name: str) -> str:
    clean_name = raw_name.strip().replace("/", "-").replace("\\", "-")
    return " ".join(clean_name.split())


def count_images(class_dir: Path) -> int:
    return sum(1 for path in class_dir.rglob("*") if path.is_file())


def dataset_class_names(dataset_dir: Path, include_utility_folders: bool = True) -> list[str]:
    if not dataset_dir.exists():
        return []

    class_names = []
    for entry in dataset_dir.iterdir():
        if not entry.is_dir():
            continue
        if not include_utility_folders and entry.name in UTILITY_FOLDERS:
            continue
        class_names.append(entry.name)
    return sorted(class_names)


def resolve_dataset_dir(dataset_dir: str) -> Path:
    return Path(dataset_dir).expanduser().resolve(strict=False)


def is_path_inside(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def dataset_path_is_configured(dataset_dir: str) -> bool:
    normalized = dataset_dir.strip()
    return bool(normalized)


def dataset_location_message(dataset_dir: str) -> tuple[str, str]:
    examples = " o ".join(DATASET_PATH_EXAMPLES)
    if not dataset_path_is_configured(dataset_dir):
        return (
            "warning",
            "Configura una ruta real de dataset fuera del repo antes de entrenar o cargar clases nuevas. "
            f"Ejemplos: {examples}",
        )

    resolved = resolve_dataset_dir(dataset_dir)
    if is_path_inside(WORKSPACE_ROOT, resolved):
        return (
            "warning",
            f"La ruta actual del dataset esta dentro del repo: {resolved}. "
            "Se recomienda mover el dataset fuera del proyecto para no mezclar codigo con datos.",
        )

    return ("info", f"Dataset actual: {resolved}")


def sync_loaded_model(loaded_checkpoint, loaded_from_disk: bool, restore_dataset_dir: bool = True) -> None:
    metadata = dict(loaded_checkpoint.metadata)
    st.session_state.trained_model = loaded_checkpoint.model
    st.session_state.classes = list(loaded_checkpoint.classes)
    st.session_state.last_val_acc = metadata.get("val_accuracy")
    st.session_state.last_train_at = metadata.get("trained_at")
    st.session_state.checkpoint_metadata = metadata
    st.session_state.model_loaded_from_disk = loaded_from_disk

    saved_dataset_dir = metadata.get("dataset_dir")
    if restore_dataset_dir and saved_dataset_dir:
        st.session_state.dataset_dir = saved_dataset_dir

    checkpoint_path = metadata.get("checkpoint_path", st.session_state.checkpoint_path)
    trained_at = metadata.get("trained_at", "fecha no disponible")
    origin = "disco" if loaded_from_disk else "sesion actual"
    st.session_state.model_status_message = (
        f"Modelo disponible desde {origin}. Ultimo entrenamiento: {trained_at}. Checkpoint: {checkpoint_path}"
    )


def load_saved_model(force: bool = False, restore_dataset_dir: bool = True) -> bool:
    checkpoint_path = Path(st.session_state.checkpoint_path)
    if not checkpoint_exists(checkpoint_path):
        st.session_state.trained_model = None
        st.session_state.classes = []
        st.session_state.last_val_acc = None
        st.session_state.last_train_at = None
        st.session_state.checkpoint_metadata = {}
        st.session_state.model_loaded_from_disk = False
        if force:
            st.session_state.model_status_message = f"No existe un modelo guardado en {checkpoint_path}."
            log_event(f"No se encontro checkpoint en {checkpoint_path}.")
        return False

    try:
        loaded = load_checkpoint(checkpoint_path)
    except Exception as exc:
        st.session_state.trained_model = None
        st.session_state.classes = []
        st.session_state.last_val_acc = None
        st.session_state.last_train_at = None
        st.session_state.checkpoint_metadata = {}
        st.session_state.model_loaded_from_disk = False
        st.session_state.model_status_message = f"No se pudo cargar el checkpoint: {exc}"
        log_event(f"Error al cargar checkpoint: {exc}")
        return False

    sync_loaded_model(loaded, loaded_from_disk=True, restore_dataset_dir=restore_dataset_dir)
    log_event(f"Modelo cargado desde {checkpoint_path}.")
    return True


def checkpoint_dataset_warning() -> str | None:
    trained_classes = st.session_state.checkpoint_metadata.get("classes") or st.session_state.classes
    current_classes = dataset_class_names(Path(st.session_state.dataset_dir), include_utility_folders=False)
    if trained_classes and current_classes and set(trained_classes) != set(current_classes):
        return (
            "Las carpetas del dataset no coinciden con las clases del modelo cargado. "
            "Si cambiaste las clases, vuelve a entrenar el modelo."
        )
    return None


for key, default in {
    "logs": [],
    "trained_model": None,
    "classes": [],
    "last_val_acc": None,
    "last_train_at": None,
    "dataset_dir": DEFAULT_DATASET_DIR,
    "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
    "checkpoint_metadata": {},
    "model_loaded_from_disk": False,
    "model_status_message": "Sin modelo cargado.",
    "auto_load_attempted": False,
    "last_archive_signature": None,
}.items():
    st.session_state.setdefault(key, default)

if not st.session_state.auto_load_attempted:
    if not load_saved_model(force=False, restore_dataset_dir=True):
        st.session_state.model_status_message = (
            "Sin modelo cargado. Entrena uno nuevo o usa el checkpoint guardado en "
            f"{st.session_state.checkpoint_path}."
        )
    st.session_state.auto_load_attempted = True

st.title("Sistema de Clasificacion Automatizada de Senales Hostiles - FAR")

status_left, status_right = st.columns([3, 1])
with status_left:
    st.caption(f"Checkpoint configurado: {st.session_state.checkpoint_path}")
    st.info(st.session_state.model_status_message)
with status_right:
    if st.button("Cargar modelo guardado"):
        if load_saved_model(force=True, restore_dataset_dir=False):
            st.success("Modelo guardado cargado correctamente.")
        else:
            st.warning(st.session_state.model_status_message)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Visualizador de espectro")
    image_file = st.file_uploader("Cargar espectrograma/imagen", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file.getvalue(), use_container_width=True)

with col2:
    st.subheader("Resultados de clasificacion")
    confidence_threshold = st.slider("Umbral de confianza", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    auto_archive = st.checkbox("Auto-archivar", value=True)

    if image_file is None:
        st.info("Sube una imagen para obtener una prediccion.")
    elif st.session_state.trained_model is None:
        st.warning("No hay un modelo disponible. Entrena uno nuevo o carga el modelo guardado.")
    else:
        image_bytes = image_file.getvalue()
        try:
            top3 = predict_topk_from_bytes(
                st.session_state.trained_model,
                image_bytes,
                st.session_state.classes,
                topk=3,
            )
        except Exception as exc:
            log_event(f"Error de prediccion: {exc}")
            st.error(f"No se pudo clasificar la imagen: {exc}")
        else:
            best_label, best_prob = top3[0]
            st.markdown(f"### Medio detectado: **{best_label} ({best_prob * 100:.1f}%)**")
            st.progress(min(1.0, float(best_prob)))
            st.write("Top-3 candidatos:")
            for idx, (name, p) in enumerate(top3, start=1):
                st.write(f"{idx}. {name}: {p * 100:.1f}%")

            target_class = best_label if best_prob >= confidence_threshold else "NO_IDENTIFICADOS"
            if auto_archive:
                image_hash = hashlib.sha1(image_bytes).hexdigest()
                model_signature = st.session_state.checkpoint_metadata.get("trained_at", "session-model")
                archive_key = f"{image_hash}:{model_signature}:{st.session_state.checkpoint_path}"
                if st.session_state.last_archive_signature != archive_key:
                    target_dir = ARCHIVE_ROOT / target_class
                    target_dir.mkdir(parents=True, exist_ok=True)
                    suffix = Path(image_file.name).suffix.lower() or ".jpg"
                    filename = (
                        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        + f"_{target_class}_conf{int(best_prob * 100)}_{image_hash[:8]}{suffix}"
                    )
                    dest = target_dir / filename
                    dest.write_bytes(image_bytes)
                    st.session_state.last_archive_signature = archive_key
                    log_event(f"Deteccion {best_label} ({best_prob * 100:.1f}%). Guardado en {target_class}.")
                    st.caption(f"Imagen archivada en {dest}")
                else:
                    st.caption("La imagen ya fue archivada en esta sesion.")

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("Control SDR (simulado)")
    st.number_input("Frecuencia (MHz)", value=250.0, step=0.1)
    st.number_input("Ganancia (dB)", value=30.0, step=1.0)
    st.number_input("Sample rate (MHz)", value=2.4, step=0.1)
    st.radio("Modo de acceso", ["Exclusivo", "Compartido (rtl_tcp)"], horizontal=True)

with right:
    st.subheader("Captura automatica")
    st.checkbox("Activar monitoreo continuo cada 1 s", value=True)
    st.checkbox("Guardar solo cuando hay senal", value=True)
    st.number_input("Umbral de energia", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    st.text_input(
        "Carpeta base de datos real (usa una ruta externa al repo)",
        key="dataset_dir",
        help=(
            "Ejemplos: "
            + " | ".join(DATASET_PATH_EXAMPLES)
            + ". Las imagenes nuevas se guardan exactamente en esta ruta."
        ),
    )
    dataset_message_level, dataset_message = dataset_location_message(st.session_state.dataset_dir)
    if dataset_message_level == "warning":
        st.warning(dataset_message)
    else:
        st.info(dataset_message)

st.divider()

m1, m2 = st.columns([1, 1])
with m1:
    st.subheader("Gestion de base de datos y modelo")
    st.caption(f"Los modelos entrenados se guardan en {st.session_state.checkpoint_path}")
    st.caption(f"Las predicciones archivadas se guardan en {ARCHIVE_ROOT}")
    st.caption("Las clases nuevas se guardan en la ruta indicada en 'Carpeta base de datos real'.")

    new_class = st.text_input("Anadir nuevo medio hostil (nombre de clase)")
    new_files = st.file_uploader(
        "Imagenes del nuevo medio",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button("Anadir nuevo medio"):
        class_name = normalize_class_name(new_class)
        if not dataset_path_is_configured(st.session_state.dataset_dir):
            st.warning("Configura primero una ruta real de dataset fuera del repo.")
        elif class_name and new_files:
            class_dir = Path(st.session_state.dataset_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            saved_files = 0
            for f in new_files:
                (class_dir / Path(f.name).name).write_bytes(f.getvalue())
                saved_files += 1
            st.session_state.model_status_message = (
                "Dataset actualizado. Reentrena el modelo para incluir las nuevas imagenes."
            )
            log_event(f"Nuevo medio '{class_name}' anadido con {saved_files} imagenes.")
            st.success("Medio anadido correctamente. Reentrena el modelo para usarlo.")
        else:
            st.warning("Define un nombre de clase y sube imagenes.")

    epochs = st.number_input("Epocas", min_value=1, max_value=100, value=5, step=1)
    if st.button("Reentrenar modelo"):
        if not dataset_path_is_configured(st.session_state.dataset_dir):
            st.warning("Configura primero una ruta real de dataset fuera del repo antes de entrenar.")
        else:
            try:
                with st.spinner("Entrenando modelo por transferencia..."):
                    result = train_transfer(Path(st.session_state.dataset_dir), epochs=int(epochs))
                st.session_state.trained_model = result.model
                st.session_state.classes = result.classes
                st.session_state.last_val_acc = result.val_accuracy
                st.session_state.model_loaded_from_disk = False
                trained_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                st.session_state.last_train_at = trained_at

                try:
                    metadata = save_checkpoint(
                        model=result.model,
                        classes=result.classes,
                        dataset_dir=Path(st.session_state.dataset_dir),
                        val_accuracy=result.val_accuracy,
                        epochs=int(epochs),
                        output_path=Path(st.session_state.checkpoint_path),
                        trained_at=trained_at,
                    )
                except Exception as save_exc:
                    st.session_state.checkpoint_metadata = {
                        "classes": list(result.classes),
                        "dataset_dir": st.session_state.dataset_dir,
                        "trained_at": trained_at,
                        "val_accuracy": result.val_accuracy,
                    }
                    st.session_state.model_status_message = (
                        f"Modelo entrenado en memoria, pero no se pudo guardar el checkpoint: {save_exc}"
                    )
                    log_event(
                        f"Reentrenamiento completado ({epochs} epocas). No se pudo guardar checkpoint: {save_exc}"
                    )
                    st.warning("El modelo se entreno, pero no se pudo guardar en disco.")
                else:
                    st.session_state.checkpoint_metadata = metadata
                    st.session_state.model_status_message = (
                        "Modelo entrenado y guardado correctamente en "
                        f"{metadata['checkpoint_path']}"
                    )
                    log_event(
                        f"Reentrenamiento completado ({epochs} epocas). Precision validacion: "
                        f"{result.val_accuracy * 100:.1f}%. Checkpoint guardado."
                    )
                    st.success("Modelo entrenado y guardado correctamente.")
            except Exception as exc:
                log_event(f"Error de entrenamiento: {exc}")
                st.error(f"No se pudo entrenar el modelo: {exc}")

with m2:
    st.subheader("Clases disponibles")
    base = Path(st.session_state.dataset_dir)
    if base.exists():
        classes = [d for d in base.iterdir() if d.is_dir()]
        for class_dir in sorted(classes, key=lambda item: item.name.lower()):
            count = count_images(class_dir)
            st.write(f"- {class_dir.name} ({count} imagenes)")
    else:
        st.info("Aun no existe la carpeta base.")

    if st.session_state.last_train_at and st.session_state.last_val_acc is not None:
        st.markdown(
            f"**Ultimo entrenamiento:** {st.session_state.last_train_at}\n\n"
            f"**Precision validacion:** {st.session_state.last_val_acc * 100:.1f}%"
        )

    if st.session_state.classes:
        st.caption("Clases del modelo cargado: " + ", ".join(st.session_state.classes))

    dataset_warning = checkpoint_dataset_warning()
    if dataset_warning:
        st.warning(dataset_warning)

st.divider()
st.subheader("Registro de eventos")
for line in st.session_state.logs[:20]:
    st.code(line)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("INICIAR CAPTURA"):
        log_event("Iniciado monitoreo continuo cada 1 s.")
with c2:
    if st.button("DETENER"):
        log_event("Captura detenida.")
with c3:
    if st.button("GUARDAR CONFIGURACION"):
        cfg = Path("config_runtime.txt")
        cfg.write_text(
            f"dataset_dir={st.session_state.dataset_dir}\n"
            f"checkpoint_path={st.session_state.checkpoint_path}\n"
            f"confidence_threshold={confidence_threshold}\n"
            f"auto_archive={auto_archive}\n",
            encoding="utf-8",
        )
        log_event("Configuracion guardada en config_runtime.txt")
        st.success("Configuracion guardada")
