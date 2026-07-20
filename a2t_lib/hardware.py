"""Обнаружение GPU и рекомендации без загрузки модели Whisper."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []
_CUDA_LIBRARIES: list[object] = []
_CUDA_STATUS: CudaRuntimeStatus | None = None


@dataclass(frozen=True)
class CudaRuntimeStatus:
    ready: bool
    missing: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()


def configure_cuda_runtime() -> CudaRuntimeStatus:
    """Добавляет DLL из NVIDIA PyPI-пакетов и проверяет реальную загрузку runtime."""
    global _CUDA_STATUS
    if _CUDA_STATUS is not None:
        return _CUDA_STATUS
    if sys.platform != "win32":
        _CUDA_STATUS = CudaRuntimeStatus(ready=True)
        return _CUDA_STATUS

    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / component / "bin"
        for component in ("cublas", "cudnn", "cuda_nvrtc")
    ]
    directories = [path for path in candidates if path.is_dir()]
    for directory in directories:
        directory_text = str(directory)
        if directory_text not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
        try:
            handle = os.add_dll_directory(directory_text)
            _DLL_DIRECTORY_HANDLES.append(handle)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    missing: list[str] = []
    for library_name in ("cublas64_12.dll", "cudnn64_9.dll"):
        try:
            _CUDA_LIBRARIES.append(ctypes.WinDLL(library_name))
        except OSError:
            missing.append(library_name)
    _CUDA_STATUS = CudaRuntimeStatus(
        ready=not missing,
        missing=tuple(missing),
        directories=tuple(str(path) for path in directories),
    )
    return _CUDA_STATUS


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    driver: str = ""
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    compute_capability: str = ""
    runtime_ready: bool = False
    compute_types: tuple[str, ...] = ()

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    def summary(self) -> str:
        memory = f" · {self.memory_total_gb:.0f} ГБ VRAM" if self.memory_total_mb else ""
        state = " · CUDA готова" if self.runtime_ready else " · CUDA runtime не найден"
        return f"GPU {self.index}: {self.name}{memory}{state}"


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def detect_gpus() -> list[GpuInfo]:
    """Возвращает NVIDIA GPU из nvidia-smi, дополняя статусом CTranslate2."""
    runtime = configure_cuda_runtime()
    cuda_count = 0
    compute_types: tuple[str, ...] = ()
    try:
        import ctranslate2

        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count:
            compute_types = tuple(sorted(ctranslate2.get_supported_compute_types("cuda")))
    except Exception:
        pass

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=True,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return [
            GpuInfo(
                index=index,
                name="NVIDIA GPU",
                runtime_ready=runtime.ready,
                compute_types=compute_types,
            )
            for index in range(cuda_count)
        ]

    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
            total = int(parts[3])
            free = int(parts[4])
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=parts[1],
                driver=parts[2],
                memory_total_mb=total,
                memory_free_mb=free,
                compute_capability=parts[5],
                runtime_ready=index < cuda_count and runtime.ready,
                compute_types=compute_types if index < cuda_count else (),
            )
        )
    return gpus


def hardware_summary(gpus: list[GpuInfo] | None = None) -> str:
    detected = detect_gpus() if gpus is None else gpus
    if not detected:
        return "GPU не найден · будет использован CPU"
    return " | ".join(gpu.summary() for gpu in detected)


def vram_warning(model: str, compute_type: str, gpu: GpuInfo | None) -> str | None:
    """Возвращает понятное предупреждение для заведомо рискованной конфигурации."""
    if gpu is None or not gpu.memory_total_mb:
        return None
    if model.startswith("large") and compute_type == "float32" and gpu.memory_total_mb < 10240:
        return (
            f"Для {model} в float32 обычно требуется больше 10 ГБ VRAM, "
            f"а доступно {gpu.memory_total_gb:.0f} ГБ. Выберите float16 или пресет «Надёжный»."
        )
    if model.startswith("large") and compute_type == "float16" and gpu.memory_total_mb < 5120:
        return (
            f"Для {model} в float16 может не хватить {gpu.memory_total_gb:.0f} ГБ VRAM. "
            "Выберите int8_float16 или модель medium."
        )
    return None
