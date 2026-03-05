import os
import subprocess
import json
import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    ram_gb: float
    cpu_cores: int
    gpu_vram_gb: float
    backend: str  # "cuda", "rocm", "metal", "cpu"
    gpu_name: Optional[str] = None

    def effective_memory_gb(self) -> float:
        """Memory available for model inference."""
        if self.gpu_vram_gb > 0:
            return self.gpu_vram_gb
        # CPU inference: use ~60% of RAM, leaving headroom for OS + KV cache
        return self.ram_gb * 0.6


def _detect_ram_gb() -> float:
    plat = platform.system()
    if plat == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)

    elif plat == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)

    elif plat == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)

    import sys
    print("Warning: could not detect system RAM.", file=sys.stderr)
    return 0.0


def _detect_gpu() -> tuple:
    """Returns (vram_gb, backend, gpu_name)."""
    # NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip())
            return vram_mb / 1024, "cuda", name
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # AMD ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            for card_data in data.values():
                if isinstance(card_data, dict):
                    for k, v in card_data.items():
                        if "total" in k.lower():
                            return int(v) / (1024 ** 3), "rocm", "AMD GPU"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # macOS Metal
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for d in data.get("SPDisplaysDataType", []):
                    vram_str = d.get("spdisplays_vram") or d.get("spdisplays_vram_shared", "")
                    if vram_str:
                        parts = vram_str.split()
                        val = float(parts[0])
                        unit = parts[1].upper() if len(parts) > 1 else "GB"
                        if unit == "MB":
                            val /= 1024
                        return val, "metal", d.get("sppci_model", "Apple GPU")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    return 0.0, "cpu", None


def detect() -> HardwareProfile:
    ram_gb = _detect_ram_gb()
    cpu_cores = os.cpu_count() or 1
    vram_gb, backend, gpu_name = _detect_gpu()
    return HardwareProfile(
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        gpu_vram_gb=round(vram_gb, 1),
        backend=backend,
        gpu_name=gpu_name,
    )
