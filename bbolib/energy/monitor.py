"""
Cross-platform energy monitoring.

Priority:
  1. Linux RAPL via /sys/class/powercap  (CPU + DRAM, accurate)
  2. psutil CPU times * TDP estimate     (rough, always available)
"""
import time
import platform
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnergySample:
    timestamp: float
    joules_rapl: float | None    # None if RAPL unavailable
    cpu_percent: float           # from psutil
    wall_time: float


@dataclass
class EnergyReport:
    total_joules_rapl: float | None
    total_joules_estimated: float | None
    wall_time_seconds: float
    samples: list[EnergySample] = field(default_factory=list)
    measurement_method: str = "unknown"

    @property
    def best_estimate_joules(self) -> float | None:
        return self.total_joules_rapl if self.total_joules_rapl is not None else self.total_joules_estimated


# ---------------------------------------------------------------------------
# RAPL reader (Linux only)
# ---------------------------------------------------------------------------

_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl")

def _rapl_available() -> bool:
    return platform.system() == "Linux" and _RAPL_ROOT.exists()

def _read_rapl_uj() -> float:
    """Sum energy_uj across all RAPL packages."""
    total = 0.0
    for pkg in _RAPL_ROOT.glob("intel-rapl:*"):
        energy_file = pkg / "energy_uj"
        if energy_file.exists():
            total += int(energy_file.read_text().strip())
    return total  # in microjoules

def _rapl_max_range_uj() -> float:
    """Max range before counter wraps (to handle overflow)."""
    for pkg in _RAPL_ROOT.glob("intel-rapl:*"):
        f = pkg / "max_energy_range_uj"
        if f.exists():
            return float(f.read_text().strip())
    return 2 ** 32  # fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EnergyMonitor:
    """
    Context manager that measures energy during a code block.

    Usage:
        with EnergyMonitor(interval=0.1) as mon:
            run_algorithm(...)
        print(mon.report.best_estimate_joules)
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._samples: list[EnergySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_wall: float = 0.0
        self._start_rapl_uj: float | None = None
        self.report: EnergyReport | None = None

    def __enter__(self):
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None

        self._start_wall = time.perf_counter()
        self._start_rapl_uj = _read_rapl_uj() if _rapl_available() else None
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

        wall = time.perf_counter() - self._start_wall

        # RAPL delta
        rapl_joules = None
        if self._start_rapl_uj is not None:
            end_uj = _read_rapl_uj()
            delta_uj = end_uj - self._start_rapl_uj
            if delta_uj < 0:
                delta_uj += _rapl_max_range_uj()  # handle counter wrap
            rapl_joules = delta_uj / 1e6

        # CPU-time estimate (rough): average cpu% * TDP-based watt estimate
        estimated = None
        if self._samples and self._psutil is not None:
            avg_cpu = sum(s.cpu_percent for s in self._samples) / len(self._samples)
            # Rough: assume 15 W TDP for a typical laptop core under load
            estimated = avg_cpu / 100.0 * 15.0 * wall

        method = "RAPL" if rapl_joules is not None else ("psutil-estimate" if estimated is not None else "none")
        self.report = EnergyReport(
            total_joules_rapl=rapl_joules,
            total_joules_estimated=estimated,
            wall_time_seconds=wall,
            samples=self._samples,
            measurement_method=method,
        )

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            cpu = self._psutil.cpu_percent(interval=None) if self._psutil else 0.0
            rapl = _read_rapl_uj() if self._start_rapl_uj is not None else None
            self._samples.append(EnergySample(
                timestamp=time.perf_counter() - self._start_wall,
                joules_rapl=rapl,
                cpu_percent=cpu,
                wall_time=time.perf_counter() - self._start_wall,
            ))
