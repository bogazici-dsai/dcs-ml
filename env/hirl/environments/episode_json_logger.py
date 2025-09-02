# --- episode_json_logger.py ---
from pathlib import Path
import json, gzip, uuid, time, datetime
from typing import Optional, Dict, Any
from env.hirl.environments.json_numpy import NpEncoder

def now_iso_utc() -> str:
    # ISO8601 + 'Z'
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")

class EpisodeJSONLogger:
    """
    Her epizot için .jsonl (veya .jsonl.gz) dosya açar.
    Her step satır satır yazılır; epizot bitince summary kaydedilir.
    """
    def __init__(self, log_dir: str = "./episode_logs", gzip_enabled: bool = True, run_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.gzip_enabled = bool(gzip_enabled)
        # Koşu kimliği: aynı çalışmada açılan tüm epizot dosyalarını bağlamak için
        self.run_id = run_id or f"{datetime.datetime.utcnow():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self._fh = None
        self.episode_id = None
        self.episode_index = -1
        self._step_idx = -1
        self._t0_mono_ns = None
        self.meta = {}

    def start_episode(self, episode_index: int, meta: Optional[Dict[str, Any]] = None):
        # Dosyayı aç, epizot sayaçlarını sıfırla
        self.episode_index = int(episode_index)
        self.episode_id = f"ep-{uuid.uuid4().hex}"
        self._step_idx = -1
        self._t0_mono_ns = time.monotonic_ns()
        self.meta = dict(meta or {})
        # Dosya adı
        base = f"{self.run_id}__ep{self.episode_index:06d}__{self.episode_id}.jsonl"
        path = self.log_dir / (base + (".gz" if self.gzip_enabled else ""))
        # Aç
        self._fh = gzip.open(path, "at", encoding="utf-8") if self.gzip_enabled else open(path, "at", encoding="utf-8")
        # Başlangıç kaydı
        self._write({
            "type": "episode_start",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "ts_wall": now_iso_utc(),
            "ts_unix_ms": int(time.time() * 1000),
            "meta": self.meta,
        })

    def log_step(self, record: Dict[str, Any]):
        assert self._fh is not None, "Call start_episode() first."
        self._step_idx += 1
        t_wall_iso = now_iso_utc()
        t_unix_ms = int(time.time() * 1000)
        t_mono_ns = time.monotonic_ns()
        rec = {
            "type": "step",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "step_index": self._step_idx,
            "ts_wall": t_wall_iso,         # insanlar için okunabilir
            "ts_unix_ms": t_unix_ms,       # mutlak zaman
            "dt_mono_ns": t_mono_ns - self._t0_mono_ns,  # monotonic (sıra garantisi/latency analizi)
        }
        rec.update(record)  # sizin sağladığınız env.step() verileri
        self._write(rec)

    def end_episode(self, summary: Optional[Dict[str, Any]] = None):
        if self._fh is None:
            return
        self._write({
            "type": "episode_end",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "ts_wall": now_iso_utc(),
            "ts_unix_ms": int(time.time() * 1000),
            "summary": dict(summary or {}),
        })
        self._fh.close()
        self._fh = None

    def _write(self, obj: Dict[str, Any]):
        line = json.dumps(obj, ensure_ascii=False, sort_keys=True, cls=NpEncoder)
        self._fh.write(line + "\n")
        self._fh.flush()
