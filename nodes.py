"""
ComfyUI-SpeakerSplit
Sprecher-Trennung (Diarization) für Voice-Referenzen in Video-Generierung.

Pipeline:
  1. Audio -> Mono 16 kHz (Analyse-Kopie)
  2. VAD (Silero, Fallback: Energie-VAD) -> Sprach-Segmente
  3. Sliding Windows -> ECAPA-TDNN Speaker-Embeddings (SpeechBrain)
  4. Agglomeratives Clustering (cosine) auf n Sprecher
  5. Frame-Voting + Median-Glättung + Overlap-Verwerfung
  6. Rückprojektion auf Original-Samplerate/Kanäle
"""

import os
import math

import numpy as np
import torch
import torchaudio

try:
    import folder_paths
except Exception:  # Standalone-Test außerhalb von ComfyUI
    folder_paths = None


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

_EMBED_MODEL = None
_VAD_MODEL = None
ANALYSIS_SR = 16000


def _get_device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _models_dir(sub):
    if folder_paths is not None:
        base = folder_paths.models_dir
    else:
        base = os.path.join(os.path.dirname(__file__), "_models")
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


INSTALL_HINT = (
    "SpeechBrain fehlt. In der ComfyUI-Python-Umgebung installieren:\n"
    "    python.exe -m pip install speechbrain --no-deps\n"
    "    python.exe -m pip install hyperpyyaml joblib huggingface_hub "
    "sentencepiece scipy tqdm packaging scikit-learn\n"
    "WICHTIG: --no-deps verwenden, sonst überschreibt pip die CUDA-Version "
    "von torch/torchaudio mit CPU-Wheels.\n"
    "Alternative ohne Installation: im Node embedder='mfcc' setzen "
    "(funktioniert ohne Zusatzpakete, ist aber ungenauer)."
)


def _load_embedder(device):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    EncoderClassifier = None
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
    except Exception:
        try:
            from speechbrain.pretrained import EncoderClassifier  # speechbrain < 1.0
        except Exception as e:
            raise RuntimeError(f"{INSTALL_HINT}\n(Originalfehler: {e})")

    savedir = _models_dir(os.path.join("speaker_embed", "spkrec-ecapa-voxceleb"))
    _EMBED_MODEL = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": str(device)},
    )
    return _EMBED_MODEL


# --- Fallback-Embedder ohne Zusatzpakete (nur torchaudio) ------------------

_MFCC_TF = None


def _get_mfcc_transform(sr):
    global _MFCC_TF
    if _MFCC_TF is None:
        _MFCC_TF = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80,
                       "f_min": 50.0, "f_max": 7600.0},
        )
    return _MFCC_TF


@torch.no_grad()
def _mfcc_embed_windows(wav_np, sr, windows, batch_size=64):
    """MFCC-Statistik-Embedding: Mittelwert + Std über Zeit, plus Deltas.

    Deutlich schwächer als ECAPA - trennt zuverlässig nur klar
    unterschiedliche Stimmen (z. B. männlich/weiblich). Braucht dafür
    keinerlei Zusatzpakete.
    """
    tf = _get_mfcc_transform(sr)
    target_len = int(sr * 1.5)
    feats = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i: i + batch_size]
        buf = torch.zeros(len(chunk), target_len)
        for j, (s, e) in enumerate(chunk):
            a, b = int(s * sr), int(e * sr)
            seg = torch.from_numpy(wav_np[a:b]).float()
            if seg.numel() == 0:
                continue
            if seg.numel() >= target_len:
                buf[j] = seg[:target_len]
            else:
                reps = math.ceil(target_len / seg.numel())
                buf[j] = seg.repeat(reps)[:target_len]
        m = tf(buf)                              # (B, n_mfcc, T)
        d = m[:, :, 1:] - m[:, :, :-1]           # Delta
        vec = torch.cat([m.mean(-1), m.std(-1),
                         d.mean(-1), d.std(-1)], dim=1)
        feats.append(vec.cpu().numpy())
    X = np.concatenate(feats, axis=0)
    # CMVN über die ganze Datei: entfernt Mikrofon-/Raumanteil
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def _load_vad():
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL
    from silero_vad import load_silero_vad
    _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def _energy_vad(wav, sr, rel_db=-35.0, win_ms=30, min_speech_ms=250, min_silence_ms=150):
    """Fallback-VAD ohne externe Abhängigkeit."""
    win = max(1, int(sr * win_ms / 1000))
    n = len(wav) // win
    if n == 0:
        return [(0.0, len(wav) / sr)]
    frames = wav[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = db.max() + rel_db
    voiced = db > thr

    segs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, n))

    # Lücken schließen
    min_gap = min_silence_ms / win_ms
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    min_len = min_speech_ms / win_ms
    out = [(s * win / sr, e * win / sr) for s, e in merged if (e - s) >= min_len]
    return out


def _vad_segments(wav_np, sr, threshold=0.5, min_speech_ms=250, min_silence_ms=150):
    try:
        from silero_vad import get_speech_timestamps
        model = _load_vad()
        t = torch.from_numpy(wav_np).float()
        stamps = get_speech_timestamps(
            t, model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )
        segs = [(s["start"] / sr, s["end"] / sr) for s in stamps]
        if segs:
            return segs, "silero-vad"
    except Exception as e:
        print(f"[SpeakerSplit] Silero-VAD nicht verfügbar ({e}) -> Energie-VAD.")
    return _energy_vad(wav_np, sr, min_speech_ms=min_speech_ms,
                       min_silence_ms=min_silence_ms), "energy-vad"


def _audio_to_mono16k(audio):
    """ComfyUI AUDIO dict -> (numpy mono 16k, original waveform (C,T), sr)."""
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wf.dim() == 3:
        if wf.shape[0] > 1:
            print("[SpeakerSplit] Batch > 1: es wird nur das erste Element verarbeitet.")
        wf = wf[0]
    elif wf.dim() == 1:
        wf = wf.unsqueeze(0)
    wf = wf.float().cpu()

    mono = wf.mean(dim=0, keepdim=True)
    if sr != ANALYSIS_SR:
        mono = torchaudio.functional.resample(mono, sr, ANALYSIS_SR)
    mono_np = mono.squeeze(0).numpy().astype(np.float32)
    peak = np.abs(mono_np).max()
    if peak > 0:
        mono_np = mono_np / peak * 0.95
    return mono_np, wf, sr


def _make_windows(segments, win_s, hop_s, min_win_s):
    """Sliding Windows innerhalb der Sprach-Segmente."""
    wins = []
    for s, e in segments:
        dur = e - s
        if dur < min_win_s:
            continue
        if dur <= win_s:
            wins.append((s, e))
            continue
        t = s
        while t + win_s <= e + 1e-6:
            wins.append((t, t + win_s))
            t += hop_s
        if e - (t - hop_s + win_s) > min_win_s * 0.5:
            wins.append((max(s, e - win_s), e))
    return wins


def _embed_windows(wav_np, sr, windows, device, method="auto"):
    """Dispatcher: ECAPA wenn möglich, sonst MFCC. Gibt (X, name) zurück."""
    if method == "mfcc":
        return _mfcc_embed_windows(wav_np, sr, windows), "mfcc"

    try:
        return _ecapa_embed_windows(wav_np, sr, windows, device), "ecapa"
    except Exception as e:
        if method == "ecapa":
            raise
        print(f"[SpeakerSplit] ECAPA nicht verfügbar -> Fallback auf MFCC.\n{e}")
        return _mfcc_embed_windows(wav_np, sr, windows), "mfcc (Fallback)"


@torch.no_grad()
def _ecapa_embed_windows(wav_np, sr, windows, device, batch_size=32):
    model = _load_embedder(device)
    target_len = int(sr * 1.5)
    embs = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i: i + batch_size]
        buf = torch.zeros(len(chunk), target_len)
        for j, (s, e) in enumerate(chunk):
            a, b = int(s * sr), int(e * sr)
            seg = torch.from_numpy(wav_np[a:b]).float()
            if seg.numel() == 0:
                continue
            if seg.numel() >= target_len:
                buf[j] = seg[:target_len]
            else:
                reps = math.ceil(target_len / seg.numel())
                buf[j] = seg.repeat(reps)[:target_len]
        emb = model.encode_batch(buf.to(device))  # (B, 1, D)
        emb = emb.squeeze(1).cpu().numpy()
        embs.append(emb)
    X = np.concatenate(embs, axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def _kmeans_cosine(X, n, iters=60, seed=0):
    """Spherical k-means als Fallback, falls scikit-learn fehlt."""
    rng = np.random.default_rng(seed)
    # k-means++ Initialisierung auf Cosinus-Distanz
    cents = [X[rng.integers(len(X))]]
    for _ in range(n - 1):
        d = 1.0 - (X @ np.stack(cents).T).max(axis=1)
        d = np.clip(d, 0, None) ** 2
        if d.sum() <= 0:
            cents.append(X[rng.integers(len(X))])
        else:
            cents.append(X[rng.choice(len(X), p=d / d.sum())])
    cents = np.stack(cents)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = (X @ cents.T).argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(n):
            m = labels == k
            if m.any():
                c = X[m].mean(axis=0)
                cents[k] = c / (np.linalg.norm(c) + 1e-9)
    return labels


def _silhouette_cosine(X, labels):
    """Silhouette-Score auf Cosinus-Distanz, ohne sklearn."""
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    scores = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == k].mean() for k in uniq if k != labels[i])
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else -1.0


def _cluster(X, n_speakers):
    try:
        from sklearn.cluster import AgglomerativeClustering

        def _fit(n):
            try:
                cl = AgglomerativeClustering(n_clusters=n, metric="cosine",
                                             linkage="average")
            except TypeError:
                cl = AgglomerativeClustering(n_clusters=n, affinity="cosine",
                                             linkage="average")
            return cl.fit_predict(X)
    except ImportError:
        print("[SpeakerSplit] scikit-learn fehlt -> numpy-Fallback (spherical k-means). "
              "Für bessere Ergebnisse: pip install scikit-learn")

        def _fit(n):
            return _kmeans_cosine(X, n)

    if n_speakers > 0:
        n = min(n_speakers, len(X))
        return _fit(n), n

    # auto: 2..4 per Silhouette
    best, best_score, best_n = None, -2.0, 2
    for n in range(2, min(5, len(X))):
        lab = _fit(n)
        if len(set(lab)) < 2:
            continue
        score = _silhouette_cosine(X, lab)
        if score > best_score:
            best, best_score, best_n = lab, score, n
    if best is None:
        return np.zeros(len(X), dtype=int), 1
    print(f"[SpeakerSplit] Auto-Erkennung: {best_n} Sprecher (Silhouette {best_score:.3f})")
    return best, best_n


def _centroid_confidence(X, labels, n):
    """Margin zwischen bestem und zweitbestem Cluster-Centroid (0..2)."""
    cents = []
    for k in range(n):
        m = labels == k
        c = X[m].mean(axis=0) if m.any() else np.zeros(X.shape[1])
        cents.append(c / (np.linalg.norm(c) + 1e-9))
    cents = np.stack(cents)
    sims = X @ cents.T
    if n == 1:
        return np.ones(len(X)), cents
    part = np.sort(sims, axis=1)
    margin = part[:, -1] - part[:, -2]
    return margin, cents


def _frame_labels(windows, labels, margins, n, total_s, frame_s, margin_thr):
    n_frames = max(1, int(math.ceil(total_s / frame_s)))
    votes = np.zeros((n_frames, n), dtype=np.float32)
    for (s, e), lab, mg in zip(windows, labels, margins):
        if mg < margin_thr:
            continue
        a = int(s / frame_s)
        b = min(n_frames, int(math.ceil(e / frame_s)))
        if b > a:
            votes[a:b, lab] += float(mg)
    out = np.full(n_frames, -1, dtype=int)
    has = votes.sum(axis=1) > 0
    out[has] = votes[has].argmax(axis=1)
    return out


def _smooth(frames, kernel=5):
    if kernel <= 1:
        return frames
    half = kernel // 2
    out = frames.copy()
    for i in range(len(frames)):
        if frames[i] < 0:
            continue
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        window = frames[lo:hi]
        window = window[window >= 0]
        if len(window) == 0:
            continue
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def _frames_to_segments(frames, frame_s, min_seg_s):
    segs = []
    cur_lab, start = None, 0
    for i, lab in enumerate(list(frames) + [-99]):
        if lab != cur_lab:
            if cur_lab is not None and cur_lab >= 0:
                s, e = start * frame_s, i * frame_s
                if e - s >= min_seg_s:
                    segs.append((int(cur_lab), s, e))
            cur_lab, start = lab, i
    return segs


def _fade(x, n):
    if n <= 0 or x.shape[-1] < 2 * n:
        return x
    ramp = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)
    x[..., :n] *= ramp
    x[..., -n:] *= ramp.flip(0)
    return x


def _build_track(orig_wf, sr, segs, mode, pad_s, fade_ms):
    """orig_wf: (C, T) -> AUDIO dict"""
    C, T = orig_wf.shape
    fade_n = int(sr * fade_ms / 1000)

    if not segs:
        return {"waveform": torch.zeros(1, C, 1), "sample_rate": sr}

    if mode == "mask (Timing bleibt erhalten)":
        out = torch.zeros(C, T)
        for _, s, e in segs:
            a = max(0, int((s - pad_s) * sr))
            b = min(T, int((e + pad_s) * sr))
            if b <= a:
                continue
            piece = orig_wf[:, a:b].clone()
            out[:, a:b] += _fade(piece, fade_n)
        out = out.clamp(-1.0, 1.0)
    else:  # concat
        pieces = []
        for _, s, e in segs:
            a = max(0, int((s - pad_s) * sr))
            b = min(T, int((e + pad_s) * sr))
            if b <= a:
                continue
            pieces.append(_fade(orig_wf[:, a:b].clone(), fade_n))
        out = torch.cat(pieces, dim=1) if pieces else torch.zeros(C, 1)

    return {"waveform": out.unsqueeze(0), "sample_rate": sr}


# --------------------------------------------------------------------------
# Node 1: Diarization / Split
# --------------------------------------------------------------------------

class SpeakerSplitDiarize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Eingangs-Audio mit mehreren Sprechern.\n"
                               "Am besten sauber, ohne Musik/Hall. Bei Batch > 1 wird "
                               "nur das erste Element verarbeitet.\n"
                               "Faustregel: mindestens 6-10 s Sprache pro Person, "
                               "sonst wird das Clustering unzuverlässig."
                }),
                "num_speakers": ("INT", {
                    "default": 2, "min": 0, "max": 6, "step": 1,
                    "tooltip": "Anzahl der Sprecher im Audio.\n"
                               "2 oder 3 = exakte Vorgabe (deutlich robuster, immer bevorzugen).\n"
                               "0 = Auto-Erkennung von 2-4 Sprechern per Silhouette-Score.\n"
                               "Zu hoch gesetzt -> eine Stimme wird auf mehrere Spuren zerrissen.\n"
                               "Zu niedrig -> zwei Personen landen auf derselben Spur."
                }),
                "output_mode": (["concat (für Voice-Referenz)",
                                 "mask (Timing bleibt erhalten)"], {
                    "default": "concat (für Voice-Referenz)",
                    "tooltip": "concat: nur die Sprachanteile dieser Person, lückenlos "
                               "aneinandergehängt. Kurz, dicht, ideal als Voice-Clone-Referenz. "
                               "Das Timing zum Video geht verloren.\n"
                               "mask: Originallänge bleibt erhalten, alle anderen Sprecher werden "
                               "stumm. Lippensynchron zum Video, aber viel Stille -> als "
                               "Clone-Referenz schlechter geeignet."
                }),
                "window_seconds": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Länge des Analysefensters für ein Speaker-Embedding.\n"
                               "1.0-1.5 s: reagiert schnell auf Sprecherwechsel, aber "
                               "unschärfere Stimm-Signatur.\n"
                               "2.0-2.5 s: stabiler bei ähnlichen Stimmen (z. B. zwei weibliche "
                               "Stimmen in gleicher Tonlage), verschmiert aber kurze Einwürfe.\n"
                               "Unter 1.0 s liefert ECAPA kaum noch brauchbare Embeddings."
                }),
                "hop_seconds": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "Schrittweite zwischen zwei Analysefenstern.\n"
                               "Richtwert: die Hälfte von window_seconds.\n"
                               "Kleiner = zeitlich präzisere Sprechergrenzen, aber mehr Fenster "
                               "und damit längere Rechenzeit (etwa linear).\n"
                               "Größer = schneller, Sprecherwechsel werden ungenauer gesetzt."
                }),
                "min_segment_seconds": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Mindestlänge eines zusammenhängenden Sprecher-Abschnitts. "
                               "Kürzeres wird komplett verworfen.\n"
                               "Höher (1.0-1.5): filtert Fehlzuordnungen und Zwischenrufe wie "
                               "'mhm', 'ja' heraus - sauberer für Referenzclips.\n"
                               "Niedriger (0.2-0.4): behält schnelle Wortwechsel, holt sich aber "
                               "mehr Fremdstimme in die Spur."
                }),
                "overlap_margin": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Sicherheitsabstand zwischen bestem und zweitbestem Sprecher-"
                               "Cluster (Cosinus-Differenz). Fenster darunter gelten als unsicher "
                               "und werden verworfen statt geraten.\n"
                               "0.10-0.15: aggressiv - wirft gleichzeitiges Sprechen und "
                               "Grenzfälle raus. Empfohlen für Voice-Referenzen.\n"
                               "0.0-0.03: behält alles, auch Overlap-Stellen mit falscher Stimme.\n"
                               "Zu hoch -> Spuren werden sehr kurz oder leer."
                }),
                "vad_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Empfindlichkeit der Sprach-Erkennung (Silero-VAD).\n"
                               "Niedriger (0.2-0.35): erkennt auch leise/genuschelte Sprache, "
                               "nimmt aber Atmer, Husten und Störgeräusche mit.\n"
                               "Höher (0.6-0.8): nur klare, laute Sprache - sauber, aber "
                               "Wortanfänge und leise Passagen fallen weg.\n"
                               "Wirkt nur, wenn silero-vad installiert ist (sonst Energie-VAD)."
                }),
                "pad_seconds": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Puffer, der vor und nach jedem Segment zusätzlich "
                               "mitgeschnitten wird.\n"
                               "Verhindert abgeschnittene Wortanfänge und -enden.\n"
                               "Über ca. 0.15 s zieht man bei schnellen Dialogen die "
                               "Nachbarstimme mit in die Spur."
                }),
                "fade_ms": ("INT", {
                    "default": 20, "min": 0, "max": 200, "step": 5,
                    "tooltip": "Ein-/Ausblendung an jeder Schnittkante in Millisekunden.\n"
                               "15-30 ms entfernt Klick-/Knackgeräusche an den Übergängen.\n"
                               "0 = harte Schnitte (knackt meist).\n"
                               "Über 50 ms frisst hörbar Sprachanfang weg."
                }),
                "smooth_frames": ("INT", {
                    "default": 5, "min": 1, "max": 21, "step": 2,
                    "tooltip": "Median-Glättung der Sprecher-Zuordnung, gerechnet in "
                               "Frames zu je 100 ms (5 = 500 ms Fenster).\n"
                               "Höher (7-11): unterdrückt einzelne Ausreißer-Frames, "
                               "Sprecherwechsel werden aber träge.\n"
                               "1 = keine Glättung, Zuordnung kann zwischen den Spuren flackern.\n"
                               "Nur ungerade Werte sinnvoll."
                }),
                # WICHTIG: Neue Widgets immer ans ENDE anhängen. ComfyUI ordnet
                # gespeicherte Werte positionell zu - eine Einfügung in der Mitte
                # verschiebt alle folgenden Werte und zerlegt bestehende Workflows.
                "embedder": (["auto", "ecapa", "mfcc"], {
                    "default": "auto",
                    "tooltip": "Verfahren zur Stimm-Erkennung.\n"
                               "auto: nimmt ecapa, fällt bei fehlendem Paket automatisch "
                               "auf mfcc zurück statt abzubrechen.\n"
                               "ecapa: neuronales Speaker-Embedding (SpeechBrain), klar "
                               "genauer - trennt auch ähnliche Stimmen. Braucht "
                               "'pip install speechbrain --no-deps'.\n"
                               "mfcc: eingebauter Fallback, nur torchaudio nötig. Läuft "
                               "sofort und schnell, trennt aber zuverlässig nur deutlich "
                               "unterschiedliche Stimmen (z. B. männlich/weiblich)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("speaker_1", "speaker_2", "speaker_3", "report")
    OUTPUT_TOOLTIPS = (
        "Sprecher mit der meisten Redezeit.",
        "Sprecher mit der zweitmeisten Redezeit.",
        "Dritter Sprecher - leer (1 Sample), wenn num_speakers < 3.",
        "Statistik zum Prüfen: Gesamtlänge, Sprachanteil, Anzahl Fenster, erkannte "
        "Cluster sowie Redezeit, Segmentzahl und Zeitstempel pro Sprecher. "
        "An einen PreviewText-Node hängen - wenn ein Sprecher über 90 % hat, "
        "ist das Clustering schiefgegangen.",
    )
    DESCRIPTION = ("Trennt ein Audio mit mehreren Sprechern in einzelne Spuren "
                   "(VAD -> ECAPA-Embeddings -> Clustering). Für Voice-Clone-Referenzen "
                   "danach 'Voice Reference Extract' anhängen.")
    FUNCTION = "run"
    CATEGORY = "audio/speaker"

    def run(self, audio, num_speakers, output_mode, window_seconds, hop_seconds,
            min_segment_seconds, overlap_margin, vad_threshold, pad_seconds,
            fade_ms, smooth_frames, embedder="auto"):

        device = _get_device()
        mono, orig_wf, sr = _audio_to_mono16k(audio)
        total_s = len(mono) / ANALYSIS_SR

        vad_segs, vad_name = _vad_segments(mono, ANALYSIS_SR, threshold=vad_threshold)
        speech_s = sum(e - s for s, e in vad_segs)
        if not vad_segs:
            empty = {"waveform": torch.zeros(1, orig_wf.shape[0], 1), "sample_rate": sr}
            return (empty, empty, empty, "Keine Sprache gefunden.")

        windows = _make_windows(vad_segs, window_seconds, hop_seconds, 0.35)
        if len(windows) < 2:
            full = {"waveform": orig_wf.unsqueeze(0), "sample_rate": sr}
            empty = {"waveform": torch.zeros(1, orig_wf.shape[0], 1), "sample_rate": sr}
            return (full, empty, empty, "Zu wenig Sprachmaterial für Clustering.")

        X, emb_name = _embed_windows(mono, ANALYSIS_SR, windows, device, embedder)
        labels, n_found = _cluster(X, num_speakers)
        margins, _ = _centroid_confidence(X, labels, n_found)

        frame_s = 0.1
        frames = _frame_labels(windows, labels, margins, n_found, total_s,
                               frame_s, overlap_margin)
        frames = _smooth(frames, smooth_frames)
        segs = _frames_to_segments(frames, frame_s, min_segment_seconds)

        # Sprecher nach Sprechzeit sortieren (Speaker 1 = meiste Redezeit)
        dur = {}
        for lab, s, e in segs:
            dur[lab] = dur.get(lab, 0.0) + (e - s)
        order = sorted(dur.keys(), key=lambda k: -dur[k])

        outs = []
        lines = [
            f"Datei: {total_s:.1f}s  |  Sprache: {speech_s:.1f}s  |  VAD: {vad_name}",
            f"Fenster: {len(windows)}  |  Cluster: {n_found}"
            + ("  (auto)" if num_speakers == 0 else ""),
            f"Embedder: {emb_name}  |  Modus: {output_mode}",
            "",
        ]

        for idx in range(3):
            if idx < len(order):
                lab = order[idx]
                spk_segs = [s for s in segs if s[0] == lab]
                track = _build_track(orig_wf, sr, spk_segs, output_mode,
                                     pad_seconds, fade_ms)
                outs.append(track)
                lines.append(
                    f"Speaker {idx+1}: {dur[lab]:.1f}s in {len(spk_segs)} Segmenten "
                    f"({dur[lab]/max(speech_s,1e-6)*100:.0f}% der Sprache)"
                )
                preview = ", ".join(f"{s:.1f}-{e:.1f}" for _, s, e in spk_segs[:8])
                lines.append(f"    {preview}" + (" ..." if len(spk_segs) > 8 else ""))
            else:
                outs.append({"waveform": torch.zeros(1, orig_wf.shape[0], 1),
                             "sample_rate": sr})
                lines.append(f"Speaker {idx+1}: -")

        return (outs[0], outs[1], outs[2], "\n".join(lines))


# --------------------------------------------------------------------------
# Node 2: Voice-Referenz extrahieren
# --------------------------------------------------------------------------

class VoiceReferenceExtract:
    """Schneidet aus einer Einzel-Sprecher-Spur den saubersten Abschnitt
    mit gewünschter Länge – ideal als Voice-Clone-Referenz."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Spur mit nur EINER Stimme, z. B. speaker_1 aus dem "
                               "Speaker-Split-Node.\n"
                               "Bei gemischtem Audio wird die Referenz unbrauchbar - "
                               "vorher trennen."
                }),
                "target_seconds": ("FLOAT", {
                    "default": 10.0, "min": 2.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Gewünschte Länge des Referenzclips.\n"
                               "8-15 s ist der Sweetspot für die meisten Voice-Clone-Modelle.\n"
                               "Unter 5 s: Stimmcharakter wird nicht sauber erfasst.\n"
                               "Über 20 s: bringt kaum noch Qualität, erhöht nur das Risiko, "
                               "Störstellen mitzunehmen.\n"
                               "Ist das Eingangs-Audio kürzer, wird es komplett durchgereicht."
                }),
                "normalize_db": ("FLOAT", {
                    "default": -3.0, "min": -30.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Ziel-Spitzenpegel in dBFS (Peak-Normalisierung).\n"
                               "-3 dB ist der Standard: laut, aber mit Headroom gegen Clipping.\n"
                               "-6 bis -9 dB, wenn das Zielmodell empfindlich auf laute "
                               "Eingaben reagiert.\n"
                               "0 dB = Vollaussteuerung, kann bei Weiterverarbeitung zerren."
                }),
                "fade_ms": ("INT", {
                    "default": 15, "min": 0, "max": 200, "step": 5,
                    "tooltip": "Ein-/Ausblendung an Anfang und Ende des Referenzclips.\n"
                               "10-20 ms verhindert Knackser am Schnitt.\n"
                               "0 = harter Schnitt."
                }),
                "force_mono": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Kanäle zu Mono zusammenmischen.\n"
                               "An lassen: praktisch alle Voice-Clone- und TTS-Modelle "
                               "erwarten Mono.\n"
                               "Aus nur, wenn die Stereo-Information bewusst erhalten "
                               "bleiben soll."
                }),
                "resample_to": ([0, 16000, 22050, 24000, 32000, 44100, 48000], {
                    "default": 0,
                    "tooltip": "Ziel-Samplerate der Referenz.\n"
                               "0 = unverändert lassen.\n"
                               "24000: gängig für Voice-Clone/TTS (XTTS, F5, CosyVoice).\n"
                               "16000: ältere Modelle und Whisper-basierte Pipelines.\n"
                               "Hochskalieren von einer niedrigen Quelle bringt keine "
                               "Qualität zurück - nur setzen, wenn das Zielmodell es fordert."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("reference", "info")
    OUTPUT_TOOLTIPS = (
        "Der ausgeschnittene, normalisierte Referenzclip - direkt an SaveAudio oder "
        "an den Voice-Eingang des Video-/TTS-Nodes.",
        "Kontrollinfo: tatsächliche Länge, Startzeit im Quellmaterial, Samplerate, "
        "Kanalzahl und Zielpegel.",
    )
    DESCRIPTION = ("Schneidet aus einer Einzel-Sprecher-Spur den lautheitsmäßig "
                   "gleichmäßigsten Abschnitt der gewünschten Länge heraus und "
                   "bereitet ihn als Voice-Clone-Referenz auf.")
    FUNCTION = "run"
    CATEGORY = "audio/speaker"

    def run(self, audio, target_seconds, normalize_db, fade_ms, force_mono, resample_to):
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wf.dim() == 3:
            wf = wf[0]
        wf = wf.float().cpu()

        if force_mono and wf.shape[0] > 1:
            wf = wf.mean(dim=0, keepdim=True)

        T = wf.shape[1]
        want = int(target_seconds * sr)

        if T <= want:
            best = wf.clone()
            pos = 0.0
        else:
            mono = wf.mean(dim=0).numpy()
            hop = max(1, int(sr * 0.25))
            frame = max(1, int(sr * 0.25))
            n = (T - frame) // hop + 1
            rms = np.array([
                np.sqrt((mono[i * hop: i * hop + frame] ** 2).mean() + 1e-12)
                for i in range(n)
            ])
            k = max(1, want // hop)
            csum = np.concatenate([[0.0], np.cumsum(rms)])
            scores = csum[k:] - csum[:-k]
            # Varianz-Strafe: gleichmäßige Lautheit bevorzugen
            var = np.array([rms[i:i + k].std() for i in range(len(scores))])
            scores = scores / k - 0.5 * var
            b = int(scores.argmax())
            a = b * hop
            best = wf[:, a: a + want].clone()
            pos = a / sr

        n_fade = int(sr * fade_ms / 1000)
        best = _fade(best, n_fade)

        peak = best.abs().max().item()
        if peak > 0:
            target = 10 ** (normalize_db / 20.0)
            best = best * (target / peak)

        out_sr = sr
        if resample_to and int(resample_to) != sr:
            best = torchaudio.functional.resample(best, sr, int(resample_to))
            out_sr = int(resample_to)

        info = (f"Referenz: {best.shape[1]/out_sr:.2f}s ab {pos:.2f}s | "
                f"{out_sr} Hz | {best.shape[0]} Kanal(e) | Peak {normalize_db:.1f} dBFS")
        return ({"waveform": best.unsqueeze(0), "sample_rate": out_sr}, info)


# --------------------------------------------------------------------------
# Node 3: Mehrere Spuren speichern
# --------------------------------------------------------------------------

class SaveSpeakerAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speaker_1": ("AUDIO", {
                    "tooltip": "Erste Spur - wird als ..._spk1_ gespeichert."
                }),
                "filename_prefix": ("STRING", {
                    "default": "speakers/voice",
                    "tooltip": "Pfad und Namensbasis relativ zu ComfyUI/output/.\n"
                               "Ein '/' erzeugt einen Unterordner, z. B. "
                               "'speakers/voice' -> output/speakers/voice_spk1_00001_.wav.\n"
                               "Der laufende Zähler wird automatisch angehängt, "
                               "nichts wird überschrieben."
                }),
                "format": (["wav", "flac"], {
                    "default": "wav",
                    "tooltip": "wav: unkomprimiert, wird von allen Nodes und Tools "
                               "gelesen - Standard für Referenzdateien.\n"
                               "flac: verlustfrei komprimiert, ca. halb so groß, aber "
                               "nicht überall als Eingabe unterstützt."
                }),
            },
            "optional": {
                "speaker_2": ("AUDIO", {
                    "tooltip": "Optional - wird als ..._spk2_ gespeichert. "
                               "Leere Spuren werden übersprungen."
                }),
                "speaker_3": ("AUDIO", {
                    "tooltip": "Optional - wird als ..._spk3_ gespeichert. "
                               "Leere Spuren werden übersprungen."
                }),
            },
        }

    RETURN_TYPES = ()
    DESCRIPTION = ("Speichert bis zu drei Sprecher-Spuren in einem Durchlauf nach "
                   "ComfyUI/output/ und zeigt sie im Node als Player an.")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio/speaker"

    def save(self, speaker_1, filename_prefix, format, speaker_2=None, speaker_3=None):
        if folder_paths is None:
            raise RuntimeError("folder_paths nicht verfügbar (nur innerhalb von ComfyUI nutzbar).")

        out_dir = folder_paths.get_output_directory()
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir
        )
        results = []
        for i, a in enumerate([speaker_1, speaker_2, speaker_3], start=1):
            if a is None:
                continue
            wf = a["waveform"]
            if wf.dim() == 3:
                wf = wf[0]
            if wf.shape[-1] < 2:
                continue
            name = f"{filename}_spk{i}_{counter:05}_.{format}"
            path = os.path.join(full_dir, name)
            torchaudio.save(path, wf.cpu(), int(a["sample_rate"]),
                            format=("wav" if format == "wav" else "flac"))
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})
            print(f"[SpeakerSplit] gespeichert: {path}")

        return {"ui": {"audio": results}}


# --------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SpeakerSplitDiarize": SpeakerSplitDiarize,
    "VoiceReferenceExtract": VoiceReferenceExtract,
    "SaveSpeakerAudio": SaveSpeakerAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerSplitDiarize": "Speaker Split (Diarization)",
    "VoiceReferenceExtract": "Voice Reference Extract",
    "SaveSpeakerAudio": "Save Speaker Audio",
}
