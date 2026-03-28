"""Evaluate MERT embedding quality across different chunk durations.

Creates separate ChromaDB collections per chunk duration, runs artist-based
similarity queries, and reports Precision@k / MRR metrics for comparison.

Usage:
    uv run python evaluate_chunk_duration.py
    uv run python evaluate_chunk_duration.py --durations 5 10 20 --max-songs 50
    uv run python evaluate_chunk_duration.py --skip-build --ground-truth ground_truth.csv
    uv run python evaluate_chunk_duration.py --keep
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from core.config import reload_settings
from core.exceptions import EmbeddingError
from services.chroma_service import ChromaService
from services.mert_service import MERTService
from services.audio_service import AudioService
from utils import find_audio_files, sanitize_filename


DEFAULT_DURATIONS = [5, 10, 15, 20, 30]
QUERY_DIR = Path("inference_music_files")
MUSIC_DIR = Path("music_files")
DEFAULT_N_RESULTS = 20


def parse_artist(filename: str) -> str:
    """Extract artist from filename like 'Artist - Title.mp3'."""
    stem = Path(filename).stem
    if " - " in stem:
        return stem.split(" - ", 1)[0].strip()
    return stem.strip()


def setup_chunk_config(chunk_duration: int, chunk_overlap: int) -> MERTService:
    """Set env vars, reload settings, return fresh MERTService."""
    os.environ["MUSICDB_CHUNK_DURATION"] = str(chunk_duration)
    os.environ["MUSICDB_CHUNK_OVERLAP"] = str(chunk_overlap)
    reload_settings()
    return MERTService()


def build_collection(
    chunk_duration: int,
    collection_name: str,
    music_dir: Path,
    max_songs: int | None,
) -> bool:
    """Embed all songs into a collection for a given chunk duration.

    Returns True on success, False on OOM or other failure.
    """
    chunk_overlap = max(1, int(chunk_duration * 0.2))
    logger.info(
        f"Building collection '{collection_name}' "
        f"(chunk={chunk_duration}s, overlap={chunk_overlap}s)"
    )

    try:
        embedder = setup_chunk_config(chunk_duration, chunk_overlap)
    except Exception as e:
        logger.error(f"Failed to initialize services for chunk={chunk_duration}s: {e}")
        return False

    db = ChromaService(collection_name=collection_name)
    audio_svc = AudioService()

    if db.count() > 0:
        logger.info(
            f"Collection '{collection_name}' already has {db.count()} songs, skipping build"
        )
        return True

    files = find_audio_files(music_dir)
    if max_songs:
        files = files[:max_songs]

    if not files:
        logger.warning(f"No audio files found in {music_dir}")
        return False

    logger.info(f"Embedding {len(files)} files with chunk_duration={chunk_duration}s")

    failed = 0
    for i, file_path in enumerate(files, 1):
        song_id = sanitize_filename(file_path.stem)
        if i % 10 == 0 or i == len(files):
            logger.info(f"  [{i}/{len(files)}] {file_path.name}")

        try:
            embedding = embedder.embed_file(file_path)
        except EmbeddingError as e:
            if "out of memory" in str(e).lower() or (
                torch.cuda.is_available()
                and torch.cuda.oom_error is not None
                and "out of memory" in str(torch.cuda.oom_error).lower()
            ):
                logger.error(
                    f"OOM at chunk={chunk_duration}s (file {i}/{len(files)}), "
                    f"aborting this duration"
                )
                cleanup_collection(collection_name)
                return False
            logger.warning(f"  Failed to embed {file_path.name}: {e}")
            failed += 1
            continue
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"CUDA OOM at chunk={chunk_duration}s (file {i}/{len(files)}), "
                f"aborting this duration"
            )
            torch.cuda.empty_cache()
            cleanup_collection(collection_name)
            return False
        except Exception as e:
            logger.warning(f"  Failed to embed {file_path.name}: {e}")
            failed += 1
            continue

        duration = audio_svc.get_duration(file_path)
        metadata = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "duration_seconds": duration,
            "artist": parse_artist(file_path.name),
        }
        db.add_song(song_id, embedding.tolist(), metadata)

    count = db.count()
    logger.info(
        f"Collection '{collection_name}': {count} songs added ({failed} failed)"
    )

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return count > 0


def cleanup_collection(collection_name: str) -> None:
    """Delete a ChromaDB collection."""
    try:
        client = ChromaService(collection_name=collection_name)._get_client()
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
    except Exception:
        pass


def evaluate_artist_similarity(
    collection_name: str,
    n_results: int = DEFAULT_N_RESULTS,
) -> dict:
    """Run leave-one-out artist-based evaluation using stored embeddings.

    For each song in the collection that shares an artist with at least one
    other song, use its stored embedding as a query (excluding itself) and
    measure whether same-artist songs rank highly.

    No MERT inference needed — uses embeddings already in ChromaDB.

    Returns dict with precision/mrr metrics.
    """
    logger.info(f"Evaluating artist similarity for '{collection_name}'")

    db = ChromaService(collection_name=collection_name)
    collection = db._get_collection()

    all_data = collection.get(include=["embeddings", "metadatas"])
    song_ids = all_data["ids"]
    song_embs = all_data["embeddings"]
    song_meta = all_data["metadatas"]

    artist_to_songs: dict[str, list[tuple[int, str]]] = {}
    for idx, (sid, meta) in enumerate(zip(song_ids, song_meta)):
        artist = meta.get("artist", parse_artist(meta.get("filename", sid)))
        artist_to_songs.setdefault(artist.lower(), []).append((idx, sid))

    multi_song_artists = {
        a: songs for a, songs in artist_to_songs.items() if len(songs) >= 2
    }

    if not multi_song_artists:
        logger.warning("No artists with >= 2 songs found, cannot evaluate")
        return {
            "collection": collection_name,
            "num_queries": 0,
            "num_artists": 0,
            "error": "no multi-song artists",
        }

    eval_songs: list[tuple[int, str, str]] = []
    for artist, songs in multi_song_artists.items():
        for idx, sid in songs:
            eval_songs.append((idx, sid, artist))

    logger.info(
        f"Leave-one-out eval: {len(eval_songs)} songs from "
        f"{len(multi_song_artists)} artists (>= 2 songs each)"
    )

    all_precisions = {1: [], 3: [], 5: [], 10: []}
    reciprocal_ranks = []

    for query_idx, query_id, query_artist in eval_songs:
        query_emb = song_embs[query_idx]

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_results + 1,
        )

        filtered_ranks = []
        for rank_i, (rid, meta) in enumerate(
            zip(results["ids"][0], results["metadatas"][0]), 1
        ):
            if rid == query_id:
                continue
            effective_rank = rank_i - 1
            result_artist = meta.get("artist", parse_artist(meta.get("filename", rid)))
            if result_artist.lower() == query_artist:
                filtered_ranks.append(effective_rank)

        for k in all_precisions:
            hits = sum(1 for r in filtered_ranks if r <= k)
            all_precisions[k].append(hits / min(k, n_results))

        if filtered_ranks:
            reciprocal_ranks.append(1.0 / min(filtered_ranks))
        else:
            reciprocal_ranks.append(0.0)

    metrics: dict = {
        "collection": collection_name,
        "num_queries": len(eval_songs),
        "num_artists": len(multi_song_artists),
    }
    for k in sorted(all_precisions):
        vals = all_precisions[k]
        metrics[f"precision@{k}"] = round(np.mean(vals), 4) if vals else 0.0
    metrics["mrr"] = round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0

    return metrics


def evaluate_ground_truth(
    collection_name: str,
    ground_truth_path: Path,
    n_results: int = DEFAULT_N_RESULTS,
) -> dict | None:
    """Evaluate using ground truth CSV of known similar/dissimilar pairs.

    CSV format: query_file,similar_file (one pair per row).

    Returns dict with recall@k and MRR, or None if file missing.
    """
    if not ground_truth_path.exists():
        logger.info(f"No ground truth file at {ground_truth_path}, skipping")
        return None

    logger.info(f"Evaluating ground truth for '{collection_name}'")

    pairs: list[tuple[str, str]] = []
    with open(ground_truth_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["query_file"], row["similar_file"]))

    if not pairs:
        logger.warning("Ground truth CSV is empty")
        return None

    chunk_duration = collection_name.split("_")[-1].replace("s", "")
    chunk_overlap = max(1, int(int(chunk_duration) * 0.2))
    try:
        embedder = setup_chunk_config(int(chunk_duration), chunk_overlap)
    except Exception as e:
        logger.error(f"Failed to init MERTService for ground truth eval: {e}")
        return {"collection": collection_name, "error": str(e)}

    db = ChromaService(collection_name=collection_name)

    reciprocal_ranks = []
    recalls = {1: [], 3: [], 5: [], 10: []}

    for query_fname, target_fname in pairs:
        query_path = find_query_file(query_fname)
        target_id = sanitize_filename(Path(target_fname).stem)
        if query_path is None:
            logger.warning(f"Query file not found: {query_fname}")
            continue

        try:
            query_emb = embedder.embed_file(query_path)
        except Exception as e:
            logger.warning(f"Failed to embed query {query_fname}: {e}")
            continue

        results = db.search_similar(query_emb.tolist(), n_results=n_results)
        result_ids = [r["id"] for r in results]

        for rank, rid in enumerate(result_ids, 1):
            if rid == target_id:
                reciprocal_ranks.append(1.0 / rank)
                for k in recalls:
                    if rank <= k:
                        recalls[k].append(1.0)
                    else:
                        recalls[k].append(0.0)
                break
        else:
            reciprocal_ranks.append(0.0)
            for k in recalls:
                recalls[k].append(0.0)

    metrics: dict = {
        "collection": collection_name,
        "num_pairs": len(pairs),
    }
    for k in sorted(recalls):
        vals = recalls[k]
        metrics[f"recall@{k}"] = round(np.mean(vals), 4) if vals else 0.0
    metrics["mrr"] = round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def find_query_file(filename: str) -> Path | None:
    """Find a query file by name in QUERY_DIR."""
    for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]:
        p = QUERY_DIR / (Path(filename).stem + ext)
        if p.exists():
            return p
    return None


def print_report(
    artist_results: list[dict],
    gt_results: list[dict | None],
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("CHUNK DURATION EVALUATION RESULTS")
    print("=" * 80)

    if artist_results:
        print("\n--- Artist-Based Similarity ---\n")
        header = f"{'Duration':<12}"
        metric_keys = [k for k in artist_results[0] if k.startswith("precision")]
        metric_keys.append("mrr")
        for k in metric_keys:
            header += f"{k.upper():<14}"
        print(header)
        print("-" * len(header))

        for r in artist_results:
            if "error" in r:
                print(f"{r['collection']:<12} ERROR: {r['error']}")
                continue
            dur = r["collection"].replace("eval_chunk_", "").replace("s", "s")
            line = f"{dur:<12}"
            for k in metric_keys:
                val = r.get(k, "N/A")
                line += f"{str(val):<14}"
            print(line)

    if any(r is not None for r in gt_results):
        print("\n--- Ground Truth Pairs ---\n")
        header = f"{'Duration':<12}"
        gt_valid = [r for r in gt_results if r is not None]
        if gt_valid:
            metric_keys = [k for k in gt_valid[0] if k.startswith("recall")]
            metric_keys.append("mrr")
            for k in metric_keys:
                header += f"{k.upper():<14}"
            print(header)
            print("-" * len(header))

            for r in gt_valid:
                if r is None or "error" in r:
                    continue
                dur = r["collection"].replace("eval_chunk_", "").replace("s", "s")
                line = f"{dur:<12}"
                for k in metric_keys:
                    val = r.get(k, "N/A")
                    line += f"{str(val):<14}"
                print(line)

    print("\n" + "=" * 80)


def export_csv(
    output_path: Path,
    artist_results: list[dict],
    gt_results: list[dict | None],
) -> None:
    """Export results to CSV."""
    rows = []

    for r in artist_results:
        if "error" in r:
            continue
        row = {"evaluation_type": "artist", **r}
        rows.append(row)

    for r in gt_results:
        if r is None or "error" in r:
            continue
        row = {"evaluation_type": "ground_truth", **r}
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results exported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MERT embedding quality across chunk durations"
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=DEFAULT_DURATIONS,
        help="Chunk durations to evaluate (default: 5 10 15 20 30)",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Limit number of songs to embed (default: all)",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=MUSIC_DIR,
        help=f"Directory with songs to index (default: {MUSIC_DIR})",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=QUERY_DIR,
        help=f"Directory with query files (default: {QUERY_DIR})",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth CSV (query_file,similar_file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to export results CSV",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep eval collections after running (default: delete)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building collections (use existing ones)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=DEFAULT_N_RESULTS,
        help=f"Number of search results per query (default: {DEFAULT_N_RESULTS})",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    total_start = time.time()

    collections_created: list[str] = []

    try:
        if not args.skip_build:
            logger.info(f"Building collections for durations: {args.durations}")
            for dur in args.durations:
                coll_name = f"eval_chunk_{dur}s"
                ok = build_collection(dur, coll_name, args.music_dir, args.max_songs)
                if ok:
                    collections_created.append(coll_name)
                else:
                    logger.warning(
                        f"Skipping evaluation for chunk_duration={dur}s "
                        f"(build failed or OOM)"
                    )
        else:
            client = ChromaService()._get_client()
            existing = client.list_collections()
            for dur in args.durations:
                coll_name = f"eval_chunk_{dur}s"
                if any(c.name == coll_name for c in existing):
                    collections_created.append(coll_name)
                    logger.info(f"Using existing collection '{coll_name}'")
                else:
                    logger.warning(f"Collection '{coll_name}' not found, skipping")

        if not collections_created:
            logger.error("No collections available for evaluation")
            return

        artist_results: list[dict] = []
        gt_results: list[dict | None] = []

        for coll_name in collections_created:
            ar = evaluate_artist_similarity(coll_name, args.n_results)
            artist_results.append(ar)

            if args.ground_truth:
                gr = evaluate_ground_truth(coll_name, args.ground_truth, args.n_results)
                gt_results.append(gr)

        print_report(artist_results, gt_results)

        if args.output:
            export_csv(args.output, artist_results, gt_results)

        elapsed = time.time() - total_start
        logger.info(f"Total evaluation time: {elapsed:.1f}s")

    finally:
        if not args.keep:
            logger.info("Cleaning up evaluation collections...")
            for coll_name in collections_created:
                cleanup_collection(coll_name)


if __name__ == "__main__":
    main()
