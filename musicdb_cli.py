"""MusicDB CLI with modern patterns."""

import click
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.config import get_settings
from services.chroma_service import ChromaService
from services.feature_service import FeatureService
from music_pipeline import MusicPipeline
from utils import format_duration


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose: bool):
    """MusicDB - Music Embedding and Search System"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.group()
def download():
    """Download music from YouTube"""
    pass


@download.command()
@click.argument("url")
@click.option("--output-dir", "-o", default="music_files", help="Output directory")
@click.option("--cleanup", is_flag=True, help="Delete files after processing")
@click.option("--collection", "-c", help="ChromaDB collection name")
@click.pass_context
def video(ctx, url: str, output_dir: str, cleanup: bool, collection: str | None):
    """Download a single YouTube video"""

    async def run():
        pipeline = MusicPipeline(
            collection_name=collection, output_dir=output_dir, cleanup=cleanup
        )
        result = await pipeline.process_video(url)

        if result["status"] == "success":
            click.echo(f"✅ Added: {result['title']}")
        elif result["status"] == "skipped":
            click.echo(f"⏭️ Skipped: {result.get('reason', 'duplicate')}")
        else:
            click.echo(f"❌ Failed: {result.get('reason', 'unknown error')}")

    asyncio.run(run())


@download.command()
@click.argument("url")
@click.option("--output-dir", "-o", default="music_files", help="Output directory")
@click.option("--cleanup", is_flag=True, help="Delete files after processing")
@click.option("--collection", "-c", help="ChromaDB collection name")
@click.pass_context
def playlist(ctx, url: str, output_dir: str, cleanup: bool, collection: str | None):
    """Download all videos from a YouTube playlist"""

    async def run():
        pipeline = MusicPipeline(
            collection_name=collection, output_dir=output_dir, cleanup=cleanup
        )
        results = await pipeline.process_playlist(url)

        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = len(results) - success - skipped

        click.echo("\n📊 Results:")
        click.echo(f"   ✅ Success: {success}")
        click.echo(f"   ⏭️ Skipped: {skipped}")
        click.echo(f"   ❌ Failed: {failed}")

    asyncio.run(run())


@cli.group()
def db():
    """Database operations"""
    pass


@db.command()
@click.option("--collection", "-c", help="Collection name")
def stats(collection: str | None):
    """Show database statistics"""
    db = ChromaService(collection_name=collection)
    count = db.count()

    click.echo("\n📀 Database Statistics:")
    click.echo(f"   Collection: {db.collection_name}")
    click.echo(f"   Total songs: {count}")


@db.command(name="list")
@click.option("--limit", "-n", default=20, help="Number of songs to list")
@click.option("--collection", "-c", help="Collection name")
def list_songs(limit: int, collection: str | None):
    """List songs in the database"""
    db = ChromaService(collection_name=collection)

    click.echo("\n🎵 Songs in database:\n")

    songs = list(db.list_songs(limit=limit))
    if not songs:
        click.echo("   (empty)")
        return

    for i, song in enumerate(songs, 1):
        meta = song.get("metadata", {})
        title = meta.get("title", song["id"])
        duration = meta.get("duration_seconds")
        click.echo(f"{i:3d}. {title} ({format_duration(duration)})")


@db.command(name="backfill-features")
@click.option("--collection", "-c", help="Collection name")
def backfill_features(collection: str | None):
    """Extract and store audio features for all songs missing them"""
    db = ChromaService(collection_name=collection)
    feature_svc = FeatureService()

    songs = list(db.list_songs(limit=99999))
    if not songs:
        click.echo("No songs in database.")
        return

    updated = 0
    skipped = 0

    for song in songs:
        meta = song.get("metadata", {})
        filepath = meta.get("filepath")

        if not filepath:
            logger.warning(
                f"Skipping {song['id']}: no filepath (file may have been cleaned up)"
            )
            skipped += 1
            continue

        if "tempo" in meta:
            logger.debug(f"Skipping {song['id']}: already has features")
            skipped += 1
            continue

        try:
            fp = Path(filepath)
            if not fp.exists():
                logger.warning(f"Skipping {song['id']}: file not found at {filepath}")
                skipped += 1
                continue

            features = feature_svc.extract_features_from_file(fp)
            meta.update(features)
            db.update_metadata(song["id"], meta)
            updated += 1
            click.echo(f"  Updated: {song['id']}")
        except Exception as e:
            logger.error(f"Failed to process {song['id']}: {e}")
            skipped += 1

    click.echo(f"\nBackfill complete: {updated} updated, {skipped} skipped")


@cli.group()
def search():
    """Search for similar songs"""
    pass


@search.command()
@click.argument("query_path")
@click.option("--n-results", "-n", default=5, help="Number of results")
@click.option("--collection", "-c", help="Collection name")
def audio(query_path: str, n_results: int, collection: str | None):
    """Search using an audio file"""
    from query_database import search_similar_songs, print_results

    results = search_similar_songs(query_path, n_results, collection)
    print_results(results)


@cli.group()
def process():
    """Process local audio files"""
    pass


@process.command()
@click.argument("directory")
@click.option("--collection", "-c", help="Collection name")
@click.option("--validate/--no-validate", default=True, help="Validate audio files")
def directory(directory: str, collection: str | None, validate: bool):
    """Process all audio files in a directory"""
    from add_music_embeddings import process_music_directory

    process_music_directory(directory, collection)


@cli.command()
def info():
    """Show system information"""
    import torch
    import platform

    settings = get_settings()

    click.echo("\nℹ️  System Information:")
    click.echo(f"   Python: {platform.python_version()}")
    click.echo(f"   Platform: {platform.system()} {platform.release()}")
    click.echo(f"   PyTorch: {torch.__version__}")
    click.echo(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        click.echo(f"   GPU: {torch.cuda.get_device_name(0)}")

    click.echo(f"\n   MERT Model: {settings.mert.model_name}")
    click.echo(f"   Sample rate: {settings.mert.sample_rate}Hz")
    click.echo(f"   Embedding dim: {settings.mert.embedding_dim}")
    click.echo(f"   Chunk duration: {settings.processing.chunk_duration}s")


if __name__ == "__main__":
    cli()
