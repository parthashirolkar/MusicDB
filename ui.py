"""Gradio UI for interactive music similarity search."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
import gradio as gr
from PIL import Image

from services.youtube_service import YouTubeService
from services.mert_service import MERTService
from services.chroma_service import ChromaService
from plot_similarity import generate_spectrogram_comparison
from utils import format_duration


_mert_service: MERTService | None = None
_chroma_service: ChromaService | None = None


def get_mert_service() -> MERTService:
    """Get cached MERT service instance."""
    global _mert_service
    if _mert_service is None:
        _mert_service = MERTService()
    return _mert_service


def get_chroma_service() -> ChromaService:
    """Get cached Chroma service instance."""
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaService()
    return _chroma_service


def format_results_for_display(results: list[dict]) -> list[list]:
    """Format search results for display."""
    rows = []
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        title = metadata.get("title", filename)
        duration = metadata.get("duration_seconds")
        distance = result.get("distance", 1.0)
        similarity = 1 - distance

        rows.append(
            [
                i,
                title,
                f"{similarity:.4f}",
                format_duration(duration) if duration else "N/A",
            ]
        )
    return rows


async def run_search(youtube_url: str, top_k: int):
    """Main search pipeline: download (if cache miss), search, and generate visualization."""
    cache_dir = Path("inference_music_files")
    cache_dir.mkdir(parents=True, exist_ok=True)

    status_log = []

    youtube_service = YouTubeService(output_dir=str(cache_dir))

    try:
        info = youtube_service.get_info(youtube_url)
        video_id = info["video_id"]
        video_title = info["title"]
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None, None, f"Error: {e}"

    cached_path = youtube_service.find_cached_file(video_id, video_title)
    if cached_path:
        status_log.append(f"Using cached: {cached_path.name}")
        query_path = cached_path
    else:
        status_log.append(f"Downloading: {video_title}")
        try:
            result = await youtube_service.download_async(youtube_url)
            query_path = result["file_path"]
            status_log.append(f"Downloaded: {result['title']}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None, None, f"Error: {e}"

    status_log.append(f"Searching (top_k={top_k})...")

    query_path = Path(query_path)
    if not query_path.exists():
        return None, None, f"Query file not found: {query_path}"

    try:
        embedder = get_mert_service()
        db = get_chroma_service()

        logger.info(f"Processing query: {video_title}")
        query_embedding = embedder.embed_file(query_path)
        results = db.search_similar(query_embedding.tolist(), n_results=top_k + 1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None, None, f"Search failed: {e}"

    if not results:
        return None, None, "No results. Ensure the database has songs."

    status_log.append(f"Found {len(results)} results")
    results_rows = format_results_for_display(results)

    plot_images = []
    plot_captions = []

    for i, result in enumerate(results):
        match_metadata = result.get("metadata", {})
        match_path = match_metadata.get("filepath")
        match_title = match_metadata.get(
            "title", match_metadata.get("filename", "Unknown")
        )
        similarity = 1 - result["distance"]

        if match_path:
            try:
                fig = generate_spectrogram_comparison(
                    query_path=query_path,
                    match_path=match_path,
                    query_title=video_title,
                    match_title=match_title,
                    similarity=similarity,
                )
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                pil_img = Image.open(buf).convert("RGB")
                plot_images.append(pil_img)
                plot_captions.append(f"#{i + 1}: {match_title} (sim: {similarity:.3f})")
            except Exception as e:
                logger.error(f"Plot generation failed for {match_title}: {e}")

    gallery_data = list(zip(plot_images, plot_captions)) if plot_images else None

    status_log.append(f"Generated {len(plot_images)} spectrograms")
    return results_rows, gallery_data, "\n".join(status_log)


def build_ui():
    """Build and launch the Gradio interface."""
    with gr.Blocks(title="MusicDB") as demo:
        with gr.Row(equal_height=False):
            gr.Markdown(
                "<div style='padding: 8px 0;'>"
                "<h1 style='margin: 0; font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em;'>"
                "MusicDB</h1>"
                "<p style='margin: 4px 0 0 0; color: #666; font-size: 0.9rem;'>"
                "Similarity Search</p>"
                "</div>"
            )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Input")

                youtube_url = gr.Textbox(
                    placeholder="YouTube URL",
                    lines=1,
                    show_label=False,
                )

                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Results",
                        show_label=True,
                        container=True,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=0)

                gr.Markdown("### Results")

                results_table = gr.Dataframe(
                    headers=["Rank", "Title", "Similarity", "Duration"],
                    column_widths=["10%", "50%", "22%", "18%"],
                    show_label=False,
                    wrap=True,
                )

                gr.Markdown("### Status")
                status_output = gr.Textbox(
                    lines=2,
                    show_label=False,
                    interactive=False,
                    placeholder="Ready...",
                )

            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### Spectrograms")
                gr.Markdown(
                    "<span style='color: #888; font-size: 0.8rem;'>"
                    "Top: query (30-60s). Bottom: each retrieved song."
                    "</span>"
                )
                gallery_output = gr.Gallery(
                    label="Spectrogram Comparisons",
                    columns=2,
                    object_fit="contain",
                    height="100%",
                )

        gr.Markdown(
            "<div style='margin-top: 16px; padding-top: 12px; border-top: 1px solid #eee; "
            "color: #666; font-size: 0.8rem;'>"
            "Files cache to <code>inference_music_files/</code> after first download."
            "</div>"
        )

        search_btn.click(
            fn=run_search,
            inputs=[youtube_url, top_k],
            outputs=[results_table, gallery_output, status_output],
        )
        youtube_url.submit(
            fn=run_search,
            inputs=[youtube_url, top_k],
            outputs=[results_table, gallery_output, status_output],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container {max-width: 1400px !important; margin: auto !important;}
        .gr-row {gap: 1.5rem !important;}
        .gr-column {gap: 1rem !important;}
        h1 {font-weight: 600 !important;}
        .gr-markdown p {margin: 0 !important;}
        .gr-dataframe {font-size: 0.85rem !important;}
        .gr-textbox {font-size: 0.85rem !important;}
        .gr-slider {font-size: 0.85rem !important;}
        .gr-gallery {min-height: 600px !important;}
        """,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    build_ui()
