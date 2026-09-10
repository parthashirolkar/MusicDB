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
from services.audio_service import AudioService
from services.feature_service import FeatureService
from services.reranker_service import AudioFeatureReranker
from plot_similarity import generate_spectrogram_comparison
from utils import format_duration
from core.config import get_settings


_mert_service: MERTService | None = None
_chroma_service: ChromaService | None = None


def get_mert_service() -> MERTService:
    global _mert_service
    if _mert_service is None:
        _mert_service = MERTService()
    return _mert_service


def get_chroma_service() -> ChromaService:
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaService()
    return _chroma_service


def format_results_for_display(results: list[dict]) -> list[list]:
    rows = []
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        title = metadata.get("title", filename)
        duration = metadata.get("duration_seconds")
        similarity = result.get("reranked_score", 1 - result.get("distance", 1.0))
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

        settings = get_settings()
        overfetch = settings.reranker.overfetch if settings.reranker.enabled else 0
        results = db.search_similar(
            query_embedding.tolist(), n_results=top_k + 1 + overfetch
        )

        if settings.reranker.enabled and results:
            audio_svc = AudioService()
            feature_svc = FeatureService()
            reranker = AudioFeatureReranker()

            query_y = audio_svc.load_audio(query_path)
            query_features = feature_svc.extract_features(
                query_y, audio_svc.sample_rate
            )
            results = reranker.rerank(query_features, results, top_n=top_k + 1)
            status_log.append("Reranked results using audio features")
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
        similarity = result.get("reranked_score", 1 - result["distance"])

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


LOGIC_PRO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap');

:root {
    --lp-bg-deep: #0d0d12;
    --lp-bg: #14141c;
    --lp-bg-surface: #1c1c28;
    --lp-bg-elevated: #242434;
    --lp-bg-hover: #2a2a3e;
    --lp-border: #2e2e42;
    --lp-border-subtle: #222236;
    --lp-text-primary: #e8e8f0;
    --lp-text-secondary: #8888a0;
    --lp-text-muted: #5c5c74;
    --lp-accent: #7b61ff;
    --lp-accent-hover: #9078ff;
    --lp-accent-glow: rgba(123, 97, 255, 0.25);
    --lp-accent-secondary: #5eafff;
    --lp-green: #34d399;
    --lp-amber: #fbbf24;
    --lp-red: #f87171;
}

.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    background: var(--lp-bg-deep) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
}

/* ── Top Bar ────────────────────────────────────── */
.lp-topbar {
    background: linear-gradient(180deg, #1a1a28 0%, #14141e 100%);
    border-bottom: 1px solid var(--lp-border);
    padding: 14px 24px !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -12px -12px 0 -12px !important;
}
.lp-topbar h1 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--lp-text-primary) !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
    line-height: 1 !important;
}
.lp-topbar .lp-subtitle {
    font-size: 0.7rem !important;
    color: var(--lp-text-muted) !important;
    margin: 2px 0 0 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 500 !important;
}
.lp-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--lp-bg-elevated);
    border: 1px solid var(--lp-border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.7rem !important;
    color: var(--lp-text-secondary) !important;
    letter-spacing: 0.04em;
}
.lp-badge .lp-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--lp-green);
    box-shadow: 0 0 6px var(--lp-green);
    animation: lp-pulse 2s ease-in-out infinite;
}
@keyframes lp-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Panels ─────────────────────────────────────── */
.lp-panel {
    background: var(--lp-bg) !important;
    border: 1px solid var(--lp-border-subtle) !important;
    border-radius: 10px !important;
    padding: 0 !important;
    overflow: hidden;
}
.lp-panel-header {
    background: var(--lp-bg-surface) !important;
    border-bottom: 1px solid var(--lp-border-subtle) !important;
    padding: 10px 16px !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0 !important;
}
.lp-panel-header .lp-icon {
    width: 8px;
    height: 8px;
    border-radius: 2px;
}
.lp-panel-header h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--lp-text-secondary) !important;
    margin: 0 !important;
}
.lp-panel-body {
    padding: 16px !important;
}

/* ── Input Bar ──────────────────────────────────── */
.lp-input-row {
    display: flex !important;
    gap: 8px !important;
    align-items: stretch !important;
}
.lp-input-row .lp-url-wrap {
    flex: 1 !important;
}

/* ── Transport Bar ──────────────────────────────── */
.lp-transport {
    background: linear-gradient(180deg, #18182a 0%, #12121c 100%);
    border: 1px solid var(--lp-border-subtle);
    border-radius: 8px;
    padding: 10px 16px !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.lp-transport-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--lp-text-muted) !important;
    white-space: nowrap;
}
.lp-transport-status {
    flex: 1;
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--lp-text-secondary) !important;
}

/* ── Spectrogram Area ───────────────────────────── */
.lp-spectro-info {
    font-size: 0.7rem !important;
    color: var(--lp-text-muted) !important;
    padding: 0 2px !important;
    margin: 0 !important;
}

/* ── Footer ─────────────────────────────────────── */
.lp-footer {
    border-top: 1px solid var(--lp-border-subtle) !important;
    padding: 10px 0 0 0 !important;
    margin-top: 12px !important;
}
.lp-footer-text {
    font-size: 0.7rem !important;
    color: var(--lp-text-muted) !important;
    margin: 0 !important;
}
.lp-footer-text code {
    background: var(--lp-bg-elevated) !important;
    color: var(--lp-accent-secondary) !important;
    padding: 1px 6px !important;
    border-radius: 3px !important;
    font-size: 0.65rem !important;
}

/* ── Global Component Overrides ─────────────────── */

/* Text input */
input[type="text"], textarea {
    background: var(--lp-bg-surface) !important;
    border: 1px solid var(--lp-border) !important;
    border-radius: 6px !important;
    color: var(--lp-text-primary) !important;
    font-size: 0.85rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--lp-accent) !important;
    box-shadow: 0 0 0 2px var(--lp-accent-glow) !important;
    outline: none !important;
}
input[type="text"]::placeholder {
    color: var(--lp-text-muted) !important;
}

/* Slider */
input[type="range"] {
    accent-color: var(--lp-accent) !important;
}

/* Primary button — override all Gradio internal layers */
.lp-btn-search,
.lp-btn-search *,
.lp-btn-search button,
.lp-btn-search .gr-button,
.lp-btn-search .svelte-1lz5yja,
.lp-btn-search [data-testid="base-button"] {
    background: linear-gradient(135deg, var(--lp-accent) 0%, #6366f1 100%) !important;
    background-color: transparent !important;
    background-image: linear-gradient(135deg, var(--lp-accent) 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px var(--lp-accent-glow) !important;
    min-width: 100px !important;
    height: auto !important;
    line-height: 1.4 !important;
    outline: none !important;
    text-shadow: none !important;
}
.lp-btn-search:hover,
.lp-btn-search:hover *,
.lp-btn-search:hover button,
.lp-btn-search:hover .svelte-1lz5yja {
    background-color: transparent !important;
    background-image: linear-gradient(135deg, var(--lp-accent-hover) 0%, #818cf8 100%) !important;
    box-shadow: 0 4px 16px var(--lp-accent-glow) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}
.lp-btn-search:active,
.lp-btn-search:active * {
    transform: translateY(0px) !important;
    color: #ffffff !important;
}

/* Dataframe table */
table {
    background: var(--lp-bg) !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}
thead tr {
    background: var(--lp-bg-surface) !important;
    border-bottom: 1px solid var(--lp-border) !important;
}
thead th {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--lp-text-muted) !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--lp-border) !important;
}
tbody td {
    font-size: 0.8rem !important;
    color: var(--lp-text-secondary) !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--lp-border-subtle) !important;
}
tbody tr:hover {
    background: var(--lp-bg-hover) !important;
}

/* Gallery */
.lp-gallery .grid-wrap {
    gap: 8px !important;
}
.lp-gallery img {
    border-radius: 6px !important;
    border: 1px solid var(--lp-border-subtle) !important;
}

/* Hide default Gradio labels for cleaner look */
.lp-hide-label label span, .lp-hide-label .svelte-1gfkn6j {
    display: none !important;
}

/* Section dividers */
.lp-divider {
    height: 1px;
    background: var(--lp-border-subtle);
    margin: 8px 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--lp-bg) !important;
}
::-webkit-scrollbar-thumb {
    background: var(--lp-bg-elevated) !important;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--lp-text-muted) !important;
}

/* Row/Column spacing override */
.lp-compact > .flex {
    gap: 0 !important;
}
"""


def build_ui():
    with gr.Blocks(title="MusicDB") as demo:
        # ── Top Bar ──────────────────────────────────
        gr.HTML(
            """
            <div class="lp-topbar">
                <div>
                    <h1>&#9654; MusicDB</h1>
                    <p class="lp-subtitle">Audio Similarity Engine</p>
                </div>
                <div class="lp-badge">
                    <span class="lp-dot"></span>
                    <span>MERT-v1-95M &middot; ChromaDB</span>
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            # ── Left Panel: Search + Results ──────────
            with gr.Column(scale=2, min_width=380):
                # Input panel
                gr.HTML(
                    """
                    <div class="lp-panel">
                        <div class="lp-panel-header">
                            <div class="lp-icon" style="background: var(--lp-accent);"></div>
                            <h3>Query Input</h3>
                        </div>
                    """
                )
                with gr.Column(elem_classes=["lp-panel-body"]):
                    youtube_url = gr.Textbox(
                        placeholder="Paste a YouTube URL to search...",
                        lines=1,
                        show_label=False,
                        elem_classes=["lp-hide-label"],
                        container=False,
                    )
                    with gr.Row(elem_classes=["lp-input-row"]):
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Top K",
                            show_label=True,
                            container=False,
                            scale=3,
                        )
                        search_btn = gr.Button(
                            "\u25b6 Search",
                            variant="primary",
                            elem_classes=["lp-btn-search"],
                            scale=1,
                        )
                gr.HTML("</div>")

                # Results panel
                gr.HTML(
                    """
                    <div class="lp-panel" style="margin-top: 10px;">
                        <div class="lp-panel-header">
                            <div class="lp-icon" style="background: var(--lp-green);"></div>
                            <h3>Results</h3>
                        </div>
                    """
                )
                with gr.Column(elem_classes=["lp-panel-body"]):
                    results_table = gr.Dataframe(
                        headers=["#", "Title", "Similarity", "Duration"],
                        column_widths=["8%", "52%", "22%", "18%"],
                        show_label=False,
                        show_search="none",
                        wrap=True,
                    )
                gr.HTML("</div>")

                # Transport bar (status)
                gr.HTML(
                    """
                    <div class="lp-transport" style="margin-top: 10px;">
                        <span class="lp-transport-label">&#9679; Status</span>
                    """
                )
                status_output = gr.Textbox(
                    lines=1,
                    show_label=False,
                    interactive=False,
                    placeholder="Ready — enter a YouTube URL to begin",
                    elem_classes=["lp-transport-status", "lp-hide-label"],
                    container=False,
                )
                gr.HTML("</div>")

            # ── Right Panel: Spectrograms ─────────────
            with gr.Column(scale=3, min_width=520):
                gr.HTML(
                    """
                    <div class="lp-panel" style="height: 100%;">
                        <div class="lp-panel-header">
                            <div class="lp-icon" style="background: var(--lp-amber);"></div>
                            <h3>Spectrogram Analysis</h3>
                        </div>
                    """
                )
                with gr.Column(elem_classes=["lp-panel-body"]):
                    gr.HTML(
                        '<p class="lp-spectro-info">'
                        "Top row: query audio (30–60s) &middot; "
                        "Bottom row: matched song &middot; "
                        "Mel spectrogram comparison"
                        "</p>"
                    )
                    gallery_output = gr.Gallery(
                        label="Spectrograms",
                        columns=2,
                        object_fit="contain",
                        height="100%",
                        show_label=False,
                        elem_classes=["lp-gallery"],
                    )
                gr.HTML("</div>")

        # ── Footer ───────────────────────────────────
        gr.HTML(
            """
            <div class="lp-footer">
                <p class="lp-footer-text">
                    Audio files cache to <code>inference_music_files/</code> after first download
                    &nbsp;&middot;&nbsp;
                    Embeddings via MERT-v1-95M (768-dim)
                    &nbsp;&middot;&nbsp;
                    Vector search via ChromaDB HNSW cosine
                </p>
            </div>
            """
        )

        # ── Events ───────────────────────────────────
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
        head="""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <meta name="theme-color" content="#0d0d12">
        """,
        css=LOGIC_PRO_CSS,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    build_ui()
