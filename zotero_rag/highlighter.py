from pathlib import Path

import fitz  # PyMuPDF

from .retriever import SourceChunk
from .utils import get_logger

logger = get_logger(__name__)

HIGHLIGHT_COLOR = (1, 1, 0)   # Yellow
HIGHLIGHT_OPACITY = 0.4
FRAGMENT_WORDS = 30
FALLBACK_WORDS = 5


class PDFHighlighter:
    def highlight_sources(
        self,
        sources: list[SourceChunk],
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group chunks by source PDF
        by_pdf: dict[str, list[SourceChunk]] = {}
        for src in sources:
            if src.pdf_path:
                by_pdf.setdefault(src.pdf_path, []).append(src)

        result: dict[str, Path] = {}
        for pdf_path_str, chunks in by_pdf.items():
            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            item_key = chunks[0].item_key
            out_path = output_dir / f"{item_key}_highlighted.pdf"

            try:
                result[item_key] = self._process_pdf(pdf_path, chunks, out_path)
            except Exception as e:
                logger.error(f"Failed to highlight {pdf_path}: {e}")

        return result

    # ------------------------------------------------------------------

    def _process_pdf(
        self,
        pdf_path: Path,
        chunks: list[SourceChunk],
        out_path: Path,
    ) -> Path:
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)

        for chunk in chunks:
            page_idx = chunk.page_number - 1
            if page_idx < 0 or page_idx >= n_pages:
                logger.warning(f"Page {chunk.page_number} out of range for {pdf_path.name}")
                continue

            page = doc[page_idx]
            self._highlight_chunk(page, chunk)

        doc.save(str(out_path), garbage=4, deflate=True)
        doc.close()
        logger.info(f"Highlighted PDF saved: {out_path.name}")
        return out_path

    @staticmethod
    def _highlight_chunk(page: fitz.Page, chunk: SourceChunk) -> None:
        words = chunk.text.split()
        found_any = False

        for i in range(0, len(words), FRAGMENT_WORDS):
            fragment = " ".join(words[i : i + FRAGMENT_WORDS])
            rects = page.search_for(fragment, quads=False)

            if not rects:
                # Fallback: first N words of the fragment
                short = " ".join(words[i : i + FALLBACK_WORDS])
                rects = page.search_for(short, quads=False)
                if not rects:
                    logger.debug(f"Fragment not found: '{fragment[:40]}…'")
                    continue

            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=HIGHLIGHT_COLOR)
                annot.set_opacity(HIGHLIGHT_OPACITY)
                annot.update()
                found_any = True

        # Score annotation in margin
        if found_any:
            margin_pt = fitz.Point(page.rect.width - 5, 20 + (chunk.page_number % 5) * 15)
            page.add_text_annot(
                margin_pt,
                f"Score: {chunk.score:.3f}\n{chunk.section[:40] if chunk.section else ''}",
                icon="Note",
            )
