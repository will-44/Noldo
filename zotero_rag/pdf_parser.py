from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core.schema import Document as LlamaDocument

from .utils import get_logger
from .zotero_client import ZoteroItem

logger = get_logger(__name__)


class PDFParser:
    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, pdf_path: Path, item: ZoteroItem) -> list[LlamaDocument]:
        logger.info(f"Parsing PDF: {pdf_path.name}")
        try:
            result = self.converter.convert(str(pdf_path))
        except Exception as e:
            logger.error(f"Docling failed on {pdf_path}: {e}")
            return []

        doc = result.document
        source_str = self._source_str(item)

        # Group text elements by page, tracking current section
        pages: dict[int, dict] = {}  # page_no → {texts: [], section: str}
        current_section = ""

        for element, _level in doc.iterate_items():
            label = getattr(element, "label", None)
            text = getattr(element, "text", None)
            if not text or not text.strip():
                continue

            # Track section headings
            if label and "section_header" in str(label).lower():
                current_section = text.strip()

            # Skip headings themselves from body text (very short)
            if len(text.strip()) < 40:
                continue

            page_no = 1
            prov = getattr(element, "prov", None)
            if prov:
                page_no = prov[0].page_no if hasattr(prov[0], "page_no") else 1

            if page_no not in pages:
                pages[page_no] = {"texts": [], "section": current_section}
            pages[page_no]["texts"].append(text.strip())
            # Update section for this page only when we see a new one
            if current_section:
                pages[page_no]["section"] = current_section

        documents: list[LlamaDocument] = []
        for page_no, page_data in sorted(pages.items()):
            combined = "\n\n".join(page_data["texts"])
            if len(combined.strip()) < 80:
                continue

            metadata = {
                "item_key": item.item_key,
                "title": item.title,
                "authors": ", ".join(item.authors) if item.authors else "Unknown",
                "year": item.year,
                "doi": item.doi or "",
                "page_number": page_no,
                "section": page_data["section"],
                "pdf_path": str(pdf_path.absolute()),
                "source": source_str,
            }
            documents.append(LlamaDocument(text=combined, metadata=metadata))

        logger.info(f"Parsed {len(documents)} pages from {pdf_path.name}")
        return documents

    @staticmethod
    def _source_str(item: ZoteroItem) -> str:
        authors_short = ", ".join(item.authors[:2]) if item.authors else "Unknown"
        if len(item.authors) > 2:
            authors_short += " et al."
        year = item.year or "n.d."
        return f"{authors_short} ({year}) — {item.title}"
