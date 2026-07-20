import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from pyzotero import zotero

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class ZoteroItem:
    item_key: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    tags: list[str]
    pdf_attachment_key: str
    date_modified: str = ""


class ZoteroClient:
    def __init__(self, config: dict):
        zc = config["zotero"]
        self.zot = zotero.Zotero(
            zc["library_id"],
            zc["library_type"],
            zc["web_api_key"],
        )
        self.collection_key = zc.get("collection_key", "")
        self.local_files_dir = Path(zc.get("local_files_dir", ""))
        self.pdf_cache_dir = Path(config["rag"]["pdf_cache_dir"])
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        self._library_version: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_items_with_pdfs(self) -> list[ZoteroItem]:
        logger.info("Fetching all PDF attachments from Zotero…")
        attachments = self._fetch_all(
            self.zot.items(itemType="attachment")
        )
        pdf_atts = [
            a for a in attachments
            if a["data"].get("contentType") == "application/pdf"
            and a["data"].get("linkMode") in ("imported_file", "imported_url")
            and a["data"].get("parentItem")
        ]
        logger.info(f"Found {len(pdf_atts)} PDF attachments")

        parent_keys = list({a["data"]["parentItem"] for a in pdf_atts})
        parents = self._fetch_parents_by_key(parent_keys)

        items: list[ZoteroItem] = []
        for att in pdf_atts:
            parent_key = att["data"]["parentItem"]
            parent = parents.get(parent_key)
            if not parent:
                continue
            item = self._build_item(parent, att["key"])
            if item:
                items.append(item)

        self._library_version = self._read_library_version()
        logger.info(f"Returning {len(items)} items with PDFs (library v{self._library_version})")
        return items

    def download_pdf(self, item: ZoteroItem) -> Path:
        cache_path = self.pdf_cache_dir / f"{item.item_key}.pdf"
        if cache_path.exists():
            return cache_path

        # Try local WebDAV zip first
        if self.local_files_dir.exists():
            zip_path = self.local_files_dir / f"{item.pdf_attachment_key}.zip"
            if zip_path.exists():
                extracted = self._extract_pdf_from_zip(zip_path, cache_path)
                if extracted:
                    logger.info(f"Extracted PDF from local zip: {item.item_key}")
                    return cache_path

        # Fallback: download via Zotero API
        logger.info(f"Downloading PDF via API: {item.item_key}")
        try:
            self.zot.dump(item.pdf_attachment_key, str(cache_path))
            return cache_path
        except Exception as e:
            raise FileNotFoundError(
                f"Could not get PDF for {item.item_key}: {e}"
            ) from e

    def get_new_or_modified_items(self, since_version: int) -> list[ZoteroItem]:
        logger.info(f"Fetching items modified since version {since_version}…")
        try:
            changed = self._fetch_all(
                self.zot.items(itemType="attachment", since=since_version)
            )
        except Exception:
            # Fallback if 'since' not supported
            changed = self._fetch_all(self.zot.items(itemType="attachment"))

        pdf_atts = [
            a for a in changed
            if a["data"].get("contentType") == "application/pdf"
            and a["data"].get("linkMode") in ("imported_file", "imported_url")
            and a["data"].get("parentItem")
        ]
        if not pdf_atts:
            return []

        parent_keys = list({a["data"]["parentItem"] for a in pdf_atts})
        parents = self._fetch_parents_by_key(parent_keys)

        items = []
        for att in pdf_atts:
            parent = parents.get(att["data"]["parentItem"])
            if parent:
                item = self._build_item(parent, att["key"])
                if item:
                    items.append(item)

        self._library_version = self._read_library_version()
        return items

    def get_library_version(self) -> int:
        return self._library_version

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_library_version(self) -> int:
        """pyzotero expose last_modified_version() (méthode) qui renvoie la version
        de la dernière requête. On la lit de façon défensive."""
        try:
            lmv = self.zot.last_modified_version
            value = lmv() if callable(lmv) else lmv
            return int(value or 0)
        except Exception:
            return 0

    def _fetch_all(self, generator) -> list:
        return self.zot.everything(generator)

    def _fetch_parents_by_key(self, keys: list[str]) -> dict[str, dict]:
        parents: dict[str, dict] = {}
        batch = 50
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            try:
                results = self.zot.items(itemKey=",".join(chunk))
                for item in results:
                    parents[item["key"]] = item
            except Exception as e:
                logger.warning(f"Failed to fetch parent batch: {e}")
        return parents

    def _build_item(self, parent: dict, att_key: str) -> ZoteroItem | None:
        data = parent.get("data", {})
        item_type = data.get("itemType", "")
        if item_type in ("attachment", "note"):
            return None

        title = data.get("title", "Untitled")
        creators = data.get("creators", [])
        authors = [
            f"{c.get('lastName', '')} {c.get('firstName', '')}".strip()
            or c.get("name", "")
            for c in creators
            if c.get("creatorType") in ("author", "editor")
        ]

        date_str = data.get("date", "")
        year = None
        for part in date_str.replace("-", " ").replace("/", " ").split():
            if len(part) == 4 and part.isdigit():
                year = int(part)
                break

        doi = data.get("DOI") or data.get("doi") or None
        tags = [t.get("tag", "") for t in data.get("tags", [])]
        date_modified = parent.get("meta", {}).get("parsedDate", data.get("dateModified", ""))

        return ZoteroItem(
            item_key=parent["key"],
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            tags=tags,
            pdf_attachment_key=att_key,
            date_modified=date_modified,
        )

    @staticmethod
    def _extract_pdf_from_zip(zip_path: Path, dest: Path) -> bool:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
                if not pdf_names:
                    return False
                with zf.open(pdf_names[0]) as src, open(dest, "wb") as out:
                    out.write(src.read())
            return True
        except Exception as e:
            logger.warning(f"Failed to extract zip {zip_path}: {e}")
            return False
