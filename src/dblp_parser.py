from lxml import etree
from typing import Iterator, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DBLPStreamParser:

    PUBLICATION_TYPES = {
        "article",
        "inproceedings",
        "proceedings",
        "book",
        "incollection",
        "phdthesis",
        "mastersthesis",
        "www"
    }

    DBLP_XML_URL = "https://dblp.org/xml/dblp.xml.gz"
    DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"

    def __init__(self, xml_file, chunk_size=10000, pub_types=None):
        self.xml_file = xml_file
        self.chunk_size = chunk_size
        # Record types to yield. Note that "www" records are author homepages
        # whose <author> tags list name aliases, not co-authors.
        self.pub_types = set(pub_types) if pub_types else set(self.PUBLICATION_TYPES)

    @staticmethod
    def download_dblp(xml_path: str = "data/dblp.xml"):
        """
        Download and decompress the DBLP XML dump plus its DTD.

        The DTD must sit next to the XML so that named entities (e.g. &uuml;)
        resolve during parsing.
        """
        import gzip
        import shutil
        import urllib.request
        from pathlib import Path

        xml_path = Path(xml_path)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        gz_path = xml_path.with_name(xml_path.name + ".gz")
        dtd_path = xml_path.with_name("dblp.dtd")

        for url, dest in ((DBLPStreamParser.DBLP_DTD_URL, dtd_path),
                          (DBLPStreamParser.DBLP_XML_URL, gz_path)):
            logger.info(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)

        logger.info(f"Decompressing {gz_path} -> {xml_path}")
        with gzip.open(gz_path, "rb") as src, open(xml_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

        return str(xml_path)

    def parse(self) -> Iterator[Dict]:

        from lxml import etree

        # Use iterparse flags to load the DTD and resolve named entities
        # (e.g. &uuml;). Some lxml versions accept these flags directly.
        try:
            context = etree.iterparse(
                self.xml_file,
                events=("end",),
                tag=tuple(self.pub_types),   # <-- IMPORTANT
                load_dtd=True,
                resolve_entities=True,
                huge_tree=True,
            )
        except TypeError:
            # Older lxml builds may not accept the flags above and raise
            # a TypeError. Fall back to calling iterparse without flags
            # but enable a recover retry if a syntax error occurs.
            try:
                context = etree.iterparse(
                    self.xml_file,
                    events=("end",),
                    tag=tuple(self.pub_types),
                )
            except etree.XMLSyntaxError:
                logger.warning("Initial parse failed; retrying with recover=True")
                context = etree.iterparse(
                    self.xml_file,
                    events=("end",),
                    tag=tuple(self.pub_types),
                    recover=True,
                )
        except etree.XMLSyntaxError:
            # If parse fails due to entity or minor syntax problems,
            # retry with recover enabled so parsing continues.
            logger.warning("Initial parse failed; retrying with recover=True")
            context = etree.iterparse(
                self.xml_file,
                events=("end",),
                tag=tuple(self.pub_types),
                load_dtd=True,
                resolve_entities=True,
                huge_tree=True,
                recover=True,
            )

        count = 0

        for _, elem in context:

            # Skip comments, processing instructions, etc.
            if not isinstance(elem.tag, str):
                elem.clear()
                continue

            tag = elem.tag

            if tag not in self.pub_types:
                elem.clear()
                continue

            paper = self._extract_paper(elem, tag)

            if paper["authors"]:
                yield paper

                count += 1

                if count % self.chunk_size == 0:
                    logger.info(f"Parsed {count:,} papers")

            elem.clear()

            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]

        logger.info(f"Finished parsing {count:,} papers")

    # def _extract_paper(self, elem, tag):

    #     print("Publication:", tag)

    #     for child in elem:
    #         print(
    #             "TAG =", repr(child.tag),
    #             "TEXT =", repr(child.text)
    #         )

    #     raise SystemExit

    def _extract_paper(self, elem, tag):

        paper = {
            "key": elem.get("key", ""),
            "title": "",
            "authors": [],
            "year": None,
            "venue": "",
            "doi": "",
            # Key of the proceedings volume, e.g. conf/icde/2006w for workshops
            "crossref": "",
            "pub_type": tag,
            # "informal" marks preprints (e.g. CoRR) and similar records
            "publtype": elem.get("publtype", ""),
        }

        for child in elem:

            if not isinstance(child.tag, str):
                continue

            ctag = child.tag

            text = child.text.strip() if child.text else ""

            if ctag == "author":
                paper["authors"].append(text)

            elif ctag == "title":
                # Titles may contain markup such as <i>k</i>-means
                paper["title"] = "".join(child.itertext()).strip()

            elif ctag == "crossref":
                paper["crossref"] = text

            elif ctag == "ee" and not paper["doi"]:
                for prefix in ("https://doi.org/", "http://doi.org/",
                               "https://dx.doi.org/", "http://dx.doi.org/"):
                    if text.lower().startswith(prefix):
                        paper["doi"] = text[len(prefix):].lower()
                        break

            elif ctag == "year":
                try:
                    paper["year"] = int(text)
                except ValueError:
                    pass

            elif ctag in ("journal", "booktitle"):
                paper["venue"] = text

        return paper

from collections import defaultdict
from typing import Dict, List, Tuple


class AuthorPaperIndex:
    """
    Indexes authors and papers for fast collaboration graph construction.
    """

    def __init__(self):

        # Paper storage
        self.papers: List[Dict] = []

        # Author mappings
        self.author_name_to_id: Dict[str, int] = {}
        self.author_id_to_name: Dict[int, str] = {}

        # author_id -> list of paper indices
        self.author_to_papers = defaultdict(list)

        self.next_author_id = 0

    def add_paper(self, paper: Dict) -> int:
        """
        Add one paper to the index.

        Returns
        -------
        int
            Index of the paper.
        """

        paper_idx = len(self.papers)
        self.papers.append(paper)

        for author in paper["authors"]:

            if author not in self.author_name_to_id:

                aid = self.next_author_id

                self.author_name_to_id[author] = aid
                self.author_id_to_name[aid] = author

                self.next_author_id += 1

            aid = self.author_name_to_id[author]

            self.author_to_papers[aid].append(paper_idx)

        return paper_idx

    def get_author_id(self, author_name: str):

        return self.author_name_to_id.get(author_name)

    def get_author_name(self, author_id: int):

        return self.author_id_to_name.get(author_id)

    def get_paper(self, paper_idx: int):

        return self.papers[paper_idx]

    def get_author_papers(self, author_id: int):

        return self.author_to_papers.get(author_id, [])

    def get_author_coauthors(self, author_id: int):
        """
        Returns

        [
            (coauthor_id, [shared_paper_indices]),
            ...
        ]
        """

        coauthors = defaultdict(list)

        author_name = self.author_id_to_name[author_id]

        for pid in self.author_to_papers[author_id]:

            paper = self.papers[pid]

            for coauthor in paper["authors"]:

                if coauthor == author_name:
                    continue

                cid = self.author_name_to_id[coauthor]

                coauthors[cid].append(pid)

        return list(coauthors.items())

    def num_authors(self):

        return len(self.author_name_to_id)

    def num_papers(self):

        return len(self.papers)

    def stats(self):

        return {
            "num_papers": len(self.papers),
            "num_authors": len(self.author_name_to_id),
            "author_paper_links": sum(
                len(v) for v in self.author_to_papers.values()
            )
        }