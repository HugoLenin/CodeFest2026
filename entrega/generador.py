#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import faiss
import pysbd
import torch
from sentence_transformers import SentenceTransformer

ENCODER_NAME = "intfloat/multilingual-e5-base"
ENCODER_MAX_SEQ_LEN = 512
QUERY_PREFIX = "query: "

TOP_DOCUMENTS = 3
TOP_FRAGMENTS = 10
MAX_WORDS_PER_FRAGMENT = 250
RETRIEVE_TOP_K = 100


GRAPH_TOP_K = 100            # candidatos aportados por el grafo antes de fusionar
RRF_K0 = 60                  # constante de suavizado k0 de la ec. 7 del spec (Sección 8.4)
GRAPH_NEIGHBOR_DISCOUNT = 0.5  # peso de vecinos de primer orden vs. entidad consultada
NER_BACKEND = "transformers"
NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"  # AFL-3.0, 10 idiomas incl. es/en/pt
NER_BATCH_SIZE = 32
NER_SCORE_MIN = 0.60         # umbral de confianza para aceptar una entidad
SPACY_MODELS = {             # tokenización/gazetteer para el linking de consultas
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
}

# pysbd no soporta portugués nativamente; se usa español como respaldo para
# segmentar oraciones en texto en portugués (mismas convenciones de
# puntuación de fin de oración). Ver rag/chunk.py para el detalle.
_PYSBD_SUPPORTED = {"es", "en"}
_PYSBD_LANG_FALLBACK = {"pt": "es"}
_segmenters: dict[str, pysbd.Segmenter] = {}


def get_segmenter(lang: str) -> pysbd.Segmenter:
    pysbd_lang = lang if lang in _PYSBD_SUPPORTED else _PYSBD_LANG_FALLBACK.get(lang, "es")
    if pysbd_lang not in _segmenters:
        _segmenters[pysbd_lang] = pysbd.Segmenter(language=pysbd_lang, clean=False)
    return _segmenters[pysbd_lang]


def split_sentences(text: str, lang: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        sents = get_segmenter(lang).segment(text)
    except Exception:
        sents = [text]
    return [s.strip() for s in sents if s.strip()]


def split_into_word_limited_subfragments(text: str, lang: str, max_words: int) -> list[str]:
    sentences = split_sentences(text, lang) or [text]
    subfrags: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush():
        nonlocal current, current_words
        if current:
            subfrags.append(" ".join(current))
        current, current_words = [], 0

    for sent in sentences:
        w = len(sent.split())
        if current and current_words + w > max_words:
            flush()
        if w > max_words and not current:
            # Oración única que excede el límite (p.ej. preámbulos legales
            # sin punto interno). El límite de 250 palabras es estricto y
            # con penalización automática (Sección 9.3.2), así que se
            # trunca por palabras en vez de conservarla completa.
            subfrags.append(" ".join(sent.split()[:max_words]))
            continue
        current.append(sent)
        current_words += w
    flush()
    return subfrags




_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_LEADING_DET = {"the", "el", "la", "los", "las", "un", "una", "o", "a", "os", "as", "um", "uma"}


def normalize_surface(text: str) -> str:
    """Forma normalizada de una mención: minúsculas, sin tildes, sin
    puntuación, espacios colapsados. Debe coincidir exactamente con la
    normalización usada al construir el grafo (rag/graph.py) para que las
    menciones de la consulta liguen con los alias de los nodos."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT_RE.sub(" ", text.casefold())
    text = _WS_RE.sub(" ", text).strip()
    parts = text.split(" ")
    if len(parts) > 1 and parts[0] in _LEADING_DET:
        parts = parts[1:]
    return " ".join(parts)


# Gazetteer de alias translingües y conceptos del dominio (idéntico al usado
# en la construcción del grafo, rag/graph.py) para que el linking de
# consultas reconozca las mismas superficies.
ALIAS_GAZETTEER: dict[str, tuple[str, list[str]]] = {
    "Estados Unidos": ("LOC", ["United States", "United States of America", "EE.UU.", "EEUU", "EE. UU.", "USA", "U.S.", "US", "Estados Unidos da América", "America"]),
    "China": ("LOC", ["People's Republic of China", "República Popular China", "PRC"]),
    "Rusia": ("LOC", ["Russia", "Russian Federation", "Federación Rusa", "Rússia"]),
    "Colombia": ("LOC", ["Republic of Colombia", "República de Colombia"]),
    "Brasil": ("LOC", ["Brazil", "República Federativa do Brasil"]),
    "México": ("LOC", ["Mexico", "Estados Unidos Mexicanos"]),
    "Argentina": ("LOC", []),
    "Chile": ("LOC", []),
    "Perú": ("LOC", ["Peru"]),
    "Venezuela": ("LOC", []),
    "Ecuador": ("LOC", []),
    "Bolivia": ("LOC", []),
    "Unión Europea": ("ORG", ["European Union", "União Europeia", "UE", "EU"]),
    "Reino Unido": ("LOC", ["United Kingdom", "UK", "Great Britain", "Gran Bretaña"]),
    "India": ("LOC", []),
    "Japón": ("LOC", ["Japan", "Japão"]),
    "Corea del Sur": ("LOC", ["South Korea", "República de Corea", "Coreia do Sul"]),
    "Corea del Norte": ("LOC", ["North Korea", "Coreia do Norte"]),
    "Ucrania": ("LOC", ["Ukraine", "Ucrânia"]),
    "Israel": ("LOC", []),
    "Irán": ("LOC", ["Iran"]),
    "Francia": ("LOC", ["France", "França"]),
    "Alemania": ("LOC", ["Germany", "Alemanha"]),
    "América Latina": ("LOC", ["Latin America", "América Latina y el Caribe", "Latin America and the Caribbean", "LATAM", "América Latina e Caribe"]),
    "Caribe": ("LOC", ["Caribbean"]),
    "Amazonía": ("LOC", ["Amazon", "Amazonia", "Amazon rainforest", "Amazônia"]),
    "Naciones Unidas": ("ORG", ["United Nations", "ONU", "UN", "Nações Unidas"]),
    "OTAN": ("ORG", ["NATO", "North Atlantic Treaty Organization", "Organización del Tratado del Atlántico Norte"]),
    "NASA": ("ORG", ["National Aeronautics and Space Administration"]),
    "ESA": ("ORG", ["European Space Agency", "Agencia Espacial Europea"]),
    "UNOOSA": ("ORG", ["United Nations Office for Outer Space Affairs", "Oficina de Asuntos del Espacio Ultraterrestre"]),
    "SpaceX": ("ORG", ["Space Exploration Technologies"]),
    "OCDE": ("ORG", ["OECD", "Organisation for Economic Co-operation and Development", "Organización para la Cooperación y el Desarrollo Económicos"]),
    "Banco Mundial": ("ORG", ["World Bank", "Banco Mundial"]),
    "FMI": ("ORG", ["IMF", "International Monetary Fund", "Fondo Monetario Internacional"]),
    "BID": ("ORG", ["IDB", "Inter-American Development Bank", "Banco Interamericano de Desarrollo"]),
    "CEPAL": ("ORG", ["ECLAC", "Comisión Económica para América Latina y el Caribe"]),
    "OEA": ("ORG", ["OAS", "Organization of American States", "Organización de los Estados Americanos"]),
    "Comisión Europea": ("ORG", ["European Commission", "Comissão Europeia"]),
    "Fuerza Aeroespacial Colombiana": ("ORG", ["FAC", "Fuerza Aérea Colombiana", "Colombian Aerospace Force"]),
    "Ministerio de Defensa": ("ORG", ["Ministry of Defense", "Ministry of Defence", "Ministério da Defesa", "Ministerio de Defensa Nacional"]),
    "DARPA": ("ORG", ["Defense Advanced Research Projects Agency"]),
    "Pentágono": ("ORG", ["Pentagon", "Department of Defense", "Departamento de Defensa", "DoD"]),
    "Universidad de Stanford": ("ORG", ["Stanford University", "Stanford"]),
    "MIT": ("ORG", ["Massachusetts Institute of Technology"]),
    "Google": ("ORG", ["Alphabet"]),
    "Microsoft": ("ORG", []),
    "OpenAI": ("ORG", []),
    "IBM": ("ORG", []),
    "Nvidia": ("ORG", ["NVIDIA"]),
    "Amazon": ("ORG", ["Amazon Web Services", "AWS"]),
    "Meta": ("ORG", ["Facebook"]),
    "inteligencia artificial": ("CONCEPTO", ["artificial intelligence", "AI", "IA", "inteligência artificial"]),
    "aprendizaje automático": ("CONCEPTO", ["machine learning", "ML", "aprendizaje de máquina", "aprendizado de máquina"]),
    "aprendizaje profundo": ("CONCEPTO", ["deep learning", "redes neuronales profundas"]),
    "sistemas de armas autónomos": ("CONCEPTO", ["autonomous weapons", "autonomous weapon systems", "armas autónomas", "LAWS", "lethal autonomous weapons"]),
    "drones": ("CONCEPTO", ["drone", "UAV", "unmanned aerial vehicle", "vehículos aéreos no tripulados", "UAS"]),
    "ciberseguridad": ("CONCEPTO", ["cybersecurity", "cyber security", "cibersegurança", "seguridad cibernética"]),
    "computación cuántica": ("CONCEPTO", ["quantum computing", "computação quântica"]),
    "basura espacial": ("CONCEPTO", ["space debris", "orbital debris", "desechos espaciales", "detritos espaciais", "chatarra espacial", "space junk"]),
    "órbita terrestre baja": ("CONCEPTO", ["low earth orbit", "LEO", "órbita baja", "órbita terrestre baixa"]),
    "megaconstelaciones": ("CONCEPTO", ["mega-constellations", "megaconstellations", "satellite constellations", "constelaciones de satélites"]),
    "sostenibilidad espacial": ("CONCEPTO", ["space sustainability", "sustentabilidade espacial"]),
    "seguridad espacial": ("CONCEPTO", ["space security", "space safety", "segurança espacial"]),
    "satélites": ("CONCEPTO", ["satellite", "satélite", "satellites"]),
    "migración": ("CONCEPTO", ["migration", "migração", "flujos migratorios"]),
    "deforestación": ("CONCEPTO", ["deforestation", "desmatamento"]),
    "narcotráfico": ("CONCEPTO", ["drug trafficking", "tráfico de drogas", "narcotráfico"]),
    "crimen organizado": ("CONCEPTO", ["organized crime", "organised crime", "crime organizado"]),
    "cambio climático": ("CONCEPTO", ["climate change", "mudança climática", "mudanças climáticas"]),
    "derechos humanos": ("CONCEPTO", ["human rights", "direitos humanos"]),
    "desigualdad": ("CONCEPTO", ["inequality", "desigualdade"]),
    "gobernanza": ("CONCEPTO", ["governance", "governança"]),
    "grupos armados": ("CONCEPTO", ["armed groups", "grupos armados ilegales", "grupos armados organizados"]),
    "minería ilegal": ("CONCEPTO", ["illegal mining", "mineração ilegal"]),
    "seguridad humana": ("CONCEPTO", ["human security", "segurança humana"]),
}


LABEL_MAP = {
    "PER": "PER", "PERSON": "PER",
    "ORG": "ORG",
    "LOC": "LOC", "GPE": "LOC", "FAC": "LOC",
    "MISC": "MISC", "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": None, "LAW": "MISC", "LANGUAGE": None,
    "DATE": None, "TIME": None, "PERCENT": None, "MONEY": None,
    "QUANTITY": None, "ORDINAL": None, "CARDINAL": None,
}


class EntityExtractor:
    """NER multilingüe (transformer HuggingFace) + gazetteer de conceptos,
    igual que en la construcción del grafo (rag/graph.py, misma clase). Se
    reutiliza tal cual para que la Sección 8.5 ("mismo componente NER
    utilizado durante la construcción del grafo") se cumpla literalmente.
    """

    def __init__(self, backend: str | None = None, model_name: str | None = None):
        self.backend = backend or NER_BACKEND
        self.model_name = model_name or NER_MODEL
        self._hf_pipe = None
        self._nlps: dict[str, "spacy.language.Language"] = {}
        self._matchers: dict[str, "spacy.matcher.PhraseMatcher"] = {}

    def nlp_for(self, lang: str):
        import spacy
        lang = lang if lang in SPACY_MODELS else "en"
        if lang not in self._nlps:
            nlp = spacy.load(SPACY_MODELS[lang])
            self._nlps[lang] = nlp
            self._matchers[lang] = self._build_matcher(nlp)
        return self._nlps[lang]

    def _build_matcher(self, nlp):
        from spacy.matcher import PhraseMatcher
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        for canonical, (tipo, aliases) in ALIAS_GAZETTEER.items():
            patterns = [nlp.make_doc(t) for t in [canonical, *aliases]]
            matcher.add(f"{tipo}||{canonical}", patterns)
        return matcher

    def _hf_pipeline(self):
        if self._hf_pipe is None:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._hf_pipe = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=device,
            )
        return self._hf_pipe

    def hf_entities_batch(self, texts: list[str]) -> list[list[dict]]:
        pipe = self._hf_pipeline()
        raw = pipe(texts, batch_size=NER_BATCH_SIZE)
        if texts and isinstance(raw, list) and raw and isinstance(raw[0], dict):
            raw = [raw]
        out = []
        for ents in raw:
            spans = []
            for e in ents:
                tipo = LABEL_MAP.get(e["entity_group"])
                if tipo and e["score"] >= NER_SCORE_MIN:
                    spans.append({"start": int(e["start"]), "end": int(e["end"]), "tipo": tipo})
            out.append(spans)
        return out

    def entities_in_doc(self, doc, lang: str, hf_spans: list[dict] | None = None):
        spans = []
        if self.backend == "transformers" and hf_spans is not None:
            for s in hf_spans:
                sp = doc.char_span(s["start"], s["end"], alignment_mode="expand")
                if sp is not None:
                    spans.append((sp, s["tipo"]))
        else:
            for ent in doc.ents:
                tipo = LABEL_MAP.get(ent.label_)
                if tipo:
                    spans.append((ent, tipo))
        matcher = self._matchers[lang if lang in self._matchers else "en"]
        for match_id, start, end in matcher(doc):
            tipo, canonical = doc.vocab.strings[match_id].split("||", 1)
            spans.append((doc[start:end], f"GAZ||{tipo}||{canonical}"))
        spans = [(sp, t) for sp, t in spans if len(sp.text.strip()) >= 2
                 and not sp.text.strip().isdigit()]
        spans.sort(key=lambda st: (not st[1].startswith("GAZ||"), -(st[0].end - st[0].start)))
        chosen, taken = [], set()
        for sp, t in spans:
            rng = set(range(sp.start, sp.end))
            if rng & taken:
                continue
            taken |= rng
            chosen.append((sp, t))
        return chosen


def rrf_fuse(rankings: list[list[int]], k0: int | None = None) -> list[tuple[int, float]]:

    k0 = RRF_K0 if k0 is None else k0
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k0 + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class GraphRetriever:


    REL_BONUS = 1.0  

    def __init__(self, G, extractor: "EntityExtractor | None" = None,
                 chunkid_to_faissid: dict[str, int] | None = None):
        self.G = G
        self.extractor = extractor or EntityExtractor()
  
        self.chunkid_to_faissid = chunkid_to_faissid
        self.alias_index: dict[str, str] = {}
        for nid, d in G.nodes(data=True):
            for alias in [d.get("etiqueta", ""), *d.get("aliases", "").split("|")]:
                norm = normalize_surface(alias)
                if len(norm) >= 3:
                    self.alias_index.setdefault(norm, nid)
        self._node_chunks: dict[str, list[int]] = {
            nid: [int(x) for x in d.get("chunks", "").split(",") if x]
            for nid, d in G.nodes(data=True)
        }

    @staticmethod
    def _guess_lang(text: str) -> str:
        """Heurística mínima por palabras funcionales (las consultas del
        reto son cortas; no amerita un detector estadístico)."""
        toks = set(normalize_surface(text).split())
        en = len(toks & {"the", "of", "in", "and", "what", "which", "how", "is", "are", "for"})
        es = len(toks & {"el", "la", "los", "las", "de", "del", "en", "que", "cual", "cuales", "como", "es", "son", "para", "y"})
        return "en" if en > es else "es"

    def link_query(self, query_text: str) -> set[str]:
        """Entidades de la consulta -> nodos del grafo (Sección 8.5, paso 1)."""
        linked: set[str] = set()
        lang = self._guess_lang(query_text)
        nlp = self.extractor.nlp_for(lang)
        doc = nlp(query_text)
        hf_spans = None
        if self.extractor.backend == "transformers":
            try:
                hf_spans = self.extractor.hf_entities_batch([query_text])[0]
            except Exception:
                hf_spans = None
        for sp, tipo in self.extractor.entities_in_doc(doc, lang, hf_spans):
            surface = tipo.split("||", 2)[2] if tipo.startswith("GAZ||") else sp.text
            nid = self.alias_index.get(normalize_surface(surface))
            if nid:
                linked.add(nid)
        # respaldo: n-gramas de la consulta contra el índice de alias, para
        # entidades que el NER no marque (p.ej. consultas muy cortas)
        tokens = normalize_surface(query_text).split()
        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                nid = self.alias_index.get(" ".join(tokens[i:i + n]))
                if nid:
                    linked.add(nid)
        return linked

    def search(self, query_text: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Lista ordenada [(faiss_id, score_grafo)] (Sección 8.5, pasos 2-3)."""
        top_k = top_k or GRAPH_TOP_K
        linked = self.link_query(query_text)
        if not linked:
            return []
        weights: dict[str, float] = {nid: 1.0 for nid in linked}
        for nid in linked:
            for _, nb, edata in self.G.out_edges(nid, data=True):
                w = GRAPH_NEIGHBOR_DISCOUNT * min(1.0, edata.get("peso", 1) / 5.0)
                weights[nb] = max(weights.get(nb, 0.0), w)
            for nb, _, edata in self.G.in_edges(nid, data=True):
                w = GRAPH_NEIGHBOR_DISCOUNT * min(1.0, edata.get("peso", 1) / 5.0)
                weights[nb] = max(weights.get(nb, 0.0), w)
        chunk_scores: dict[int, float] = defaultdict(float)
        for nid, w in weights.items():
            for fid in self._node_chunks.get(nid, []):
                chunk_scores[fid] += w
   
        if len(linked) > 1 and self.chunkid_to_faissid:
            for u in linked:
                for _, v, edata in self.G.out_edges(u, data=True):
                    if v not in linked:
                        continue
                    for cid in edata.get("evidencia", "").split(";"):
                        fid = self.chunkid_to_faissid.get(cid)
                        if fid is not None:
                            chunk_scores[fid] += self.REL_BONUS
        ranked = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


class RetrievalIndex:
    def __init__(self, index_path: Path, metadata_path: Path,
                 grafo_path: Path | None = None):
        self.index = faiss.read_index(str(index_path))
        self.metadata: list[dict] = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Desalineación índice/metadata: {self.index.ntotal} vectores "
                f"vs {len(self.metadata)} líneas de metadata."
            )
        self._by_doc_pos = {(m["doc_id"], m["posicion"]): m for m in self.metadata}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(ENCODER_NAME, device=device)
        self.model.max_seq_length = ENCODER_MAX_SEQ_LEN


        self.graph_retriever: GraphRetriever | None = None
        if grafo_path is not None:
            self.graph_retriever = self._load_graph_retriever(grafo_path)

    def _load_graph_retriever(self, grafo_path: Path) -> GraphRetriever | None:
        if not grafo_path.exists():
            print(f"[grafo] {grafo_path} no encontrado; recuperación solo con FAISS.",
                  file=sys.stderr)
            return None
        try:
            import networkx as nx
        except ImportError:
            print("[grafo] networkx no está instalado; recuperación solo con FAISS.",
                  file=sys.stderr)
            return None
        try:
            print(f"[grafo] Cargando {grafo_path} ...", file=sys.stderr)
            G = nx.read_graphml(str(grafo_path))
            chunkid_to_faissid = {m["chunk_id"]: i for i, m in enumerate(self.metadata)}
            retriever = GraphRetriever(G, extractor=EntityExtractor(),
                                       chunkid_to_faissid=chunkid_to_faissid)

            retriever.search("prueba de disponibilidad del grafo")
        except Exception as e:
            print(f"[grafo] No se pudo inicializar la fusión con el grafo ({e}); "
                  f"recuperación solo con FAISS.", file=sys.stderr)
            return None
        print(f"[grafo] {grafo_path.name} cargado: {G.number_of_nodes()} nodos, "
              f"{G.number_of_edges()} aristas. Fusión con FAISS vía RRF habilitada "
              f"(k0={RRF_K0}).", file=sys.stderr)
        return retriever

    def embed_query(self, text: str):
        return self.model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

    def _raw_search(self, query_text: str, k: int) -> list[dict]:
        q_emb = self.embed_query(query_text)
        scores, ids = self.index.search(q_emb, k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            hits.append({**self.metadata[idx], "score": float(score), "_faiss_id": int(idx)})
        return hits

    def _graph_ranking(self, query_text: str) -> list[int]:
        """IDs FAISS ordenados según el grafo de conocimiento (Sección 8.5,
        pasos 1-3). Lista vacía si no hay grafo cargado o si la consulta no
        liga ninguna entidad."""
        if self.graph_retriever is None:
            return []
        try:
            graph_hits = self.graph_retriever.search(query_text, top_k=GRAPH_TOP_K)
        except Exception as e:
            print(f"[grafo] búsqueda falló para esta consulta ({e}); se ignora "
                  f"el grafo solo para esta consulta.", file=sys.stderr)
            return []
        return [fid for fid, _ in graph_hits]

    def _fused_hits(self, dense_hits: list[dict], query_text: str) -> list[dict]:
  
        graph_ranking = self._graph_ranking(query_text)
        if not graph_ranking:
            return dense_hits
        dense_ranking = [h["_faiss_id"] for h in dense_hits]
        fused = rrf_fuse([dense_ranking, graph_ranking], k0=RRF_K0)
        return [{**self.metadata[fid], "score": score, "_faiss_id": fid}
                for fid, score in fused]

    @staticmethod
    def _top_documents(hits: list[dict], top_n: int) -> list[str]:
        best_per_doc: dict[str, float] = {}
        for h in hits:
            doc_id = h["doc_id"]
            if doc_id not in best_per_doc or h["score"] > best_per_doc[doc_id]:
                best_per_doc[doc_id] = h["score"]  # max pooling (Sección 8.6)
        ranked = sorted(best_per_doc.items(), key=lambda kv: kv[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_n]]

    def _top_fragments(self, hits: list[dict], top_n: int, max_words: int) -> list[dict]:
        fragments = []
        rank = 1
        for h in hits:
            if rank > top_n:
                break
            text = h["texto"]
            words = text.split()
            lang = h.get("idioma", "es")

            if len(words) > max_words:
                for sub in split_into_word_limited_subfragments(text, lang, max_words):
                    if rank > top_n:
                        break
                    fragments.append({"rank": rank, "chunk_id": h["chunk_id"],
                                       "doc_id": h["doc_id"], "text": sub})
                    rank += 1
                continue

            enriched = text
            if len(words) < max_words // 2:
                neighbor = self._by_doc_pos.get((h["doc_id"], h["posicion"] + 1))
                if neighbor:
                    candidate = text + " " + neighbor["texto"]
                    if len(candidate.split()) <= max_words:
                        enriched = candidate
            fragments.append({"rank": rank, "chunk_id": h["chunk_id"],
                               "doc_id": h["doc_id"], "text": enriched})
            rank += 1
        return fragments

    def answer_query(self, query_id: str, query_text: str) -> dict:
        k = RETRIEVE_TOP_K
        dense_hits = self._raw_search(query_text, k)
        hits = self._fused_hits(dense_hits, query_text)
        doc_ids = self._top_documents(hits, TOP_DOCUMENTS)
        fragments = self._top_fragments(hits, TOP_FRAGMENTS, MAX_WORDS_PER_FRAGMENT)

        while (len(doc_ids) < TOP_DOCUMENTS or len(fragments) < TOP_FRAGMENTS) and k < self.index.ntotal:
            k = min(k * 4, self.index.ntotal)
            dense_hits = self._raw_search(query_text, k)
            hits = self._fused_hits(dense_hits, query_text)
            doc_ids = self._top_documents(hits, TOP_DOCUMENTS)
            fragments = self._top_fragments(hits, TOP_FRAGMENTS, MAX_WORDS_PER_FRAGMENT)

        return {
            "query_id": query_id,
            "documents": [{"rank": i + 1, "doc_id": d} for i, d in enumerate(doc_ids)],
            "fragments": fragments,
        }


def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(here / "base_vectorial" / "encoder_multilingual-e5-base" / "index.faiss"))
    ap.add_argument("--metadata", default=str(here / "base_vectorial" / "encoder_multilingual-e5-base" / "metadata.jsonl"))
    ap.add_argument("--consultas", default=str(here / "lib" / "consultas.jsonl"))
    ap.add_argument("--out", default=str(here / "resultados.jsonl"))
    ap.add_argument("--grafo", default=str(here / "base_vectorial" / "grafo" / "grafo.graphml"),
                    help="Ruta a grafo.graphml (Sección 8.5). Si no existe o falla su "
                         "carga, se ignora y la recuperación usa solo FAISS.")
    ap.add_argument("--no-grafo", action="store_true",
                    help="Desactiva la fusión con el grafo aunque grafo.graphml exista.")
    args = ap.parse_args()

    queries = load_queries(Path(args.consultas))
    print(f"Consultas cargadas: {len(queries)}", file=sys.stderr)

    grafo_path = None if args.no_grafo else Path(args.grafo)
    ri = RetrievalIndex(Path(args.index), Path(args.metadata), grafo_path=grafo_path)
    print(f"Índice cargado: {ri.index.ntotal} vectores", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for q in queries:
            result = ri.answer_query(q["query_id"], q["query"])
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"  {q['query_id']} listo", file=sys.stderr)

    print(f"resultados.jsonl escrito en: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
