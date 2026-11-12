"""Knowledge banks: multimodal chunking, write/search over HTTP, and the memory-bank link.

The embedder is a deterministic fake (keywords and colours mapped to fixed axes), so these
tests check the plumbing — what a file becomes, what is stored, what a filter keeps, what
reflect reads from a linked bank — not the model. Retrieval quality is measured by
hindsight-dev/benchmarks/knowledge_banks_mm.
"""

import io
import json
import math
import shutil
import subprocess
import uuid
from dataclasses import replace

import pytest
from PIL import Image

from hindsight_api.knowledge import media
from hindsight_api.knowledge.embedders import EmbedInput, KnowledgeEmbedder
from hindsight_api.knowledge.service import _EMBED_ROUND, _Work, get_knowledge_service, run_ingest_task
from hindsight_api.knowledge.store import StatusUpdate

KB = "/v1/default/knowledge-banks"
_AXES = ("red", "blue", "key", "password", "invoice", "audio", "video", "green")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FakeEmbedder(KnowledgeEmbedder):
    provider = "fake"
    model = "fake"
    dimension = len(_AXES)
    modalities = frozenset({"text", "image", "audio", "video"})

    def __init__(self, modalities: frozenset[str] | None = None) -> None:
        if modalities is not None:
            self.modalities = modalities  # type: ignore[assignment]
        self.calls: list[list[EmbedInput]] = []

    @staticmethod
    def _text(text: str) -> list[float]:
        words = text.lower()
        return [float(words.count(axis)) for axis in _AXES]

    @staticmethod
    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else [1.0 / math.sqrt(len(v))] * len(v)

    def _one(self, item: EmbedInput) -> list[float]:
        if item.modality == "text":
            return self._norm(self._text(item.text or ""))
        if item.modality == "image":
            r, g, b = Image.open(io.BytesIO(item.data or b"")).convert("RGB").resize((1, 1)).getpixel((0, 0))
            v = [0.0] * len(_AXES)
            v[0], v[1], v[7] = r / 255, b / 255, g / 255
            return self._norm(v)
        v = [0.0] * len(_AXES)
        v[_AXES.index(item.modality)] = 1.0
        return v

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        self.calls.append(items)
        return [self._one(i) for i in items]

    async def embed_query(self, text: str | None, image: bytes | None = None) -> list[float]:
        if image is not None:
            return self._one(EmbedInput(modality="image", data=image))
        return self._norm(self._text(text or ""))


def _png(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def _pdf(*colors: tuple[int, int, int]) -> bytes:
    pages = [Image.new("RGB", (200, 280), c) for c in colors]
    out = io.BytesIO()
    pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:])
    return out.getvalue()


def _text_pdf(text: str) -> bytes:
    """A one-page PDF with a real text layer (PIL-made PDFs are images only)."""
    stream = f"BT /F1 24 Tf 40 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>stream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.writelines(b"%010d 00000 n \n" % o for o in offsets)
    out.write(b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref))
    return out.getvalue()


def _tone(seconds: int, *, video: bool) -> bytes:
    with __import__("tempfile").TemporaryDirectory() as tmp:
        out = f"{tmp}/clip.{'mp4' if video else 'm4a'}"
        args = ["ffmpeg", "-loglevel", "error", "-y"]
        if video:
            args += ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=5"]
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        if video:
            args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        subprocess.run([*args, "-c:a", "aac", "-shortest", out], check=True)
        return open(out, "rb").read()


# --- chunking (no database) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "filename", "kind"),
    [
        ("application/octet-stream", "manual.PDF", "pdf"),
        ("application/octet-stream", "photo.jpeg", "image"),
        ("audio/mpeg", None, "audio"),
        ("application/octet-stream", "talk.mov", "video"),
        ("text/plain", "notes", "text"),
        ("application/octet-stream", "spec.docx", "text"),
        ("application/json", None, "text"),
        ("application/zip", "archive.zip", None),
    ],
)
def test_classify_uses_extension_then_content_type(content_type, filename, kind):
    assert media.classify(content_type, filename) == kind


def test_windows_cover_the_media_with_overlap_and_fold_a_short_tail():
    spans = lambda *args: [(w.start, w.end) for w in media.windows(*args)]  # noqa: E731
    assert spans(0, 60, 5) == []
    assert spans(42, 60, 5) == [(0.0, 42)]
    assert spans(130, 60, 5) == [(0.0, 60), (55.0, 115.0), (110.0, 130)]
    # A 5s tail after the last full window is folded into it rather than standing alone.
    assert spans(120, 60, 5)[-1][1] == 120
    assert all(end - start >= 20 for start, end in spans(121, 60, 5))


def test_pdf_becomes_one_page_image_per_passage():
    drafts = list(media.split_pdf(_pdf((255, 0, 0), (0, 0, 255)), page_pixels=256, max_pages=10))
    assert [d.page for d in drafts] == [1, 2]
    assert all(d.modality == "image" and d.mime == "image/png" for d in drafts)
    assert max(Image.open(io.BytesIO(drafts[0].data or b"")).size) == 256


def test_pdf_page_keeps_its_text_layer_for_display():
    (page,) = media.split_pdf(_text_pdf("Invoices are emailed monthly"), page_pixels=128, max_pages=5)
    assert page.text == "Invoices are emailed monthly" and page.modality == "image"


def test_pdf_over_the_page_limit_is_refused():
    with pytest.raises(media.MediaError, match="limit is 1"):
        list(media.split_pdf(_pdf((1, 1, 1), (2, 2, 2)), page_pixels=64, max_pages=1))


def test_image_is_kept_as_is_or_converted_when_an_api_would_not_take_it():
    png = _png((0, 255, 0), size=(300, 300))
    assert media.split_image(png)[0].data == png
    bmp = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(bmp, format="BMP")
    converted = media.split_image(bmp.getvalue())[0]
    assert converted.mime == "image/png" and converted.data != bmp.getvalue()
    with pytest.raises(media.MediaError):
        media.split_image(b"not an image")


def test_text_is_split_into_token_sized_passages():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
    drafts = media.split_text(text, tokens=128)
    assert len(drafts) > 3 and all(d.modality == "text" and d.text for d in drafts)


def test_passages_follow_structure_and_keep_exact_order():
    from hindsight_api.engine.token_encoding import count_tokens
    from hindsight_api.knowledge.text_passages import split_into_passages

    para = "The tenant pays rent monthly. Late payment accrues interest. " * 3
    doc = (
        f"# Lease\n\n{para}\n\nARTICLE II\n\n{para}\n\n"
        "5. TERMINATION. Either party may end the lease with notice.\n\n" + "Long clause sentence here. " * 60
    )
    chunks = split_into_passages(doc, passage_size=64)
    # Every passage is an exact slice of the input, in order.
    cursor = 0
    for c in chunks:
        cursor = doc.index(c, cursor) + len(c)
    assert all(count_tokens(c) <= 64 for c in chunks)
    # Headings open passages: nothing merges across "ARTICLE II" or the numbered "5. TERMINATION".
    assert any(c.startswith("ARTICLE II") for c in chunks)
    assert any(c.startswith("5. TERMINATION") for c in chunks)


@needs_ffmpeg
def test_audio_and_video_are_cut_into_timed_windows():
    audio = list(
        media.split_media(
            _tone(70, video=False),
            "audio",
            filename="a.m4a",
            window_seconds=30,
            overlap_seconds=5,
            max_seconds=3600,
        )
    )
    assert [(d.start_ms, d.modality) for d in audio] == [(0, "audio"), (25000, "audio"), (50000, "audio")]
    assert audio[-1].end_ms == pytest.approx(70000, abs=200)
    video = list(
        media.split_media(
            _tone(8, video=True), "video", filename="v.mp4", window_seconds=30, overlap_seconds=5, max_seconds=3600
        )
    )
    assert len(video) == 1 and video[0].mime == "video/mp4"
    assert media.decode_video_frames(video[0].data or b"").shape[1:] == (448, 448, 3)
    assert media.decode_audio_pcm(video[0].data or b"").size > 16000 * 7
    with pytest.raises(media.MediaError, match="limit is 5s"):
        list(media.split_media(_tone(8, video=False), "audio", filename="a.m4a", window_seconds=30, overlap_seconds=5, max_seconds=5))  # fmt: skip


# --- HTTP ---------------------------------------------------------------------------------------


@pytest.fixture
def fake_embedder(memory):
    service = get_knowledge_service(memory)
    previous = service._embedder
    service._embedder = FakeEmbedder()
    yield service._embedder
    service._embedder = previous


@pytest.fixture
async def kb(api_client, fake_embedder):
    bank_id = f"kb-{uuid.uuid4().hex[:8]}"
    r = await api_client.post(KB, json={"bank_id": bank_id})
    assert r.status_code == 200, r.text
    yield bank_id
    await api_client.delete(f"{KB}/{bank_id}")


async def _upload(client, bank_id: str, files: list[tuple[str, bytes, str]], documents: list[dict] | None = None):
    form = {"request": json.dumps({"files": documents})} if documents else None
    return await client.post(
        f"{KB}/{bank_id}/files/ingest", files=[("files", (n, d, t)) for n, d, t in files], data=form
    )


async def _search(client, bank_id: str, **body) -> list[dict]:
    r = await client.post(f"{KB}/{bank_id}/search", json=body)
    assert r.status_code == 200, r.text
    return r.json()["results"]


def test_page_score_follows_the_recipe():
    from hindsight_api.knowledge.store import page_score

    assert page_score(0.8, {}, 0.6) == 0.8  # anything with one vector scores as that vector
    image = 0.5 * 0.4 + 0.5 * 0.9  # page image averaged with its best band
    assert page_score(0.4, {"band": [0.1, 0.9]}, 0.6) == image
    roles = {"band": [0.9], "page_text": [0.3], "unit": [0.7, 0.5]}
    assert abs(page_score(0.4, roles, 0.6) - (0.4 * image + 0.6 * 0.7)) < 1e-9


async def test_openai_compatible_embedder_sends_text_as_input_and_media_as_messages():
    """The remote provider speaks vLLM's OpenAI embeddings dialect: texts batched, media one per request."""
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder, UnsupportedModalityError
    from tests.aiohttp_stub import stub_server

    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append(body)
        n = len(body["input"]) if "input" in body else 1
        return web.json_response({"data": [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(n))]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None,
            modalities=frozenset({"text", "image"}), batch_size=2,
        )  # fmt: skip
        vectors = await embedder.embed_documents(
            [
                EmbedInput(modality="text", text="a", title="T"),
                EmbedInput(modality="image", data=_png((1, 2, 3)), mime="image/png"),
                EmbedInput(modality="text", text="b"),
            ]
        )
        assert vectors == [[0.6, 0.8]] * 3 and embedder.dimension == 2
        assert seen[0] == {"model": "m", "input": ["title: T | text: a", "title: none | text: b"]}
        part = seen[1]["messages"][0]["content"][0]
        assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
        with pytest.raises(UnsupportedModalityError):
            await embedder.embed_documents([EmbedInput(modality="audio", data=b"x", mime="audio/mp4")])


def test_pdfs_split_concurrently_from_many_threads_do_not_crash():
    """pdfium is not thread-safe; the worker ingests PDFs on parallel threads (exit 134 before the lock)."""
    from concurrent.futures import ThreadPoolExecutor

    pdf = _pdf(*[(i * 20, 0, 0) for i in range(6)])
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: len(list(media.split_pdf(pdf, page_pixels=256, max_pages=10))), range(16)))
    assert results == [6] * 16


def test_small_or_transparent_images_are_upscaled_onto_a_contrasting_background():
    sprite = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(8, 24):
        for y in range(8, 24):
            sprite.putpixel((x, y), (250, 250, 250, 255))  # a white sprite: a white background would erase it
    out = io.BytesIO()
    sprite.save(out, format="PNG")
    (draft,) = media.split_image(out.getvalue())
    image = Image.open(io.BytesIO(draft.data or b""))
    assert min(image.size) >= 256 and image.mode == "RGB"
    assert image.getpixel((0, 0)) == (32, 32, 32)  # dark behind a light sprite
    photo = _png((10, 200, 10), size=(512, 400))
    assert media.split_image(photo)[0].data == photo  # big opaque images pass through untouched


async def test_media_is_embedded_with_its_name_and_tags_in_one_input():
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder
    from tests.aiohttp_stub import stub_server

    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None,
            modalities=frozenset({"text", "audio"}), batch_size=4,
        )  # fmt: skip
        await embedder.embed_documents(
            [
                EmbedInput(
                    modality="audio", data=b"x", mime="audio/mp4", title="footstep_gravel_03", tags=("sfx", "walk")
                )
            ]
        )
        await embedder.embed_documents([EmbedInput(modality="audio", data=b"x", mime="audio/mp4")])
    named, bare = seen[0]["messages"][0]["content"], seen[1]["messages"][0]["content"]
    assert named[0] == {"type": "text", "text": "title: footstep_gravel_03 | tags: sfx, walk | text: "}
    assert named[1]["type"] == "audio_url" and [p["type"] for p in bare] == ["audio_url"]


async def test_remote_embedder_retries_a_dropped_connection():
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder
    from tests.aiohttp_stub import stub_server

    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return web.Response(status=503, text="busy")
        return web.json_response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None, modalities=frozenset({"text"}), batch_size=4
        )
        assert await embedder.embed_query("hello") == [1.0, 0.0]
    assert calls["n"] == 2


def test_language_detection_and_keyword_fusion():
    from hindsight_api.knowledge.language import detect, text_search_config
    from hindsight_api.knowledge.store import fuse_keyword

    assert detect("The rotation of the keys is done by the admin in the console") == "english"
    assert detect("La rotation des clés est faite par l'administrateur dans la console") == "french"
    assert detect("ERR-4417") is None
    assert (text_search_config("german"), text_search_config(None)) == ("german", "simple")
    # No keyword hit: dense scores pass through untouched.
    assert fuse_keyword([0.5, 0.4], [0.0, 0.0], 0.2) == [0.5, 0.4]
    # A keyword hit can lift a slightly weaker dense match over the best one.
    fused = fuse_keyword([0.50, 0.49, 0.10], [0.0, 1.0, 0.0], 0.2)
    assert fused[1] > fused[0] > fused[2]


def test_running_headers_and_footers_are_not_embedded():
    topics = ["Entropy", "Spin", "Lasers", "Relativity", "Chaos"]
    # Page numbers differ, so digits are ignored when lines are compared.
    pages = [f"Physics for everyone - 2024-2025 {i + 1}\n{t} explained\n{i + 1}" for i, t in enumerate(topics)]
    stripped = media.strip_repeated_lines([*pages, None])
    assert stripped[:5] == [f"{t} explained" for t in topics]
    assert stripped[5] is None
    # Two short documents: nothing is frequent enough to call a header.
    assert media.strip_repeated_lines(["Title\nA", "Title\nB"]) == ["Title\nA", "Title\nB"]


def test_the_file_name_is_the_title_unless_it_was_made_up_from_the_id():
    from types import SimpleNamespace

    from hindsight_api.knowledge.service import _title_hint

    def title(document_id: str, filename: str | None) -> str | None:
        return _title_hint(SimpleNamespace(document_id=document_id, filename=filename))  # type: ignore[arg-type]

    assert title("annual-report", "annual-report.pdf") == "annual report"  # named like its id: kept
    assert title("doc-1", "footstep_gravel_03.wav") == "footstep gravel 03"
    assert title("3f2a", "3f2a.md") is None  # the name a text write gets when it has none
    assert title("x", None) is None


def test_pdf_page_text_is_packed_by_whole_lines():
    from hindsight_api.engine.token_encoding import count_tokens

    lines = [f"Row {i}: revenue grew in region {i}" for i in range(60)]
    units = media.pack_lines("\r\n".join(lines), tokens=40)
    assert all(count_tokens(u) <= 40 for u in units)
    assert [line for u in units for line in u.split("\n")] == lines  # whole lines, in order, none lost
    long = " ".join(["word"] * 200)
    assert all(count_tokens(u) <= 40 for u in media.pack_lines(long, tokens=40))


def test_pages_lean_on_the_image_when_their_text_is_junk_and_on_their_document():
    from hindsight_api.knowledge.store import page_text_weight, with_document_prior

    prose = "The employment rate of women rose across the member states in the last decade. " * 4
    equations = "∂ψ/∂t = −iħ⁻¹ Hψ ; E = mc² ; λ = h/p ; 𝜓(𝑟,𝑡) = Σ cₙ φₙ e^(−iEₙt/ħ) ; " * 6
    assert page_text_weight(prose, 0.6) == 0.6
    assert page_text_weight(equations, 0.6) == 0.3
    assert page_text_weight("short", 0.6) == 0.3  # almost no text layer
    # A page of the best-matching PDF overtakes a slightly better page of a weaker PDF...
    scores = with_document_prior(["deck", "deck", "other"], [0.80, 0.70, 0.72])
    assert scores[1] > scores[2]
    # ...and a one-passage document keeps its plain similarity.
    assert with_document_prior(["img"], [0.42]) == [pytest.approx(0.42)]


def test_an_abandoned_pdf_split_can_be_closed_while_pdfium_is_in_use():
    # A failed ingest leaves its page generator half-read; closing it (the PDF, under pdfium's lock)
    # from a thread already inside that lock used to deadlock.
    drafts = media.split_pdf(_pdf((255, 0, 0), (0, 0, 255)), page_pixels=256, max_pages=10)
    next(drafts)
    with media._PDFIUM_LOCK:
        drafts.close()


async def test_remote_embedder_keeps_a_few_media_requests_in_flight():
    import asyncio

    from hindsight_api.knowledge import embedders

    embedder = embedders.OpenAIEmbedder(
        "m", base_url="http://unused/v1", api_key=None, dimensions=3, modalities=frozenset({"image"}), batch_size=8
    )
    in_flight = peak = 0

    async def one(item: EmbedInput) -> list[float]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [float(item.title or 0), 0.0, 0.0]

    embedder._embed_media = one  # type: ignore[method-assign]
    pages = [EmbedInput(modality="image", data=b"x", mime="image/png", title=str(n)) for n in range(20)]
    vectors = await embedder.embed_documents(pages)
    assert [v[0] for v in vectors] == list(range(20))  # every page, in order
    assert peak == embedders._MEDIA_CONCURRENCY


@pytest.fixture
def counted(memory, monkeypatch):
    """Counts the store writes and queued operations an ingest makes, and keeps each task's payload."""
    store = get_knowledge_service(memory).store
    counts = {"put_documents": 0, "replace_passages": 0}
    payloads: list[dict] = []
    for name in counts:
        real = getattr(store, name)

        async def spy(*args, _real=real, _name=name, **kwargs):
            counts[_name] += 1
            return await _real(*args, **kwargs)

        monkeypatch.setattr(store, name, spy)
    submit = memory._submit_async_operation

    async def submit_spy(*args, **kwargs):
        payloads.append(kwargs["task_payload"])
        return await submit(*args, **kwargs)

    monkeypatch.setattr(memory, "_submit_async_operation", submit_spy)
    return counts, payloads
