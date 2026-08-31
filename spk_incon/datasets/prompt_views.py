"""InstructTTSEval-shaped prompt views for LibriTTS-P training samples."""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Mapping, NamedTuple, Sequence


class PromptView(str, Enum):
    """Instruction register a LibriTTS-P sample is rendered into."""

    RAW = "raw"
    APS = "aps"
    DSD = "dsd"
    RP = "rp"


APS_KEYS: tuple[str, ...] = (
    "gender",
    "pitch",
    "speed",
    "volume",
    "age",
    "clarity",
    "fluency",
    "accent",
    "texture",
    "emotion",
    "tone",
    "personality",
)

APS_OMITTED_KEYS: frozenset[str] = frozenset({"accent", "emotion"})

DEFAULT_VIEW_WEIGHTS: dict[PromptView, float] = {
    PromptView.RAW: 1 / 3,
    PromptView.APS: 1 / 4,
    PromptView.DSD: 1 / 4,
    PromptView.RP: 1 / 6,
}

RP_GENDER_IMPLICIT_RATE: float = 0.6

_SALT_STRIDE = 1_000_000_007
_SELECT_SALT = 97
_VIEW_SALTS: dict[PromptView, int] = {
    PromptView.RAW: 0,
    PromptView.APS: 1,
    PromptView.DSD: 2,
    PromptView.RP: 3,
}

_GENDER_ALIASES = {
    "m": "M",
    "male": "M",
    "man": "M",
    "f": "F",
    "female": "F",
    "woman": "F",
}

_PITCH_ALIASES = {
    "very low": "very low",
    "low": "low",
    "normal": "normal",
    "mid": "normal",
    "medium": "normal",
    "high": "high",
    "very high": "very high",
}

_SPEED_ALIASES = {
    "very slow": "very slow",
    "slow": "slow",
    "normal": "normal",
    "medium": "normal",
    "fast": "fast",
    "very fast": "very fast",
}

_COARSE_LEVELS = {
    "very low": "low",
    "low": "low",
    "normal": "normal",
    "high": "high",
    "very high": "high",
    "very slow": "slow",
    "slow": "slow",
    "fast": "fast",
    "very fast": "fast",
}

_TAG_SLOTS: dict[str, tuple[str, ...]] = {
    "young": ("age",),
    "adult-like": ("age",),
    "middle-aged": ("age",),
    "mature": ("age",),
    "old": ("age",),
    "clear": ("clarity",),
    "muffled": ("clarity",),
    "nasal": ("clarity",),
    "thick": ("clarity", "texture"),
    "thin": ("clarity", "texture"),
    "fluent": ("fluency",),
    "halting": ("fluency",),
    "bright": ("texture",),
    "dark": ("texture",),
    "raspy": ("texture",),
    "soft": ("texture",),
    "sweet": ("texture",),
    "light": ("texture",),
    "sharp": ("texture",),
    "weak": ("texture",),
    "powerful": ("texture",),
    "hard": ("texture",),
    "elegant": ("tone",),
    "cool": ("tone",),
    "strict": ("tone",),
    "sincere": ("tone",),
    "reassuring": ("tone",),
    "refreshing": ("tone",),
    "intellectual": ("personality",),
    "friendly": ("personality",),
    "kind": ("personality",),
    "modest": ("personality",),
    "wild": ("personality",),
    "lively": ("personality",),
    "calm": ("personality",),
    "relaxed": ("personality",),
    "tensed": ("personality",),
    "intense": ("personality",),
    "unique": ("personality",),
    "cute": ("personality",),
    "sexy": ("personality",),
    "masculine": ("gender",),
    "feminine": ("gender",),
    "gender-neutral": ("gender",),
}

_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "young": ("young", "youthful", "young-sounding"),
    "adult-like": ("adult", "grown-up", "fully adult"),
    "middle-aged": ("middle-aged", "midlife"),
    "mature": ("mature", "seasoned", "well-settled"),
    "old": ("elderly", "older", "aged"),
    "clear": ("clear", "crisp", "clean-edged"),
    "muffled": ("muffled", "veiled", "blunted"),
    "nasal": ("nasal", "nasally colored"),
    "thick": ("thick", "heavy-bodied", "dense"),
    "thin": ("thin", "reedy", "slender"),
    "fluent": ("fluent", "smooth", "unbroken"),
    "halting": ("halting", "hesitant", "stumbling"),
    "bright": ("bright", "brilliant", "gleaming"),
    "dark": ("dark", "shadowed", "low-weighted"),
    "raspy": ("raspy", "gravelly", "grainy"),
    "soft": ("soft", "cushioned", "downy"),
    "sweet": ("sweet", "honeyed"),
    "light": ("light", "airy", "feathery"),
    "sharp": ("sharp", "keen-edged", "incisive"),
    "weak": ("weak", "slight", "fragile"),
    "powerful": ("powerful", "robust", "full-throated"),
    "hard": ("hard", "firm-edged", "steely"),
    "elegant": ("elegant", "refined", "polished"),
    "cool": ("cool", "composed", "even-tempered"),
    "strict": ("strict", "stern", "no-nonsense"),
    "sincere": ("sincere", "earnest", "candid"),
    "reassuring": ("reassuring", "steadying", "comforting"),
    "refreshing": ("refreshing", "bracing", "invigorating"),
    "intellectual": ("intellectual", "studious", "thoughtful"),
    "friendly": ("friendly", "warm", "approachable"),
    "kind": ("kind", "gentle", "considerate"),
    "modest": ("modest", "unassuming", "understated"),
    "wild": ("wild", "unruly", "untamed"),
    "lively": ("lively", "animated", "spirited"),
    "calm": ("calm", "settled", "unruffled"),
    "relaxed": ("relaxed", "easygoing", "loose"),
    "tensed": ("tense", "keyed-up", "tightly wound"),
    "intense": ("intense", "driven", "burning"),
    "unique": ("distinctive", "singular", "unmistakable"),
    "cute": ("cute", "endearing", "charming"),
    "sexy": ("alluring", "sultry", "seductive"),
    "masculine": ("masculine",),
    "feminine": ("feminine",),
    "gender-neutral": ("gender-neutral", "androgynous"),
}

_DEGREE_ADVERBS: dict[str, tuple[str, ...]] = {
    "slightly": ("slightly", "somewhat", "mildly", "faintly"),
    "very": ("very", "markedly", "distinctly", "strongly"),
}

_PAIR_JOINERS: tuple[str, ...] = (" and ", ", ")

_SLOT_MAX_TERMS: dict[str, int] = {
    "age": 2,
    "clarity": 2,
    "fluency": 1,
    "texture": 2,
    "tone": 2,
    "personality": 3,
}

_SLOT_FRAMES: dict[str, tuple[str, ...]] = {
    "age": ("{d}", "Sounds {d}", "Reads as {d}", "{d}, judging by the timbre"),
    "clarity": ("{d}", "Articulation is {d}", "Diction reads as {d}", "{d} throughout"),
    "fluency": ("{d}", "Delivery is {d}", "Phrasing is {d}", "{d} from phrase to phrase"),
    "texture": ("{d}", "Texture is {d}", "The timbre is {d}", "{d} in texture"),
    "tone": ("{d}", "Tone is {d}", "Overall {d}", "{d} in tone"),
    "personality": ("{d}", "Comes across as {d}", "Reads as {d}", "{d} in manner"),
}

_GENDER_PHRASES: dict[str, tuple[str, ...]] = {
    "M": ("Male", "Male voice", "A male speaker", "Adult male"),
    "F": ("Female", "Female voice", "A female speaker", "Adult female"),
}

_GENDER_FRAMES: tuple[str, ...] = ("{base}, {d} in timbre", "{base}, reading as {d}")

_APS_PITCH: dict[str, tuple[str, ...]] = {
    "very low": (
        "Deep and resonant, well below the typical {g} range",
        "Notably low-pitched, sitting at the bottom of the {g} register",
        "A deep register held low across the whole utterance",
    ),
    "low": (
        "Low-pitched for a {g} voice, settling into the lower register",
        "Below the average {g} range, with a grounded low placement",
        "Lower than average, holding a steady low register",
    ),
    "normal": (
        "Mid-range {g} pitch, holding steady across the utterance",
        "Average pitch for this speaker, with only slight movement",
        "Conversational mid-range pitch throughout",
    ),
    "high": (
        "Higher than the average {g} range, with a bright placement",
        "Raised pitch, sitting above the usual {g} register",
        "A high placement carried above the conversational midpoint",
    ),
    "very high": (
        "Markedly high-pitched, riding near the top of the {g} range",
        "Very high placement, bright and elevated throughout",
        "Well above the usual {g} range, lifted and bright",
    ),
}

_APS_SPEED: dict[str, tuple[str, ...]] = {
    "very slow": (
        "Deliberately slow, with generous room between phrases",
        "Markedly unhurried, each phrase drawn out",
        "Very slow pacing, nothing pushed forward",
    ),
    "slow": (
        "Unhurried pacing, slower than conversational tempo",
        "A measured, slower-than-average delivery",
        "Relaxed tempo, giving each phrase space",
    ),
    "normal": (
        "Conversational tempo, steady from start to finish",
        "An even, everyday pace with no abrupt shifts",
        "Ordinary speaking speed, held consistently",
    ),
    "fast": (
        "Brisk pacing, quicker than conversational tempo",
        "A quick delivery that keeps the phrases tightly spaced",
        "Faster than average, moving through the line without pause",
    ),
    "very fast": (
        "Rapid throughout, phrases running close together",
        "Very quick delivery, barely pausing between phrases",
        "Markedly fast, the words arriving in a rush",
    ),
}

_APS_VOLUME: dict[str, tuple[str, ...]] = {
    "very low": (
        "Hushed, well below conversational level",
        "Very quiet, close to a murmur",
        "Held far under conversational volume throughout",
    ),
    "low": (
        "Softer than conversational level, kept restrained",
        "Quiet delivery, held below the usual speaking volume",
        "Subdued volume, never pushed outward",
    ),
    "normal": (
        "Conversational level, maintained steadily",
        "Everyday speaking volume with no marked swings",
        "An ordinary conversational level held throughout",
    ),
    "high": (
        "Elevated and forceful, louder than conversational norms",
        "Projected well above conversational level",
        "Loud and emphatic, carrying past the usual speaking volume",
    ),
    "very high": (
        "Loud and strongly projected, carrying far beyond conversational level",
        "Forceful volume held high from start to finish",
        "Near the top of the speaker's dynamic range throughout",
    ),
}

_DSD_OPENERS: tuple[str, ...] = (
    "Speak with {np}",
    "Craft your speech with {np}",
    "Begin with {np}",
    "Start with {np}",
    "Infuse your delivery with {np}",
    "Let your voice settle into {np}",
    "Deliver this with {np}",
    "Maintain {np}",
    "Channel {np}",
    "Elevate the delivery with {np}",
)

_DSD_LINKS: tuple[str, ...] = (
    ", keeping {np} underneath",
    ", layering in {np}",
    ", carrying {np} through the whole line",
    " and {np}",
    ", and let {np} shape the phrasing",
)

_DSD_EXTRA_LINKS: tuple[str, ...] = (
    " while holding {np}",
    ", and {np} on top of that",
    ", set against {np}",
)

_DSD_CLOSERS: tuple[str, ...] = (
    "let the {d} quality carry through every phrase",
    "keep that {d} edge audible from the first word to the last",
    "let it read as {d} without ever forcing the effect",
    "hold the {d} coloring steady all the way to the end",
    "make the {d} character the thing a listener remembers",
    "let the {d} side of the voice surface on its own rather than being announced",
    "keep the {d} coloring present in the vowels even where the line goes quiet",
)

_DSD_PLAIN_CLOSERS: tuple[str, ...] = (
    "hold that balance steady from the first word to the last",
    "let the delivery stay recognizably itself all the way through",
    "keep the shape of it consistent across every phrase",
)

_DSD_PITCH_NP: dict[str, tuple[str, ...]] = {
    "very low": ("a deep, resonant register", "a register that sits far down in your range"),
    "low": ("a low, grounded register", "a pitch settled below your usual center"),
    "normal": ("an even mid-range pitch", "a register held at its usual center"),
    "high": ("a raised, bright register", "a pitch lifted above center"),
    "very high": ("a bright register near the top of your range", "a markedly high, lifted pitch"),
}

_DSD_SPEED_NP: dict[str, tuple[str, ...]] = {
    "very slow": ("a deliberately unhurried pace", "pacing drawn out with room between phrases"),
    "slow": ("an unhurried pace", "slower-than-usual pacing"),
    "normal": ("a steady conversational pace", "an even, unforced tempo"),
    "fast": ("a brisk pace", "quickened pacing that keeps the phrases close"),
    "very fast": ("a rapid, tightly spaced pace", "urgency in every quick utterance"),
}

_DSD_VOLUME_NP: dict[str, tuple[str, ...]] = {
    "very low": ("a hushed, near-whispered level", "volume kept far below conversation"),
    "low": ("a softened, restrained volume", "volume held below conversational level"),
    "normal": ("a conversational level", "volume held at everyday level"),
    "high": ("a projected, forceful level", "volume pushed above conversational norms"),
    "very high": ("full-throated projection", "volume carried loud and far"),
}

_RP_CAPTION_ADJECTIVES: tuple[str, ...] = (
    "seasoned",
    "practiced",
    "unhurried",
    "hard-edged",
    "quietly assured",
    "long-serving",
    "unflappable",
    "road-worn",
)

_RP_TRAIT_MODIFIERS: tuple[str, ...] = (
    "unwavering",
    "quiet",
    "hard-earned",
    "undisguised",
    "practiced",
    "steady",
    "open",
)

_RP_CAPTION_FRAMES: tuple[str, ...] = (
    "The {adj} {noun} {action} with {mod} {trait}, {m1}, {m2}.",
    "The {adj} {noun} {action}, {mod} {trait} sitting under every line, {m1}.",
    "The {adj} {noun} {action}, all {mod} {trait}, {m1}, {m2}.",
)

_RP_BRIEF_OPENERS: tuple[str, ...] = ("Channel", "Take on", "Borrow", "Step into", "Summon")

_RP_PITCH_MANNER: dict[str, tuple[str, ...]] = {
    "low": ("the register kept low", "anchored well down in the register"),
    "normal": ("the register held at center", "pitch sitting where conversation sits"),
    "high": ("the register lifted bright", "riding high and bright in the range"),
}

_RP_SPEED_MANNER: dict[str, tuple[str, ...]] = {
    "slow": ("each phrase given room", "the pace unhurried", "nothing rushed between phrases"),
    "normal": ("the pace steady throughout", "tempo even from line to line"),
    "fast": ("the words coming quickly", "phrases running one into the next"),
}

_RP_ENERGY_MANNER: dict[str, tuple[str, ...]] = {
    "low": ("kept quiet and close", "volume held down to almost nothing"),
    "normal": ("held at an even level", "volume kept conversational"),
    "high": ("carrying to the back of the room", "loud enough to cut clean through"),
}

_RP_VOICE_CLAUSES: dict[str, tuple[str, ...]] = {
    "low": ("kept low and close", "pulled back almost to a murmur", "held quiet"),
    "normal": ("held steady at conversational level", "kept even and unforced"),
    "high": ("carrying to the back of the room", "pushed loud and open", "peppered with urgency"),
}

_ANY_PITCH = frozenset({"low", "normal", "high"})
_ANY_SPEED = frozenset({"slow", "normal", "fast"})
_ANY_ENERGY = frozenset({"low", "normal", "high"})


class _Role(NamedTuple):
    """One benchmark-style speaking role with its acoustic compatibility window."""

    noun: str
    traits: tuple[str, ...]
    actions: tuple[str, ...]
    names_gender: bool = False
    gender: str | None = None
    pitch: frozenset[str] = _ANY_PITCH
    speed: frozenset[str] = _ANY_SPEED
    energy: frozenset[str] = _ANY_ENERGY


_ROLES: tuple[_Role, ...] = (
    _Role(
        "broadcaster",
        ("poise", "control", "clarity"),
        ("walks a listener through the evening headlines", "hands the audience each item cleanly"),
        speed=frozenset({"normal"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "drill sergeant",
        ("grit", "vigor", "bite"),
        ("drives a formation forward across the parade ground", "barks the next order down the line"),
        pitch=frozenset({"low", "normal"}),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"high"}),
    ),
    _Role(
        "politician",
        ("conviction", "command", "resolve"),
        ("firmly refutes accusations", "turns a hostile question back on the room"),
        speed=frozenset({"slow", "normal"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "courtroom advocate",
        ("precision", "conviction", "control"),
        ("presses a closing argument to its last point", "lays the evidence out one piece at a time"),
        speed=frozenset({"slow", "normal"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "radio host",
        ("warmth", "ease", "charm"),
        ("keeps a late show rolling between records", "greets the next caller like an old friend"),
        speed=frozenset({"normal"}),
        energy=frozenset({"normal"}),
    ),
    _Role(
        "philosopher",
        ("patience", "depth", "calm"),
        ("turns a difficult idea over in the open", "lets a hard question sit before answering it"),
        speed=frozenset({"slow", "normal"}),
        energy=frozenset({"low", "normal"}),
    ),
    _Role(
        "stage performer",
        ("presence", "flourish", "projection"),
        ("holds the back row of a full theater", "plays the line out to the last seat"),
        energy=frozenset({"high"}),
    ),
    _Role(
        "tech presenter",
        ("clarity", "momentum", "assurance"),
        ("walks a packed hall through the new release", "builds toward the one number that matters"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "tech enthusiast",
        ("eagerness", "spark", "curiosity"),
        ("cannot get through the spec sheet fast enough", "races ahead to the part worth showing"),
        speed=frozenset({"fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "motivational speaker",
        ("drive", "lift", "certainty"),
        ("pushes a room toward its own next step", "turns doubt into something workable"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"high"}),
    ),
    _Role(
        "exasperated parent",
        ("weariness", "impatience", "strain"),
        ("asks for the third time and means it", "runs out of patience halfway through the sentence"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "exasperated dad",
        ("weariness", "impatience", "strain"),
        ("asks for the third time and means it", "gives up on the reasonable version of the request"),
        names_gender=True,
        gender="M",
        pitch=frozenset({"low", "normal"}),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "protester",
        ("urgency", "defiance", "heat"),
        ("carries a chant over the noise of the street", "refuses to let the line go quiet"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"high"}),
    ),
    _Role(
        "project leader",
        ("steadiness", "focus", "assurance"),
        ("sets the week's priorities before anyone objects", "keeps a stalled meeting moving"),
        speed=frozenset({"normal"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "sorceress",
        ("mystery", "command", "hush"),
        ("draws an incantation out of the dark", "names the thing that should not be named"),
        names_gender=True,
        gender="F",
        pitch=frozenset({"low", "normal"}),
        speed=frozenset({"slow", "normal"}),
    ),
    _Role(
        "activist",
        ("conviction", "urgency", "clarity"),
        ("makes the case one more time to a tired crowd", "refuses the easy compromise"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "rebel",
        ("edge", "defiance", "nerve"),
        ("says the part everyone else swallowed", "pushes back before the sentence is finished"),
        speed=frozenset({"normal", "fast"}),
        energy=frozenset({"normal", "high"}),
    ),
    _Role(
        "British lady",
        ("poise", "restraint", "polish"),
        ("offers a correction and never raises the volume", "keeps every vowel exactly where it belongs"),
        names_gender=True,
        gender="F",
        speed=frozenset({"slow", "normal"}),
        energy=frozenset({"low", "normal"}),
    ),
    _Role(
        "public speaker",
        ("composure", "clarity", "assurance"),
        ("holds a hall through a long opening", "carries the room from the first line"),
    ),
    _Role(
        "storyteller",
        ("warmth", "patience", "craft"),
        ("draws a room into a long, familiar tale", "lets the story find its own pace"),
    ),
)


def _rng(epoch: int, idx: int, salt: int) -> random.Random:
    """Builds the deterministic generator for one (epoch, sample, view) triple.

    Args:
        epoch (int): Current training epoch.
        idx (int): Global index of the sample.
        salt (int): Stream identifier keeping views decorrelated.

    Returns:
        random.Random: Seeded generator.
    """
    return random.Random(int(epoch * 1e6 + idx) + salt * _SALT_STRIDE)


def _coerce_view(view: PromptView | str) -> PromptView:
    """Resolves a view given as an enum member, value string, or member name.

    Args:
        view (PromptView or str): View identifier.

    Returns:
        PromptView: The matching member.

    Raises:
        ValueError: If the identifier matches no member.
    """
    if isinstance(view, PromptView):
        return view
    return PromptView(str(view).strip().lower())


def _article(phrase: str) -> str:
    """Chooses the indefinite article for a phrase by its leading sound.

    Args:
        phrase (str): Phrase the article precedes.

    Returns:
        str: Either ``"a"`` or ``"an"``.
    """
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def _capitalize(text: str) -> str:
    """Uppercases the first character while leaving the remainder untouched.

    Args:
        text (str): Sentence body.

    Returns:
        str: Sentence with a capital initial.
    """
    return text[:1].upper() + text[1:] if text else text


def _normalize_gender(gender: str | None) -> str | None:
    """Maps a gender field onto the ``"M"``/``"F"`` codes used by the lexicons.

    Args:
        gender (str, optional): Raw gender value.

    Returns:
        str or None: ``"M"``, ``"F"``, or None when unrecognized.
    """
    if gender is None:
        return None
    return _GENDER_ALIASES.get(str(gender).strip().lower())


def _normalize_level(value: str | None, aliases: dict[str, str]) -> str:
    """Maps a raw acoustic label onto its five-step canonical level.

    Args:
        value (str, optional): Raw label such as ``"very slow"``.
        aliases (dict): Alias table for the axis.

    Returns:
        str: Canonical level, falling back to ``"normal"``.
    """
    if value is None:
        return "normal"
    return aliases.get(str(value).strip().lower(), "normal")


def _parse_tags(tags: Sequence[str] | str | None) -> list[tuple[str, str]]:
    """Splits df1 speaker tags into (degree, base adjective) pairs.

    Args:
        tags (Sequence[str] or str, optional): Speaker tags, comma-joined or listed.

    Returns:
        list: Pairs whose base adjective is known to the slot router.
    """
    if not tags:
        return []
    if isinstance(tags, str):
        tags = tags.split(",")

    parsed: list[tuple[str, str]] = []
    for raw in tags:
        token = str(raw).strip().lower()
        degree = ""
        for prefix in ("slightly ", "very "):
            if token.startswith(prefix):
                degree = prefix.strip()
                token = token[len(prefix):]
                break
        if token in _TAG_SLOTS:
            parsed.append((degree, token))
    return parsed


def _route_tags(rng: random.Random, parsed: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Assigns every parsed tag to exactly one attribute slot.

    Args:
        rng (random.Random): Seeded generator.
        parsed (list): Pairs produced by :func:`_parse_tags`.

    Returns:
        dict: Slot name mapped to the tags routed into it.
    """
    routed: dict[str, list[tuple[str, str]]] = {}
    for degree, base in parsed:
        candidates = _TAG_SLOTS[base]
        slot = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
        routed.setdefault(slot, []).append((degree, base))
    return routed


def _describe(rng: random.Random, entries: list[tuple[str, str]], max_terms: int) -> str:
    """Turns routed tags into one lowercase descriptor phrase.

    Args:
        rng (random.Random): Seeded generator.
        entries (list): (degree, base adjective) pairs for a single slot.
        max_terms (int): Upper bound on how many tags reach the surface.

    Returns:
        str: Descriptor phrase, empty when no tags were supplied.
    """
    if not entries:
        return ""

    picked = entries if len(entries) <= max_terms else rng.sample(entries, max_terms)
    parts = []
    for degree, base in picked:
        word = rng.choice(_DESCRIPTORS[base])
        adverb = rng.choice(_DEGREE_ADVERBS[degree]) if degree else ""
        parts.append(f"{adverb} {word}".strip())

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return rng.choice(_PAIR_JOINERS).join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _gender_word(gender: str | None) -> str:
    """Returns the adjective used inside pitch phrasings.

    Args:
        gender (str, optional): Canonical gender code.

    Returns:
        str: ``"male"``, ``"female"``, or the neutral ``"speaking"``.
    """
    if gender == "M":
        return "male"
    if gender == "F":
        return "female"
    return "speaking"


def _infer_gender(gender: str | None, routed: dict[str, list[tuple[str, str]]]) -> str | None:
    """Falls back to the masculine/feminine tags when the gender field is absent.

    Args:
        gender (str, optional): Canonical gender code.
        routed (dict): Slot-routed tags.

    Returns:
        str or None: Best available gender code.
    """
    if gender is not None:
        return gender
    for _, base in routed.get("gender", []):
        if base == "masculine":
            return "M"
        if base == "feminine":
            return "F"
    return None


def _render_raw(
    rng: random.Random,
    tags: Sequence[str] | str | None,
    style_variant: str | Sequence[str] | None,
) -> str:
    """Renders the LibriTTS-P style plus speaker-tag prompt.

    Args:
        rng (random.Random): Seeded generator.
        tags (Sequence[str] or str, optional): Speaker tags.
        style_variant (str or Sequence[str], optional): One style sentence, or the
            full candidate list to draw one from.

    Returns:
        str: Combined prompt.
    """
    style = ""
    if isinstance(style_variant, str):
        style = style_variant
    elif style_variant:
        style = rng.choice(list(style_variant))

    tag_list = list(tags.split(",") if isinstance(tags, str) else (tags or []))
    tag_list = [str(tag).strip() for tag in tag_list if str(tag).strip()]
    rng.shuffle(tag_list)

    identity = ""
    if tag_list:
        identity = f"The speaker's identity can be described as {', '.join(tag_list)}."

    style = style.rstrip().rstrip(".")
    if style and identity:
        return f"{style}. {identity}"
    return f"{style}." if style else identity


def _render_aps(
    rng: random.Random,
    gender: str | None,
    pitch: str,
    speed: str,
    energy: str,
    parsed: list[tuple[str, str]],
) -> str:
    """Renders the labeled attribute-block register.

    Args:
        rng (random.Random): Seeded generator.
        gender (str, optional): Canonical gender code.
        pitch (str): Canonical pitch level.
        speed (str): Canonical speaking-speed level.
        energy (str): Canonical energy level.
        parsed (list): Pairs produced by :func:`_parse_tags`.

    Returns:
        str: Blocks of ``"key: value."`` joined by blank lines.
    """
    routed = _route_tags(rng, parsed)
    gender = _infer_gender(gender, routed)
    gender_word = _gender_word(gender)

    values: dict[str, str] = {}

    if gender is not None:
        base = rng.choice(_GENDER_PHRASES[gender])
        gender_tags = routed.get("gender", [])
        if gender_tags:
            base = rng.choice(_GENDER_FRAMES).format(base=base, d=_describe(rng, gender_tags, 1))
        values["gender"] = base

    values["pitch"] = rng.choice(_APS_PITCH[pitch]).format(g=gender_word)
    values["speed"] = rng.choice(_APS_SPEED[speed])
    values["volume"] = rng.choice(_APS_VOLUME[energy])

    for slot in ("age", "clarity", "fluency", "texture", "tone", "personality"):
        entries = routed.get(slot, [])
        if not entries:
            continue
        descriptor = _describe(rng, entries, _SLOT_MAX_TERMS[slot])
        values[slot] = _capitalize(rng.choice(_SLOT_FRAMES[slot]).format(d=descriptor))

    blocks = [
        f"{key}: {values[key]}."
        for key in APS_KEYS
        if key not in APS_OMITTED_KEYS and values.get(key)
    ]
    return "\n\n".join(blocks)


def _render_dsd(
    rng: random.Random,
    pitch: str,
    speed: str,
    energy: str,
    parsed: list[tuple[str, str]],
) -> str:
    """Renders the second-person direct-speech-description register.

    Args:
        rng (random.Random): Seeded generator.
        pitch (str): Canonical pitch level.
        speed (str): Canonical speaking-speed level.
        energy (str): Canonical energy level.
        parsed (list): Pairs produced by :func:`_parse_tags`.

    Returns:
        str: One or two imperative sentences.
    """
    axes = {
        "pitch": _DSD_PITCH_NP[pitch],
        "speed": _DSD_SPEED_NP[speed],
        "volume": _DSD_VOLUME_NP[energy],
    }
    lead, follow, extra = rng.sample(sorted(axes), 3)

    opening = rng.choice(_DSD_OPENERS).format(np=rng.choice(axes[lead]))
    link = rng.choice(_DSD_LINKS).format(np=rng.choice(axes[follow]))
    if rng.random() < 0.6:
        link += rng.choice(_DSD_EXTRA_LINKS).format(np=rng.choice(axes[extra]))

    styleable = [pair for pair in parsed if _TAG_SLOTS[pair[1]][0] != "gender"]
    if styleable:
        closer = rng.choice(_DSD_CLOSERS).format(d=_describe(rng, [rng.choice(styleable)], 1))
    else:
        closer = rng.choice(_DSD_PLAIN_CLOSERS)

    if rng.random() < 0.5:
        return f"{opening}{link}; {closer}."
    return f"{opening}{link}. {_capitalize(closer)}."


def _render_rp(
    rng: random.Random,
    gender: str | None,
    pitch: str,
    speed: str,
    energy: str,
) -> str:
    """Renders the role-play register as either a caption or a brief.

    Args:
        rng (random.Random): Seeded generator.
        gender (str, optional): Canonical gender code.
        pitch (str): Canonical pitch level.
        speed (str): Canonical speaking-speed level.
        energy (str): Canonical energy level.

    Returns:
        str: A single role-and-scenario sentence.
    """
    coarse_pitch = _COARSE_LEVELS[pitch]
    coarse_speed = _COARSE_LEVELS[speed]
    coarse_energy = _COARSE_LEVELS[energy]

    compatible = [
        role
        for role in _ROLES
        if coarse_pitch in role.pitch
        and coarse_speed in role.speed
        and coarse_energy in role.energy
        and (gender is None or role.gender is None or role.gender == gender)
    ]
    implicit = [role for role in compatible if not role.names_gender]
    if implicit and (not compatible or rng.random() < RP_GENDER_IMPLICIT_RATE):
        pool = implicit
    else:
        pool = compatible or [role for role in _ROLES if not role.names_gender]

    role = rng.choice(pool)
    pitch_manner = rng.choice(_RP_PITCH_MANNER[coarse_pitch])
    speed_manner = rng.choice(_RP_SPEED_MANNER[coarse_speed])
    energy_manner = rng.choice(_RP_ENERGY_MANNER[coarse_energy])

    if rng.random() < 0.5:
        adjective = rng.choice(_RP_CAPTION_ADJECTIVES)
        modifier = rng.choice(_RP_TRAIT_MODIFIERS)
        trait = rng.choice(role.traits)
        action = rng.choice(role.actions)
        manners = rng.sample([pitch_manner, speed_manner, energy_manner], 2)
        return rng.choice(_RP_CAPTION_FRAMES).format(
            adj=adjective,
            noun=role.noun,
            action=action,
            mod=modifier,
            trait=trait,
            m1=manners[0],
            m2=manners[1],
        )

    opener = rng.choice(_RP_BRIEF_OPENERS)
    first, second = rng.sample(list(role.traits), 2)
    voice = rng.choice(_RP_VOICE_CLAUSES[coarse_energy])
    manner = rng.choice([pitch_manner, speed_manner])
    return (
        f"{opener} the {first} and {second} of {_article(role.noun)} {role.noun}, "
        f"your voice {voice}, {manner}."
    )


def sample_view(
    epoch: int,
    idx: int,
    weights: Mapping[PromptView | str, float] | None = None,
) -> PromptView:
    """Draws the prompt register for one sample.

    Args:
        epoch (int): Current training epoch.
        idx (int): Global index of the sample.
        weights (Mapping, optional): Unnormalized mixing weights per view.
            Defaults to :data:`DEFAULT_VIEW_WEIGHTS`.

    Returns:
        PromptView: The selected register.

    Raises:
        ValueError: If every supplied weight is zero or negative.
    """
    table = DEFAULT_VIEW_WEIGHTS if weights is None else {
        _coerce_view(view): float(weight) for view, weight in weights.items()
    }
    pairs = [(view, weight) for view, weight in table.items() if weight > 0.0]
    if not pairs:
        raise ValueError("At least one prompt view must carry a positive weight.")

    total = sum(weight for _, weight in pairs)
    threshold = _rng(epoch, idx, _SELECT_SALT).random() * total
    cumulative = 0.0
    for view, weight in pairs:
        cumulative += weight
        if threshold < cumulative:
            return view
    return pairs[-1][0]


def render(
    view: PromptView | str,
    *,
    gender: str | None = None,
    pitch: str | None = None,
    speed: str | None = None,
    energy: str | None = None,
    tags: Sequence[str] | str | None = None,
    style_variant: str | Sequence[str] | None = None,
    epoch: int = 0,
    idx: int = 0,
) -> str:
    """Renders one LibriTTS-P sample in the requested instruction register.

    Args:
        view (PromptView or str): Target register.
        gender (str, optional): ``"M"``/``"F"`` or a spelled-out equivalent.
        pitch (str, optional): One of ``"very low"`` through ``"very high"``.
        speed (str, optional): One of ``"very slow"`` through ``"very fast"``.
        energy (str, optional): One of ``"very low"`` through ``"very high"``.
        tags (Sequence[str] or str, optional): df1 speaker tags.
        style_variant (str or Sequence[str], optional): Style sentence for the raw
            register, or the candidate list to draw one from.
        epoch (int, optional): Current training epoch. (default: ``0``)
        idx (int, optional): Global index of the sample. (default: ``0``)

    Returns:
        str: The rendered prompt.
    """
    view = _coerce_view(view)
    rng = _rng(epoch, idx, _VIEW_SALTS[view])

    if view is PromptView.RAW:
        return _render_raw(rng, tags, style_variant)

    gender_code = _normalize_gender(gender)
    pitch_level = _normalize_level(pitch, _PITCH_ALIASES)
    speed_level = _normalize_level(speed, _SPEED_ALIASES)
    energy_level = _normalize_level(energy, _PITCH_ALIASES)
    parsed = _parse_tags(tags)

    if view is PromptView.APS:
        return _render_aps(rng, gender_code, pitch_level, speed_level, energy_level, parsed)
    if view is PromptView.DSD:
        return _render_dsd(rng, pitch_level, speed_level, energy_level, parsed)
    return _render_rp(rng, gender_code, pitch_level, speed_level, energy_level)


def build_prompt(
    sample: Mapping[str, Any],
    epoch: int,
    idx: int,
    weights: Mapping[PromptView | str, float] | None = None,
) -> tuple[str, PromptView]:
    """Draws a register for a dataset sample and renders it.

    Args:
        sample (Mapping): Dataset row exposing any of ``gender``, ``pitch``,
            ``speaking_speed``/``speed``, ``energy``, ``speaker_prompts``/``tags``,
            and ``style_prompts``/``style_variant``.
        epoch (int): Current training epoch.
        idx (int): Global index of the sample.
        weights (Mapping, optional): Unnormalized mixing weights per view.
            Defaults to :data:`DEFAULT_VIEW_WEIGHTS`.

    Returns:
        tuple: The rendered prompt and the register it was rendered in.
    """
    view = sample_view(epoch, idx, weights)

    style_variant = sample.get("style_variant")
    if style_variant is None:
        style_variant = sample.get("style_prompts")
    if style_variant is None:
        style_variant = sample.get("style_prompt")

    tags = sample.get("tags")
    if tags is None:
        tags = sample.get("speaker_prompts")

    prompt = render(
        view,
        gender=sample.get("gender"),
        pitch=sample.get("pitch"),
        speed=sample.get("speed", sample.get("speaking_speed")),
        energy=sample.get("energy"),
        tags=tags,
        style_variant=style_variant,
        epoch=epoch,
        idx=idx,
    )

    if view is PromptView.RAW and not prompt.strip("."):
        fallback = sample.get("combined_prompt")
        if fallback:
            return str(fallback), view
    return prompt, view


__all__ = [
    "APS_KEYS",
    "APS_OMITTED_KEYS",
    "DEFAULT_VIEW_WEIGHTS",
    "PromptView",
    "RP_GENDER_IMPLICIT_RATE",
    "build_prompt",
    "render",
    "sample_view",
]
