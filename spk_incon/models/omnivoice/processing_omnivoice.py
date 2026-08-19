"""Processor class for OmniVoice."""

import difflib
import re
import unicodedata
from bisect import bisect_left
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging as hf_logging


logger = hf_logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Language name/id resolution
# ---------------------------------------------------------------------------

LANG_NAME_TO_ID = {
    "abadi": "kbt",
    "abkhazian": "ab",
    "abron": "abr",
    "abua": "abn",
    "adamawa fulfulde": "fub",
    "adyghe": "ady",
    "afade": "aal",
    "afrikaans": "af",
    "agwagwune": "yay",
    "aja (benin)": "ajg",
    "akebu": "keu",
    "alago": "ala",
    "albanian": "sq",
    "algerian arabic": "arq",
    "algerian saharan arabic": "aao",
    "ambo-pasco quechua": "qva",
    "ambonese malay": "abs",
    "amdo tibetan": "adx",
    "amharic": "am",
    "anaang": "anw",
    "angika": "anp",
    "antankarana malagasy": "xmv",
    "aragonese": "an",
    "arbëreshë albanian": "aae",
    "arequipa-la unión quechua": "qxu",
    "armenian": "hy",
    "ashe": "ahs",
    "ashéninka perené": "prq",
    "askopan": "eiv",
    "assamese": "as",
    "asturian": "ast",
    "atayal": "tay",
    "awak": "awo",
    "ayacucho quechua": "quy",
    "azerbaijani": "az",
    "baatonum": "bba",
    "bacama": "bcy",
    "bade": "bde",
    "bafia": "ksf",
    "bafut": "bfd",
    "bagirmi fulfulde": "fui",
    "bago-kusuntu": "bqg",
    "baharna arabic": "abv",
    "bakoko": "bkh",
    "balanta-ganja": "bjt",
    "balti": "bft",
    "bamenyam": "bce",
    "bamun": "bax",
    "bangwinji": "bsj",
    "banjar": "bjn",
    "bankon": "abb",
    "baoulé": "bci",
    "bara malagasy": "bhr",
    "barok": "bjk",
    "basa (cameroon)": "bas",
    "basa (nigeria)": "bzw",
    "bashkir": "ba",
    "basque": "eu",
    "batak mandailing": "btm",
    "batanga": "bnm",
    "bateri": "btv",
    "bats": "bbl",
    "bayot": "bda",
    "bebele": "beb",
    "belarusian": "be",
    "bengali": "bn",
    "betawi": "bew",
    "bhili": "bhb",
    "bhojpuri": "bho",
    "bilur": "bxf",
    "bima": "bhp",
    "bodo": "brx",
    "boghom": "bux",
    "bokyi": "bky",
    "bomu": "bmq",
    "bondei": "bou",
    "borgu fulfulde": "fue",
    "bosnian": "bs",
    "brahui": "brh",
    "braj": "bra",
    "breton": "br",
    "buduma": "bdm",
    "buginese": "bug",
    "bukharic": "bhh",
    "bulgarian": "bg",
    "bulu (cameroon)": "bum",
    "bundeli": "bns",
    "bunun": "bnn",
    "bura-pabir": "bwr",
    "burak": "bys",
    "burmese": "my",
    "burushaski": "bsk",
    "cacaloxtepec mixtec": "miu",
    "cajatambo north lima quechua": "qvl",
    "cakfem-mushere": "cky",
    "cameroon pidgin": "wes",
    "campidanese sardinian": "sro",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "cen": "cen",
    "central kurdish": "ckb",
    "central nahuatl": "nhn",
    "central pame": "pbs",
    "central pashto": "pst",
    "central puebla nahuatl": "ncx",
    "central tarahumara": "tar",
    "central yupik": "esu",
    "central-eastern niger fulfulde": "fuq",
    "chadian arabic": "shu",
    "chichewa": "ny",
    "chichicapan zapotec": "zpv",
    "chiga": "cgg",
    "chimalapa zoque": "zoh",
    "chimborazo highland quichua": "qug",
    "chinese": "zh",
    "chiquián ancash quechua": "qxa",
    "chitwania tharu": "the",
    "chokwe": "cjk",
    "chuvash": "cv",
    "cibak": "ckl",
    "coastal konjo": "kjc",
    "copainalá zoque": "zoc",
    "cornish": "kw",
    "corongo ancash quechua": "qwa",
    "croatian": "hr",
    "cross river mbembe": "mfn",
    "cuyamecalco mixtec": "xtu",
    "czech": "cs",
    "dadiya": "dbd",
    "dagbani": "dag",
    "dameli": "dml",
    "danish": "da",
    "dargwa": "dar",
    "dazaga": "dzg",
    "deccan": "dcc",
    "degema": "deg",
    "dera (nigeria)": "kna",
    "dghwede": "dgh",
    "dhatki": "mki",
    "dhivehi": "dv",
    "dhofari arabic": "adf",
    "dijim-bwilim": "cfa",
    "dogri": "dgo",
    "domaaki": "dmk",
    "dotyali": "dty",
    "duala": "dua",
    "dutch": "nl",
    "dũya": "ldb",
    "dyula": "dyu",
    "eastern balochi": "bgp",
    "eastern bolivian guaraní": "gui",
    "eastern egyptian bedawi arabic": "avl",
    "eastern krahn": "kqo",
    "eastern mari": "mhr",
    "eastern yiddish": "ydd",
    "ebrié": "ebr",
    "eggon": "ego",
    "egyptian arabic": "arz",
    "ejagham": "etu",
    "eleme": "elm",
    "eloyi": "afo",
    "embu": "ebu",
    "english": "en",
    "erzya": "myv",
    "esan": "ish",
    "esperanto": "eo",
    "estonian": "et",
    "eton (cameroon)": "eto",
    "ewondo": "ewo",
    "extremaduran": "ext",
    "fang (equatorial guinea)": "fan",
    "fanti": "fat",
    "farefare": "gur",
    "fe'fe'": "fmp",
    "filipino": "fil",
    "filomena mata-coahuitlán totonac": "tlp",
    "finnish": "fi",
    "fipa": "fip",
    "french": "fr",
    "fulah": "ff",
    "galician": "gl",
    "gambian wolof": "wof",
    "ganda": "lg",
    "garhwali": "gbm",
    "gawar-bati": "gwt",
    "gawri": "gwc",
    "gbagyi": "gbr",
    "gbari": "gby",
    "geji": "gyz",
    "gen": "gej",
    "georgian": "ka",
    "german": "de",
    "geser-gorom": "ges",
    "gheg albanian": "aln",
    "ghomálá'": "bbj",
    "gidar": "gid",
    "glavda": "glw",
    "goan konkani": "gom",
    "goaria": "gig",
    "goemai": "ank",
    "gola": "gol",
    "greek": "el",
    "guarani": "gn",
    "guduf-gava": "gdf",
    "guerrero amuzgo": "amu",
    "gujarati": "gu",
    "gujari": "gju",
    "gulf arabic": "afb",
    "gurgula": "ggg",
    "gusii": "guz",
    "gusilay": "gsl",
    "gweno": "gwe",
    "güilá zapotec": "ztu",
    "hadothi": "hoj",
    "hahon": "hah",
    "haitian": "ht",
    "hakha chin": "cnh",
    "hakö": "hao",
    "halia": "hla",
    "hausa": "ha",
    "hawaiian": "haw",
    "hazaragi": "haz",
    "hebrew": "he",
    "hemba": "hem",
    "herero": "hz",
    "highland konjo": "kjk",
    "hijazi arabic": "acw",
    "hindi": "hi",
    "huarijio": "var",
    "huautla mazatec": "mau",
    "huaxcaleca nahuatl": "nhq",
    "huba": "hbb",
    "huitepec mixtec": "mxs",
    "hula": "hul",
    "hungarian": "hu",
    "hunjara-kaina ke": "hkk",
    "hwana": "hwo",
    "ibibio": "ibb",
    "icelandic": "is",
    "idakho-isukha-tiriki": "ida",
    "idoma": "idu",
    "igbo": "ig",
    "igo": "ahl",
    "ikposo": "kpo",
    "ikwere": "ikw",
    "imbabura highland quichua": "qvi",
    "indonesian": "id",
    "indus kohistani": "mvy",
    "interlingua (international auxiliary language association)": "ia",
    "inupiaq": "ik",
    "irish": "ga",
    "iron ossetic": "os",
    "isekiri": "its",
    "isoko": "iso",
    "italian": "it",
    "ito": "itw",
    "itzá": "itz",
    "ixtayutla mixtec": "vmj",
    "izon": "ijc",
    "jambi malay": "jax",
    "japanese": "ja",
    "jaqaru": "jqr",
    "jauja wanca quechua": "qxw",
    "jaunsari": "jns",
    "javanese": "jv",
    "jiba": "juo",
    "jju": "kaj",
    "judeo-moroccan arabic": "aju",
    "juxtlahuaca mixtec": "vmc",
    "kabardian": "kbd",
    "kabras": "lkb",
    "kabuverdianu": "kea",
    "kabyle": "kab",
    "kachi koli": "gjk",
    "kairak": "ckr",
    "kalabari": "ijn",
    "kalasha": "kls",
    "kalenjin": "kln",
    "kalkoti": "xka",
    "kamba": "kam",
    "kamo": "kcq",
    "kanauji": "bjj",
    "kanembu": "kbl",
    "kannada": "kn",
    "karekare": "kai",
    "kashmiri": "ks",
    "kathoriya tharu": "tkt",
    "kati": "bsh",
    "kazakh": "kk",
    "keiyo": "eyo",
    "khams tibetan": "khg",
    "khana": "ogo",
    "khetrani": "xhe",
    "khmer": "km",
    "khowar": "khw",
    "kinga": "zga",
    "kinnauri": "kfk",
    "kinyarwanda": "rw",
    "kirghiz": "ky",
    "kirya-konzəl": "fkk",
    "kochila tharu": "thq",
    "kohistani shina": "plk",
    "kohumono": "bcs",
    "kok borok": "trp",
    "kol (papua new guinea)": "kol",
    "kom (cameroon)": "bkm",
    "koma": "kmy",
    "konkani": "knn",
    "konzo": "koo",
    "korean": "ko",
    "korwa": "kfp",
    "kota (india)": "kfe",
    "koti": "eko",
    "kuanua": "ksd",
    "kuanyama": "kj",
    "kui (india)": "uki",
    "kulung (nigeria)": "bbu",
    "kuot": "kto",
    "kushi": "kuh",
    "kwambi": "kwm",
    "kwasio": "nmg",
    "lala-roba": "lla",
    "lamang": "hia",
    "lao": "lo",
    "larike-wakasihu": "alo",
    "lasi": "lss",
    "latgalian": "ltg",
    "latvian": "lv",
    "levantine arabic": "apc",
    "liana-seti": "ste",
    "liberia kpelle": "xpe",
    "liberian english": "lir",
    "libyan arabic": "ayl",
    "ligurian": "lij",
    "lijili": "mgi",
    "lingala": "ln",
    "lithuanian": "lt",
    "loarki": "lrk",
    "logooli": "rag",
    "logudorese sardinian": "src",
    "loja highland quichua": "qvj",
    "loloda": "loa",
    "longuda": "lnu",
    "loxicha zapotec": "ztp",
    "luba-lulua": "lua",
    "luo": "luo",
    "lushai": "lus",
    "luxembourgish": "lb",
    "maasina fulfulde": "ffm",
    "maba (chad)": "mde",
    "macedo-romanian": "rup",
    "macedonian": "mk",
    "mada (cameroon)": "mxu",
    "mafa": "maf",
    "maithili": "mai",
    "malay": "ms",
    "malayalam": "ml",
    "mali": "gcc",
    "malinaltepec me'phaa": "tcf",
    "maltese": "mt",
    "mandara": "tbf",
    "mandjak": "mfv",
    "manggarai": "mqy",
    "manipuri": "mni",
    "mansoanka": "msw",
    "manx": "gv",
    "maori": "mi",
    "marathi": "mr",
    "marghi central": "mrt",
    "marghi south": "mfm",
    "maria (india)": "mrr",
    "marwari (pakistan)": "mve",
    "masana": "mcn",
    "masikoro malagasy": "msh",
    "matsés": "mcf",
    "mazaltepec zapotec": "zpy",
    "mazatlán mazatec": "vmz",
    "mazatlán mixe": "mzl",
    "mbe": "mfo",
    "mbo (cameroon)": "mbo",
    "mbum": "mdd",
    "medumba": "byv",
    "mekeo": "mek",
    "meru": "mer",
    "mesopotamian arabic": "acm",
    "mewari": "mtr",
    "min nan chinese": "nan",
    "mingrelian": "xmf",
    "mitlatongo mixtec": "vmm",
    "miya": "mkf",
    "mokpwe": "bri",
    "moksha": "mdf",
    "mom jango": "ver",
    "mongolian": "mn",
    "moroccan arabic": "ary",
    "motu": "meu",
    "mpiemo": "mcx",
    "mpumpong": "mgg",
    "mundang": "mua",
    "mungaka": "mhk",
    "musey": "mse",
    "musgu": "mug",
    "musi": "mui",
    "naba": "mne",
    "najdi arabic": "ars",
    "nalik": "nal",
    "nawdm": "nmz",
    "ndonga": "ng",
    "neapolitan": "nap",
    "nepali": "npi",
    "ngamo": "nbh",
    "ngas": "anc",
    "ngiemboon": "nnh",
    "ngizim": "ngi",
    "ngomba": "jgo",
    "ngombale": "nla",
    "nigerian fulfulde": "fuv",
    "nigerian pidgin": "pcm",
    "nimadi": "noe",
    "nobiin": "fia",
    "north mesopotamian arabic": "ayp",
    "north moluccan malay": "max",
    "northern betsimisaraka malagasy": "bmm",
    "northern hindko": "hno",
    "northern kurdish": "kmr",
    "northern pame": "pmq",
    "northern pashto": "pbu",
    "northern uzbek": "uzn",
    "northwest gbaya": "gya",
    "norwegian": "no",
    "norwegian bokmål": "nb",
    "norwegian nynorsk": "nn",
    "notsi": "ncf",
    "nyankpa": "yes",
    "nyungwe": "nyu",
    "nzanyi": "nja",
    "nüpode huitoto": "hux",
    "occitan": "oc",
    "od": "odk",
    "odia": "ory",
    "odual": "odu",
    "omani arabic": "acx",
    "orizaba nahuatl": "nlv",
    "orma": "orc",
    "ormuri": "oru",
    "oromo": "om",
    "pahari-potwari": "phr",
    "paiwan": "pwn",
    "panjabi": "pa",
    "papuan malay": "pmy",
    "parkari koli": "kvx",
    "pedi": "nso",
    "pero": "pip",
    "persian": "fa",
    "petats": "pex",
    "phalura": "phl",
    "piemontese": "pms",
    "piya-kwonci": "piy",
    "plateau malagasy": "plt",
    "polish": "pl",
    "poqomam": "poc",
    "portuguese": "pt",
    "pulaar": "fuc",
    "pular": "fuf",
    "puno quechua": "qxp",
    "pushto": "ps",
    "pökoot": "pko",
    "qaqet": "byx",
    "quiotepec chinantec": "chq",
    "rana tharu": "thr",
    "rangi": "lag",
    "rapoisi": "kyx",
    "ratahan": "rth",
    "rayón zoque": "zor",
    "romanian": "ro",
    "romansh": "rm",
    "rombo": "rof",
    "rotokas": "roo",
    "rukai": "dru",
    "russian": "ru",
    "sacapulteco": "quv",
    "saidi arabic": "aec",
    "sakalava malagasy": "skg",
    "sakizaya": "szy",
    "saleman": "sau",
    "samba daka": "ccg",
    "samba leko": "ndi",
    "san felipe otlaltepec popoloca": "pow",
    "san francisco del mar huave": "hue",
    "san juan atzingo popoloca": "poe",
    "san martín itunyoso triqui": "trq",
    "san miguel el grande mixtec": "mig",
    "sansi": "ssi",
    "sanskrit": "sa",
    "santa ana de tusi pasco quechua": "qxt",
    "santa catarina albarradas zapotec": "ztn",
    "santali": "sat",
    "santiago del estero quichua": "qus",
    "saposa": "sps",
    "saraiki": "skr",
    "sardinian": "sc",
    "saya": "say",
    "sediq": "trv",
    "serbian": "sr",
    "seri": "sei",
    "shina": "scl",
    "shona": "sn",
    "siar-lak": "sjr",
    "sibe": "nco",
    "sicilian": "scn",
    "sihuas ancash quechua": "qws",
    "sikkimese": "sip",
    "sinaugoro": "snc",
    "sindhi": "sd",
    "sindhi bhil": "sbn",
    "sinhala": "si",
    "sinicahua mixtec": "xti",
    "sipacapense": "qum",
    "siwai": "siw",
    "slovak": "sk",
    "slovenian": "sl",
    "solos": "sol",
    "somali": "so",
    "soninke": "snk",
    "south giziga": "giz",
    "south ucayali ashéninka": "cpy",
    "southeastern nochixtlán mixtec": "mxy",
    "southern betsimisaraka malagasy": "bzc",
    "southern pashto": "pbt",
    "southern pastaza quechua": "qup",
    "soyaltepec mazatec": "vmp",
    "spanish": "es",
    "standard arabic": "arb",
    "standard moroccan tamazight": "zgh",
    "sudanese arabic": "apd",
    "sulka": "sua",
    "svan": "sva",
    "swahili": "sw",
    "swedish": "sv",
    "tae'": "rob",
    "tahaggart tamahaq": "thv",
    "taita": "dav",
    "tajik": "tg",
    "tamil": "ta",
    "tandroy-mahafaly malagasy": "tdx",
    "tangale": "tan",
    "tanosy malagasy": "txy",
    "tarok": "yer",
    "tatar": "tt",
    "tedaga": "tuq",
    "telugu": "te",
    "tem": "kdh",
    "teop": "tio",
    "tepeuxila cuicatec": "cux",
    "tepinapa chinantec": "cte",
    "tera": "ttr",
    "terei": "buo",
    "termanu": "twu",
    "tesaka malagasy": "tkg",
    "tetelcingo nahuatl": "nhg",
    "teutila cuicatec": "cut",
    "thai": "th",
    "tibetan": "bo",
    "tidaá mixtec": "mtx",
    "tidore": "tvo",
    "tigak": "tgc",
    "tigre": "tig",
    "tigrinya": "ti",
    "tilquiapan zapotec": "zts",
    "tinputz": "tpz",
    "tlacoapa me'phaa": "tpl",
    "tlacoatzintepec chinantec": "ctl",
    "tlingit": "tli",
    "toki pona": "tok",
    "tomoip": "tqp",
    "tondano": "tdn",
    "tonsea": "txs",
    "tooro": "ttj",
    "torau": "ttu",
    "torwali": "trw",
    "tsimihety malagasy": "xmw",
    "tsotso": "lto",
    "tswana": "tn",
    "tugen": "tuy",
    "tuki": "bag",
    "tula": "tul",
    "tulu": "tcy",
    "tunen": "tvu",
    "tungag": "lcm",
    "tunisian arabic": "aeb",
    "tupuri": "tui",
    "turkana": "tuv",
    "turkish": "tr",
    "turkmen": "tk",
    "tututepec mixtec": "mtu",
    "twi": "tw",
    "ubaghara": "byc",
    "uighur": "ug",
    "ukrainian": "uk",
    "umbundu": "umb",
    "upper sorbian": "hsb",
    "urdu": "ur",
    "ushojo": "ush",
    "uzbek": "uz",
    "vai": "vai",
    "vietnamese": "vi",
    "votic": "vot",
    "võro": "vro",
    "waci gbe": "wci",
    "wadiyara koli": "kxp",
    "waja": "wja",
    "wakhi": "wbl",
    "wanga": "lwg",
    "wapan": "juk",
    "warji": "wji",
    "welsh": "cy",
    "wemale": "weo",
    "western frisian": "fy",
    "western highland purepecha": "pua",
    "western juxtlahuaca mixtec": "jmx",
    "western maninkakan": "mlq",
    "western mari": "mrj",
    "western niger fulfulde": "fuh",
    "western panjabi": "pnb",
    "wolof": "wo",
    "wuzlam": "udl",
    "xanaguía zapotec": "ztg",
    "xhosa": "xh",
    "yace": "ekr",
    "yakut": "sah",
    "yalahatan": "jal",
    "yanahuanca pasco quechua": "qur",
    "yangben": "yav",
    "yaqui": "yaq",
    "yauyos quechua": "qux",
    "yekhee": "ets",
    "yiddish": "yi",
    "yidgha": "ydg",
    "yoruba": "yo",
    "yutanduchi mixtec": "mab",
    "zacatlán-ahuacatlán-tepetzintla nahuatl": "nhi",
    "zarma": "dje",
    "zaza": "zza",
    "zulu": "zu",
    "ömie": "aom",
}

LANG_NAMES = set(LANG_NAME_TO_ID.keys())
LANG_IDS = set(LANG_NAME_TO_ID.values())

# Exceptions where .title() doesn't match the canonical casing from the TSV.
_TITLE_EXCEPTIONS = {
    "fe'fe'": "Fe'fe'",
    "dũya": "Dũya",
    "santiago del estero quichua": "Santiago del Estero Quichua",
    "santa ana de tusi pasco quechua": "Santa Ana de Tusi Pasco Quechua",
    "malinaltepec me'phaa": "Malinaltepec Me'phaa",
    "tlacoapa me'phaa": "Tlacoapa Me'phaa",
}


def lang_display_name(name: str) -> str:
    """Return a display-friendly version of a lowercase language name.

    Uses .title() for most names, with manual exceptions for cases like
    apostrophes and small words (de, del) that should stay lowercase.
    """
    return _TITLE_EXCEPTIONS.get(name, name.title())


def resolve_language(language: Optional[str]) -> Optional[str]:
    """Resolve a language name or code to a canonical language id.

    Args:
        language: A language id (e.g. `"en"`), a language name (e.g. `"English"`), `"None"`, or `None`.

    Returns:
        The canonical language id, or `None` for language-agnostic mode.
    """
    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    logger.warning(
        f"Language '{language}' is not recognized. "
        f"Please use a valid language ID (e.g., 'en', 'zh', 'ja', 'de') "
        f"or a full language name (e.g., 'English', 'Chinese', 'Japanese'). "
        f"Falling back to None (language-agnostic mode)."
    )
    return None


# ---------------------------------------------------------------------------
# Voice-design instruct constants and validation
# ---------------------------------------------------------------------------

_ZH_RE = re.compile(r"[一-鿿]")

# Category = set of {english: chinese, ...} items that are mutually exclusive.
# Accent (EN-only) and dialect (ZH-only) are stored as flat sets below.
_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
    # Accent (English-only, no Chinese counterpart)
    {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    },
    # Dialect (Chinese-only, no English counterpart)
    {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    },
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
_INSTRUCT_MUTUALLY_EXCLUSIVE = []
for _cat in _INSTRUCT_CATEGORIES:
    if isinstance(_cat, dict):
        _INSTRUCT_EN_TO_ZH.update(_cat)
        _INSTRUCT_ZH_TO_EN.update({v: k for k, v in _cat.items()})
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat) | set(_cat.values()))
    else:
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat))

_INSTRUCT_ALL_VALID = (
    set(_INSTRUCT_EN_TO_ZH)
    | set(_INSTRUCT_ZH_TO_EN)
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-2]  # accents
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-1]  # dialects
)

_INSTRUCT_VALID_EN = frozenset(i for i in _INSTRUCT_ALL_VALID if not _ZH_RE.search(i))
_INSTRUCT_VALID_ZH = frozenset(i for i in _INSTRUCT_ALL_VALID if _ZH_RE.search(i))


def resolve_instruct(instruct: Optional[str], use_zh: bool = False) -> Optional[str]:
    r"""Validate and normalize a voice-design instruct string.

    Supported instruct items (case-insensitive for English):

    English (comma + space separated):
        gender: male, female
        age: child, teenager, young adult, middle-aged, elderly
        pitch: very low pitch, low pitch, moderate pitch, high pitch, very high pitch
        style: whisper
        accent: american accent, british accent, australian accent, ...

    Chinese (full-width comma separated):
        gender: 男, 女
        age: 儿童, 少年, 青年, 中年, 老年
        pitch: 极低音调, 低音调, 中音调, 高音调, 极高音调
        style: 耳语
        dialect: 河南话, 陕西话, 四川话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话

    Args:
        instruct: Raw instruct string, or `None`.
        use_zh: If `True`, normalize all items to Chinese (used when the synthesis text contains
            Chinese and no accent is specified).

    Returns:
        Normalized instruct string, or `None`.

    Raises:
        `ValueError`: If any instruct item is unsupported, misspelled, or two items in the same
            category conflict, or a Chinese dialect and an English accent are mixed.
    """
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]

    unknown = []
    normalized = []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalized.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append(f"  '{raw}' -> '{n}' (unsupported; did you mean '{sug}'?)")
            else:
                lines.append(f"  '{raw}' -> '{n}' (unsupported)")
        err = (
            f"Unsupported instruct items found in {instruct_str}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. 'male, indian accent'),\n"
            "Chinese instructs should use full-width comma (e.g. '男，河南话')."
        )
        raise ValueError(err)

    has_dialect = any(n.endswith("话") for n in normalized)
    has_accent = any(" accent" in n for n in normalized)

    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    if use_zh:
        normalized = [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalized]
    else:
        normalized = [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalized]

    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalized if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = [" vs ".join(f"'{x}'" for x in group) for group in conflicts]
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) allows at most one item."
        )

    has_zh = any(any("一" <= c <= "鿿" for c in n) for n in normalized)
    separator = "，" if has_zh else ", "

    return separator.join(normalized)


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


class _RuleDurationEstimator:
    """Estimates target audio-token counts from text using per-script phonetic weights.

    The weight represents the relative speaking time of a character compared to a standard
    Latin letter (1.0 = one Latin character, roughly 40-50ms).
    """

    def __init__(self):
        self.weights = {
            "cjk": 3.0,
            "hangul": 2.5,
            "kana": 2.2,
            "ethiopic": 3.0,
            "yi": 3.0,
            "indic": 1.8,
            "thai_lao": 1.5,
            "khmer_myanmar": 1.8,
            "arabic": 1.5,
            "hebrew": 1.5,
            "latin": 1.0,
            "cyrillic": 1.0,
            "greek": 1.0,
            "armenian": 1.0,
            "georgian": 1.0,
            "punctuation": 0.5,
            "space": 0.2,
            "digit": 3.5,
            "mark": 0.0,
            "default": 1.0,
        }

        # (end_codepoint, script_key), used for a binary search over Unicode blocks.
        self.ranges = [
            (0x02AF, "latin"),
            (0x03FF, "greek"),
            (0x052F, "cyrillic"),
            (0x058F, "armenian"),
            (0x05FF, "hebrew"),
            (0x077F, "arabic"),
            (0x089F, "arabic"),
            (0x08FF, "arabic"),
            (0x097F, "indic"),
            (0x09FF, "indic"),
            (0x0A7F, "indic"),
            (0x0AFF, "indic"),
            (0x0B7F, "indic"),
            (0x0BFF, "indic"),
            (0x0C7F, "indic"),
            (0x0CFF, "indic"),
            (0x0D7F, "indic"),
            (0x0DFF, "indic"),
            (0x0EFF, "thai_lao"),
            (0x0FFF, "indic"),
            (0x109F, "khmer_myanmar"),
            (0x10FF, "georgian"),
            (0x11FF, "hangul"),
            (0x137F, "ethiopic"),
            (0x139F, "ethiopic"),
            (0x13FF, "default"),
            (0x167F, "default"),
            (0x169F, "default"),
            (0x16FF, "default"),
            (0x171F, "default"),
            (0x173F, "default"),
            (0x175F, "default"),
            (0x177F, "default"),
            (0x17FF, "khmer_myanmar"),
            (0x18AF, "default"),
            (0x18FF, "default"),
            (0x194F, "indic"),
            (0x19DF, "indic"),
            (0x19FF, "khmer_myanmar"),
            (0x1A1F, "indic"),
            (0x1AAF, "indic"),
            (0x1B7F, "indic"),
            (0x1BBF, "indic"),
            (0x1BFF, "indic"),
            (0x1C4F, "indic"),
            (0x1C7F, "indic"),
            (0x1C8F, "cyrillic"),
            (0x1CBF, "georgian"),
            (0x1CCF, "indic"),
            (0x1CFF, "indic"),
            (0x1D7F, "latin"),
            (0x1DBF, "latin"),
            (0x1DFF, "default"),
            (0x1EFF, "latin"),
            (0x309F, "kana"),
            (0x30FF, "kana"),
            (0x312F, "cjk"),
            (0x318F, "hangul"),
            (0x9FFF, "cjk"),
            (0xA4CF, "yi"),
            (0xA4FF, "default"),
            (0xA63F, "default"),
            (0xA69F, "cyrillic"),
            (0xA6FF, "default"),
            (0xA7FF, "latin"),
            (0xA82F, "indic"),
            (0xA87F, "default"),
            (0xA8DF, "indic"),
            (0xA8FF, "indic"),
            (0xA92F, "indic"),
            (0xA95F, "indic"),
            (0xA97F, "hangul"),
            (0xA9DF, "indic"),
            (0xA9FF, "khmer_myanmar"),
            (0xAA5F, "indic"),
            (0xAA7F, "khmer_myanmar"),
            (0xAADF, "indic"),
            (0xAAFF, "indic"),
            (0xAB2F, "ethiopic"),
            (0xAB6F, "latin"),
            (0xABBF, "default"),
            (0xABFF, "indic"),
            (0xD7AF, "hangul"),
            (0xFAFF, "cjk"),
            (0xFDFF, "arabic"),
            (0xFE6F, "default"),
            (0xFEFF, "arabic"),
            (0xFFEF, "latin"),
        ]
        self.breakpoints = [r[0] for r in self.ranges]

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char):
        code = ord(char)
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]
        if code == 32:
            return self.weights["space"]
        if code == 0x0640:  # Arabic Tatweel
            return self.weights["mark"]

        category = unicodedata.category(char)
        if category.startswith("M"):
            return self.weights["mark"]
        if category.startswith("P") or category.startswith("S"):
            return self.weights["punctuation"]
        if category.startswith("Z"):
            return self.weights["space"]
        if category.startswith("N"):
            return self.weights["digit"]

        idx = bisect_left(self.breakpoints, code)
        if idx < len(self.ranges):
            script_type = self.ranges[idx][1]
            return self.weights.get(script_type, self.weights["default"])

        if code > 0x20000:
            return self.weights["cjk"]
        return self.weights["default"]

    def calculate_total_weight(self, text):
        return sum(self._get_char_weight(c) for c in text)

    def estimate_duration(
        self,
        target_text: str,
        ref_text: str,
        ref_duration: float,
        low_threshold: Optional[float] = 50,
        boost_strength: float = 3,
    ) -> float:
        """Estimate the audio-token duration `target_text` would take, given a reference pair.

        Args:
            target_text: The text to estimate the duration for.
            ref_text: The reference text that was used to measure `ref_duration`.
            ref_duration: The actual duration it took to speak `ref_text`.
            low_threshold: Minimum duration threshold below which the estimate is boosted
                (short references otherwise underestimate speaking time).
            boost_strength: Controls the power-curve boost applied below `low_threshold`.
                Higher values boost small durations more aggressively; 1 = no boost.

        Returns:
            The estimated duration for `target_text`, in the same units as `ref_duration`.
        """
        if ref_duration <= 0 or not ref_text:
            return 0.0

        ref_weight = self.calculate_total_weight(ref_text)
        if ref_weight == 0:
            return 0.0

        speed_factor = ref_weight / ref_duration
        target_weight = self.calculate_total_weight(target_text)

        estimated_duration = target_weight / speed_factor
        if low_threshold is not None and estimated_duration < low_threshold:
            alpha = 1.0 / boost_strength
            return low_threshold * (estimated_duration / low_threshold) ** alpha
        return estimated_duration


# ---------------------------------------------------------------------------
# Text chunking and punctuation
# ---------------------------------------------------------------------------

_SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
_CLOSING_MARKS = set("\"'“”‘’）]》>」】")

_END_PUNCTUATION = {
    ";", ":", ",", ".", "!", "?", "…", ")", "]", "}", '"', "'", "“", "”", "‘", "’",
    "；", "：", "，", "。", "！", "？", "、", "……", "）", "】",
}

_ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.",
    "Pres.", "Gov.", "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.",
    "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.",
    "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.",
    "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.",
    "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
}


def chunk_text_punctuation(text: str, chunk_len: int, min_chunk_len: Optional[int] = None) -> list[str]:
    """Split `text` into chunks at punctuation boundaries, skipping common abbreviations.

    Args:
        text: The text to split.
        chunk_len: Maximum number of characters per merged chunk.
        min_chunk_len: Chunks shorter than this are merged into a neighboring chunk.

    Returns:
        The list of chunk strings.
    """
    sentences = []
    current_sentence = []
    tokens_list = list(text)

    for token in tokens_list:
        if len(current_sentence) == 0 and len(sentences) != 0 and (
            token in _SPLIT_PUNCTUATION or token in _CLOSING_MARKS
        ):
            sentences[-1].append(token)
        else:
            current_sentence.append(token)
            if token in _SPLIT_PUNCTUATION:
                is_abbreviation = False
                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str:
                        last_word = temp_str.split()[-1]
                        if last_word in _ABBREVIATIONS:
                            is_abbreviation = True
                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence
    if len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    if min_chunk_len is not None:
        first_chunk_short_flag = len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short_flag:
                final_chunks[-1].extend(chunk)
            elif len(chunk) >= min_chunk_len:
                final_chunks.append(chunk)
            elif len(final_chunks) == 0:
                final_chunks.append(chunk)
            else:
                final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    return ["".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()]


def add_punctuation(text: str) -> str:
    """Append missing end punctuation (Chinese or English) to `text`."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in _END_PUNCTUATION:
        is_chinese = any("一" <= char <= "鿿" for char in text)
        text += "。" if is_chinese else "."
    return text


# ---------------------------------------------------------------------------
# Optional text normalization (numbers, dates, currency, etc.)
# ---------------------------------------------------------------------------
#
# The OmniVoice inline control syntax must survive normalization: bracketed
# non-verbal tags (e.g. `[laughter]`), bracketed CMU pronunciation overrides
# (e.g. `[B EY1 S]`), and Chinese pinyin tone markers (uppercase pinyin + tone
# digit) are held out and reinserted verbatim around normalization.

_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")
_PINYIN_TONE_RE = re.compile(r"[A-Z]+[1-5]")
_CJK_RE = re.compile(r"[一-鿿]")

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

_TN_INSTALL_MSG = (
    "Text normalization (normalize_text=True) requires WeTextProcessing, which is not installed.\n"
    "  pip install WeTextProcessing         # or:  pip install 'omnivoice[tn]'\n"
    "WeTextProcessing depends on pynini, which has no prebuilt wheel for macOS arm64. On macOS, "
    "install pynini from conda-forge first:\n"
    "  conda install -c conda-forge pynini\n"
    "then:  pip install WeTextProcessing"
)

_ZH_NORMALIZER = None
_EN_NORMALIZER = None


def _get_zh_normalizer():
    global _ZH_NORMALIZER
    if _ZH_NORMALIZER is None:
        try:
            from tn.chinese.normalizer import Normalizer
        except ImportError as e:
            raise ImportError(_TN_INSTALL_MSG) from e
        _ZH_NORMALIZER = Normalizer(
            remove_interjections=False,
            remove_erhua=False,
            traditional_to_simple=False,
            remove_puncts=False,
            full_to_half=False,
        )
    return _ZH_NORMALIZER


def _get_en_normalizer():
    global _EN_NORMALIZER
    if _EN_NORMALIZER is None:
        try:
            from tn.english.normalizer import Normalizer
        except ImportError as e:
            raise ImportError(_TN_INSTALL_MSG) from e
        _EN_NORMALIZER = Normalizer()
    return _EN_NORMALIZER


def _resolve_lang_code(language: Optional[str], text: str) -> str:
    if language is not None:
        code = language.strip().lower()
        if code and code != "none":
            if code in ("zh", "en"):
                return code
            if code in LANG_IDS:
                return code
            if code in LANG_NAME_TO_ID:
                return LANG_NAME_TO_ID[code]
            return code
    return "zh" if _CJK_RE.search(text) else "en"


def _num2words_segment(text: str, lang: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text

    def _repl(match):
        try:
            return num2words(int(match.group()), lang=lang)
        except Exception:
            return match.group()

    return re.sub(r"\d+", _repl, text)


def _normalize_segment(fn, segment: str) -> str:
    if not segment.strip():
        return segment
    lead = segment[: len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()) :]
    try:
        core = fn(segment.strip())
    except Exception as e:
        logger.warning(
            "Text normalization failed on a segment (%s); keeping it unchanged.",
            type(e).__name__,
        )
        return segment
    return lead + core + trail


def _apply_with_protection(text: str, fn, protect_pinyin: bool) -> str:
    spans = [m.span() for m in _BRACKET_TAG_RE.finditer(text)]
    if protect_pinyin:
        spans += [m.span() for m in _PINYIN_TONE_RE.finditer(text)]
    if not spans:
        return _normalize_segment(fn, text)

    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: list[str] = []
    last = 0
    for start, end in merged:
        if start > last:
            out.append(_normalize_segment(fn, text[last:start]))
        out.append(text[start:end])
        last = end
    if last < len(text):
        out.append(_normalize_segment(fn, text[last:]))
    return "".join(out)


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalize numbers, dates, currency, etc. in `text` into their spoken form.

    Chinese is routed through WeTextProcessing's `ZhNormalizer` and English through its
    `EnNormalizer` (configured to only rewrite numeric/symbolic tokens). Any other language
    falls back to `num2words` for bare integers when it is installed, otherwise the text is
    returned unchanged. Bracketed non-verbal tags, CMU pronunciation overrides, and Chinese
    pinyin tone markers are preserved verbatim.

    Args:
        text: Input text.
        language: Language code (`"en"`/`"zh"`) or full name (`"English"`). `None` auto-detects
            Chinese vs. English by script.

    Returns:
        The normalized text.

    Raises:
        `ImportError`: For Chinese/English when WeTextProcessing is not installed.
    """
    if not text or not text.strip():
        return text

    code = _resolve_lang_code(language, text)
    if code == "zh":
        normalizer = _get_zh_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=True)
    if code == "en":
        normalizer = _get_en_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=False)
    return _apply_with_protection(text, lambda s: _num2words_segment(s, code), protect_pinyin=False)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize `text`, tokenizing non-verbal tags standalone for consistent token ids.

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: A `transformers` text tokenizer instance.

    Returns:
        Token ids tensor of shape `(1, seq_len)`.
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer(text, return_tensors="pt").input_ids
    combined = []
    for p in parts:
        combined.extend(p)
    return torch.tensor([combined], dtype=torch.long)


def _combine_text(text: str, ref_text: Optional[str] = None) -> str:
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    full_text = re.sub(r"[\r\n]+", "", full_text)
    full_text = full_text.replace("（", "(").replace("）", ")")
    full_text = re.sub(r"[ \t]+", " ", full_text)

    chinese_range = r"[一-鿿]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


# ---------------------------------------------------------------------------
# Audio I/O and processing (numpy float32 arrays, shape (C, T))
# ---------------------------------------------------------------------------


def _load_waveform(audio_path: str):
    try:
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sr  # (T, C) -> (C, T)
    except Exception:
        import librosa

        data, sr = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data, sr


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load a waveform from `audio_path` and resample it to `sampling_rate`.

    Returns:
        A numpy float32 array of shape `(1, T)`.
    """
    data, sr = _load_waveform(audio_path)
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate).numpy()
    return data


def _numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def _audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def _remove_silence_edges(audio: AudioSegment, lead_sil: int, trail_sil: int, silence_threshold: float) -> AudioSegment:
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    return audio.reverse()


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Remove middle silences longer than `mid_sil` ms and trim edge silences.

    Args:
        audio: Numpy array of shape `(C, T)`.
        sampling_rate: Sampling rate of `audio`.
        mid_sil: Middle-silence threshold in ms (`0` to skip).
        lead_sil: Leading silence kept, in ms.
        trail_sil: Trailing silence kept, in ms.

    Returns:
        A numpy array of shape `(C, T')`.
    """
    wave = _numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        non_silent_segs = split_on_silence(
            wave, min_silence_len=mid_sil, silence_thresh=-50, keep_silence=mid_sil, seek_step=10
        )
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    wave = _remove_silence_edges(wave, lead_sil, trail_sil, -50)
    return _audiosegment_to_numpy(wave)


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply fade-in/out and pad `audio` with silence to prevent clicks at the edges.

    Args:
        audio: Numpy array of shape `(C, T)`.
        pad_duration: Silence padding duration per side, in seconds.
        fade_duration: Fade curve duration, in seconds.
        sample_rate: Sampling rate of `audio`.

    Returns:
        A numpy array of shape `(C, T_new)`.
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in
            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros((processed.shape[0], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim `audio` to at most `max_duration` seconds by splitting at the largest silence gap.

    Only trims when `audio` exceeds `trim_threshold` seconds.

    Args:
        audio: Numpy array of shape `(C, T)`.
        sampling_rate: Sampling rate of `audio`.
        max_duration: Maximum duration, in seconds.
        min_duration: Minimum duration, in seconds.
        trim_threshold: Only trim if `audio` is longer than this, in seconds.

    Returns:
        The trimmed numpy array.
    """
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    seg = _numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(seg, min_silence_len=100, silence_thresh=-40, seek_step=10)
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    return _audiosegment_to_numpy(trimmed)


def cross_fade_chunks(chunks: list[np.ndarray], sample_rate: int, silence_duration: float = 0.3) -> np.ndarray:
    """Concatenate audio `chunks` with silence gaps and cross-fade at the boundaries.

    Args:
        chunks: List of numpy arrays, each of shape `(C, T)`.
        sample_rate: Sampling rate of the chunks.
        silence_duration: Total silence gap duration between chunks, in seconds.

    Returns:
        A numpy array of shape `(C, T_total)`.
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]

        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
            parts[-1][..., -fout_n:] *= w_out

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
            fade_in[..., :fin_n] *= w_in

        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)

    return merged


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class OmniVoiceProcessor(ProcessorMixin):
    r"""
    Constructs an OmniVoice processor which wraps an `AutoFeatureExtractor`, an `AutoTokenizer`, and a
    [`HiggsAudioV2TokenizerModel`] into a single processor. It builds the interleaved style/text/audio token
    conditioning [`OmniVoiceForConditionalGeneration.generate`] expects, and tokenizes/detokenizes reference and
    generated audio through the audio tokenizer.

    Args:
        feature_extractor (`AutoFeatureExtractor`):
            An instance of `AutoFeatureExtractor`, used to read the sampling rate and hop length the audio
            tokenizer was trained with.
        tokenizer (`AutoTokenizer`):
            An instance of `AutoTokenizer`.
        audio_tokenizer (`HiggsAudioV2TokenizerModel`):
            An instance of [`HiggsAudioV2TokenizerModel`], used to encode reference audio into discrete tokens
            and decode generated tokens back into a waveform.
        num_audio_codebook (`int`, *optional*, defaults to 8):
            Number of parallel audio codebooks the model input/output is repeated/split over.
        audio_mask_id (`int`, *optional*, defaults to 1024):
            Token id used to mark a still-masked audio position. Must match `config.audio_mask_id` of the
            [`OmniVoiceForConditionalGeneration`] this processor is paired with.
    """

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioV2TokenizerModel"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        num_audio_codebook: int = 8,
        audio_mask_id: int = 1024,
        chat_template=None,
    ):
        self.num_audio_codebook = num_audio_codebook
        self.audio_mask_id = audio_mask_id
        self.duration_estimator = _RuleDurationEstimator()
        super().__init__(feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template)

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    def supported_language_ids(self) -> set:
        """Return the set of supported ISO 639-3 language ids."""
        return LANG_IDS

    def supported_language_names(self) -> set:
        """Return the set of supported language display names."""
        return LANG_NAMES

    def resolve_language(self, language: Optional[str]) -> Optional[str]:
        """See [`resolve_language`]."""
        return resolve_language(language)

    def resolve_instruct(self, instruct: Optional[str], use_zh: bool = False) -> Optional[str]:
        """See [`resolve_instruct`]."""
        return resolve_instruct(instruct, use_zh=use_zh)

    def normalize_text(self, text: str, language: Optional[str] = None) -> str:
        """See [`normalize_text`]."""
        return normalize_text(text, language)

    def estimate_target_length(
        self,
        text: str,
        ref_text: Optional[str],
        num_ref_audio_tokens: Optional[int],
        speed: float = 1.0,
    ) -> int:
        """Estimate how many audio-token frames `text` should decode to.

        Args:
            text: Target text.
            ref_text: Transcript of the reference audio, or `None`.
            num_ref_audio_tokens: Number of audio-token frames the reference audio encoded to, or `None`.
            speed: Speaking-speed factor; `> 1.0` shortens the estimate, `< 1.0` lengthens it.

        Returns:
            The estimated number of audio-token frames, at least `1`.
        """
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25
        est = self.duration_estimator.estimate_duration(text, ref_text, num_ref_audio_tokens)
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def encode_reference(
        self,
        ref_audio: Union[str, tuple],
        ref_text: str,
        preprocess: bool = True,
    ) -> dict:
        """Prepare a reference (voice-clone) prompt from reference audio and its transcript.

        Args:
            ref_audio: File path, or a `(waveform, sample_rate)` tuple. `waveform` is a 1-D or 2-D
                array/tensor (channels x samples).
            ref_text: Transcript of the reference audio.
            preprocess: If `True`, remove silence and, for long audio, trim it to the largest silence gap
                before encoding, and append end punctuation to `ref_text` if missing.

        Returns:
            A dict with `"ref_audio_tokens"` (`torch.LongTensor` of shape `(num_audio_codebook, T)`),
            `"ref_text"` (`str`), and `"ref_rms"` (`float`, the reference waveform's RMS amplitude before
            normalization).

        Raises:
            `ValueError`: If the reference audio is empty after silence removal.
        """
        if isinstance(ref_audio, str):
            ref_wav = load_audio(ref_audio, self.sampling_rate)
        else:
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0, keepdims=True)
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform), orig_freq=sr, new_freq=self.sampling_rate
                ).numpy()
            ref_wav = waveform

        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        if preprocess:
            ref_wav = trim_long_audio(ref_wav, self.sampling_rate, trim_threshold=20.0)
            ref_wav = remove_silence(ref_wav, self.sampling_rate, mid_sil=200, lead_sil=100, trail_sil=200)
            if ref_wav.shape[-1] == 0:
                raise ValueError(
                    "Reference audio is empty after silence removal. Try calling encode_reference with "
                    "preprocess=False."
                )
            ref_text = add_punctuation(ref_text)

        if ref_wav.shape[-1] / self.sampling_rate > 20.0:
            logger.warning(
                "Reference audio is longer than 20s. This may cause slower generation, higher memory usage, "
                "and degraded voice cloning quality. Trimming it to 3-10s is recommended."
            )

        chunk_size = self.audio_tokenizer.config.hop_length
        clip_size = int(ref_wav.shape[-1] % chunk_size)
        ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        ref_audio_tokens = self.audio_tokenizer.encode(ref_wav_tensor.unsqueeze(0)).audio_codes.squeeze(0)

        return {"ref_audio_tokens": ref_audio_tokens, "ref_text": ref_text, "ref_rms": ref_rms}

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput],
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """Build the `input_ids`/`audio_mask` conditioning for one generation request.

        Args:
            text: Target text to synthesize.
            num_target_tokens: Number of masked audio-token frames to append as the generation target
                (see [`~OmniVoiceProcessor.estimate_target_length`]).
            ref_text: Transcript of the reference audio, for voice cloning. `None` for voice design/auto mode.
            ref_audio_tokens: Reference audio tokens of shape `(num_audio_codebook, T)`, from
                [`~OmniVoiceProcessor.encode_reference`]. `None` for voice design/auto mode.
            language: Resolved language id, or `None` for language-agnostic mode.
            instruct: Resolved voice-design instruct string, or `None`.
            denoise: Whether to prepend the `<|denoise|>` style token (used when conditioning on reference
                audio).
            return_tensors: Only `"pt"` is supported.

        Returns:
            [`BatchFeature`] with `"input_ids"` (`torch.LongTensor` of shape
            `(1, num_audio_codebook, cond_len)`) and `"audio_mask"` (`torch.BoolTensor` of shape
            `(1, cond_len)`).
        """
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports return_tensors='pt'.")

        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = language if language else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_tokens = (
            self.tokenizer(style_text, return_tensors="pt").input_ids.repeat(self.num_audio_codebook, 1).unsqueeze(0)
        )

        full_text = _combine_text(text, ref_text=ref_text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
            .repeat(self.num_audio_codebook, 1)
            .unsqueeze(0)
        )

        target_audio_tokens = torch.full(
            (1, self.num_audio_codebook, num_target_tokens), self.audio_mask_id, dtype=torch.long
        )

        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(1, cond_total_length, dtype=torch.bool)
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return BatchFeature(data={"input_ids": cond_input_ids, "audio_mask": cond_audio_mask})

    def decode(
        self,
        audio_tokens: Union[torch.Tensor, list[torch.Tensor]],
        ref_rms: Optional[float] = None,
        postprocess: bool = True,
        pad_duration: float = 0.1,
        fade_duration: float = 0.1,
    ) -> np.ndarray:
        """Decode generated audio tokens into a waveform.

        Args:
            audio_tokens: Audio tokens of shape `(num_audio_codebook, T)`, or a list of such tensors (one
                per chunk, cross-faded together).
            ref_rms: RMS amplitude of the reference audio, used to restore the original volume. `None` skips
                RMS-based normalization and instead peak-normalizes to `0.5`.
            postprocess: If `True`, remove long silences from the decoded waveform.
            pad_duration: Silence padding duration per side, in seconds (`0` to disable).
            fade_duration: Fade-in/out curve duration, in seconds (`0` to disable).

        Returns:
            A numpy float32 array of shape `(T,)`.
        """
        device = self.audio_tokenizer.device
        if isinstance(audio_tokens, list):
            chunk_audios = [
                self.audio_tokenizer.decode(t.to(device).unsqueeze(0)).audio_values[0].cpu().numpy()
                for t in audio_tokens
            ]
            waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            waveform = self.audio_tokenizer.decode(audio_tokens.to(device).unsqueeze(0)).audio_values[0].cpu().numpy()

        if postprocess:
            waveform = remove_silence(waveform, self.sampling_rate, mid_sil=500, lead_sil=100, trail_sil=100)

        if ref_rms is not None and ref_rms < 0.1:
            waveform = waveform * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(waveform).max()
            if peak > 1e-6:
                waveform = waveform / peak * 0.5

        waveform = fade_and_pad_audio(
            waveform, pad_duration=pad_duration, fade_duration=fade_duration, sample_rate=self.sampling_rate
        )
        return waveform.squeeze(0)

    @property
    def model_input_names(self):
        return ["input_ids", "audio_mask"]


__all__ = [
    "OmniVoiceProcessor",
    "add_punctuation",
    "chunk_text_punctuation",
    "lang_display_name",
    "normalize_text",
    "resolve_instruct",
    "resolve_language",
]
