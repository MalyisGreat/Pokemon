"""
Pokemon Vocabulary Tokenizer

Handles tokenization of Pokemon-specific vocabulary:
- Pokemon names
- Move names
- Item names
- Ability names
- Type names
- Status conditions
- Field conditions
- Generation/format tags
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

import torch


class PokemonTokenizer:
    """
    Tokenizer for Pokemon battle observations.

    The vocabulary includes:
    - Special tokens: <pad>, <unk>, <blank>, <player>, <opponent>, etc.
    - Pokemon species names
    - Move names
    - Item names
    - Ability names
    - Type names
    - Status/condition tokens
    - Generation format tokens
    """

    SPECIAL_TOKENS = [
        "<pad>",
        "<unk>",
        "<blank>",
        "<player>",
        "<opponent>",
        "<move>",
        "<switch>",
        "<moveset>",
        "<conditions>",
        "<player_prev>",
        "<opp_prev>",
        "<anychoice>",
        "<forcemove>",
        "<forceswitch>",
        # Format tokens
        "<gen1ou>",
        "<gen1uu>",
        "<gen1nu>",
        "<gen1ubers>",
        "<gen2ou>",
        "<gen2uu>",
        "<gen2nu>",
        "<gen2ubers>",
        "<gen3ou>",
        "<gen3uu>",
        "<gen3nu>",
        "<gen3ubers>",
        "<gen4ou>",
        "<gen4uu>",
        "<gen4nu>",
        "<gen4ubers>",
        "<gen5ou>",
        "<gen6ou>",
        "<gen7ou>",
        "<gen8ou>",
        "<gen9ou>",
        # Types
        "normal",
        "fire",
        "water",
        "electric",
        "grass",
        "ice",
        "fighting",
        "poison",
        "ground",
        "flying",
        "psychic",
        "bug",
        "rock",
        "ghost",
        "dragon",
        "dark",
        "steel",
        "fairy",
        "notype",
        # Categories
        "physical",
        "special",
        "status",
        # Status conditions
        "nostatus",
        "brn",
        "par",
        "slp",
        "frz",
        "psn",
        "tox",
        "fnt",
        # Field conditions
        "noweather",
        "noconditions",
        "raindance",
        "sunnyday",
        "sandstorm",
        "hail",
        "snow",
        "reflect",
        "lightscreen",
        "safeguard",
        "mist",
        "spikes",
        "toxicspikes",
        "stealthrock",
        "stickyweb",
        # Effects
        "noeffect",
        "supereffective",
        "noteffective",
        "immune",
        # Actions
        "nomove",
        # Items (common)
        "leftovers",
        "choiceband",
        "choicescarf",
        "choicespecs",
        "lifeorb",
        "focussash",
        "lumberry",
        "sitrusberry",
        "unknownitem",
        # Abilities (common)
        "unknownability",
        "levitate",
        "intimidate",
        "sandstream",
        "drought",
        "drizzle",
        "snowwarning",
        # HP buckets
        "hp0",
        "hp25",
        "hp50",
        "hp75",
        "hp100",
        # Opponent hazard markers
        "oppspikes",
        "oppstealthrock",
        "opptoxicspikes",
        "oppstickyweb",
        # Common Pokemon (Gen 1-9 OU staples)
        "pikachu", "charizard", "blastoise", "venusaur", "alakazam", "gengar",
        "starmie", "snorlax", "tauros", "chansey", "exeggutor", "jolteon",
        "dragonite", "zapdos", "articuno", "moltres", "mewtwo", "mew",
        "tyranitar", "skarmory", "blissey", "celebi", "suicune", "raikou",
        "entei", "lugia", "hooh", "forretress", "cloyster", "machamp",
        "salamence", "metagross", "garchomp", "lucario", "togekiss", "rotom",
        "rotomwash", "rotomheat", "heatran", "gliscor", "scizor", "ferrothorn",
        "landorus", "landorustherian", "thundurus", "tornadus", "keldeo",
        "volcarona", "excadrill", "conkeldurr", "reuniclus", "hydreigon",
        "greninja", "aegislash", "talonflame", "azumarill", "mawile",
        "dragapult", "toxapex", "corviknight", "clefable", "hippowdon",
        "weavile", "magnezone", "slowbro", "slowking", "pelipper", "kingdra",
        "swampert", "blaziken", "breloom", "infernape", "staraptor", "roserade",
        "gyarados", "milotic", "tentacruel", "aerodactyl", "jirachi", "latios",
        "latias", "deoxys", "darkrai", "arceus", "genesect", "bisharp",
        "mandibuzz", "krookodile", "mienshao", "terrakion", "cobalion",
        "virizion", "kyurem", "zekrom", "reshiram", "diancie", "hoopa",
        "tapu", "tapukoko", "tapulele", "tapubulu", "tapufini",
        "celesteela", "kartana", "pheromosa", "buzzwole", "xurkitree",
        "nihilego", "guzzlord", "necrozma", "solgaleo", "lunala", "marshadow",
        "zeraora", "melmetal", "zacian", "zamazenta", "eternatus", "calyrex",
        "urshifu", "regieleki", "regidrago", "spectrier", "glastrier",
        "miraidon", "koraidon", "chienyu", "tinglu", "chienpao", "wochien",
        "flutter", "ironvaliant", "ironhands", "irontreads", "ironjugulis",
        "roaringmoon", "ironbundle", "greattusk", "sandyshocks", "brutebonnet",
        "slitherwing", "screamtail", "walkingwake", "ironleaves", "gougingfire",
        "ragingbolt", "ironboulder", "ironcrown", "terapagos", "pecharunt",
        # Common moves
        "thunderbolt", "flamethrower", "icebeam", "surf", "earthquake",
        "psychic", "shadowball", "sludgebomb", "focusblast", "energyball",
        "flashcannon", "darkpulse", "dragonpulse", "aurasphere", "moonblast",
        "hydropump", "fireblast", "thunder", "blizzard", "hurricane",
        "dracometeor", "overheat", "leafstorm", "closecombat", "superpower",
        "knockoff", "uturn", "voltswitch", "scald", "willowisp", "toxic",
        "stealthrock", "spikes", "toxicspikes", "defog", "rapidspin",
        "roost", "recover", "softboiled", "synthesis", "wish", "protect",
        "substitute", "swordsdance", "nastyplot", "calmmind", "dragondance",
        "shellsmash", "quiverdance", "agility", "rockpolish", "bulkup",
        "irondefense", "amnesia", "barrier", "cosmicpower", "cottonguard",
        "bodyslam", "return", "facade", "extremespeed", "quickattack",
        "bulletpunch", "machpunch", "aquajet", "iceshard", "shadowsneak",
        "suckerpunch", "bravebird", "flareblitz", "wildcharge", "headsmash",
        "doubleedge", "woodhammer", "headcharge", "outrage", "thrash",
        "petalblizzard", "stoneEdge", "rockslide", "ironhead", "zenheadbutt",
        "playrough", "drainpunch", "hammerarm", "crunch", "bite",
        "thunder", "thunderwave", "glare", "stunspore", "sleeppowder",
        "spore", "yawn", "hypnosis", "sing", "lovelykiss", "grasswhistle",
        "encore", "taunt", "torment", "disable", "healbell", "aromatherapy",
        "trick", "switcheroo", "memento", "explosion", "selfdestruct",
        "destinybond", "perishsong", "haze", "whirlwind", "roar", "dragontail",
        "circlethrow", "batonpass", "teleport", "partingshot", "flipturn",
        "gigadrain", "leechlife", "drainpunch", "hornleech", "paraboliccharge",
        "oblivionwing", "leechseed", "ingrain", "aquaring", "rest",
        "sleeptalk", "snore", "naturepower", "secretpower", "hiddenpower",
        "judgment", "multiattack", "technoblast", "weatherball", "terrainpulse",
        "boomburst", "hypervoice", "uproar", "echoedvoice", "round",
        "gigaimpact", "hyperbeam", "frenzyplant", "blastburn", "hydrocannon",
        "earthquake", "bulldoze", "earthpower", "precipiceblades", "thousandarrows",
        "fly", "bounce", "skydrop", "acrobatics", "aerialace", "airslash",
        "bravebird", "drillpeck", "hurricane", "oblivionwing", "skyattack",
        "lavaplume", "heatwave", "eruption", "blueflare", "vcreate",
        "sacredfire", "firepunch", "firefang", "fierydance", "inferno",
        "surf", "waterfall", "aquatail", "liquidation", "originpulse",
        "waterspout", "waterpledge", "sparklingaria", "steameruption",
        "thunderpunch", "thunderfang", "discharge", "zapcannon", "boltbeak",
        "risingvoltage", "electroball", "volttackle", "fusionbolt",
        "glaciate", "freezedry", "auroraveil", "icepunch", "icefang",
        "iciclecrash", "iciclespear", "tripleaxel", "freezeshock",
        "grassknot", "powerwhip", "seedbomb", "leafblade", "solarbeam",
        "petalblizzard", "grasspledge", "frenzyplant", "floralhealing",
        "psychocut", "psyshock", "storedpower", "expandingforce", "futuresight",
        "zenheadbutt", "psychicfangs", "lusterpurge", "mistball", "photongeyser",
        "poisonjab", "gunkshot", "sludgewave", "crosspoison", "venoshock",
        "acidspray", "belch", "shellsidearm", "malignantchain",
        "rockblast", "accelerock", "diamondstorm", "powergem", "ancientpower",
        "meteormash", "ironhead", "gyroball", "heavyslam", "steelbeam",
        "bulletpunch", "smartstrike", "kingshield", "behemothbash",
        "shadowclaw", "phantomforce", "spectralthief", "moongeistbeam",
        "poltergeist", "astralbarrage", "bittermalice", "infernalparade",
        "dragonclaw", "outrage", "dragonrush", "dualchop", "dragondarts",
        "clangoroussoul", "eternabeam", "coreenforceer", "dynamaxcannon",
        "nightslash", "foulplay", "lashout", "fieryWrath", "wicked blow",
        "falsesurrender", "jawlock", "obstagoon",
        "dazzlinggleam", "drainingkiss", "spiritbreak", "strangeSteam",
        "mistyexplosion", "fleurcannon", "moonlight", "geomancy",
        # Items
        "choiceband", "choicescarf", "choicespecs", "lifeorb", "leftovers",
        "focussash", "assaultvest", "rockyhelmet", "heavydutyboots",
        "eviolite", "blacksludge", "toxicorb", "flameorb", "laggingTail",
        "airballoon", "shedshell", "redcard", "ejectbutton", "ejectpack",
        "weaknesspolicy", "throatspray", "roomservice", "blunderpolicy",
        "lumberry", "sitrusberry", "aguavberry", "figyberry", "wikiberry",
        "magoberry", "iapapaberry", "chestoberry", "rawstberry", "pechaberry",
        "aspearberry", "persimberry", "leppaberry", "oranberry", "liechiberry",
        "ganlonberry", "salacberry", "petayaberry", "apicotberry", "lansatberry",
        "starfberry", "enigmaberry", "micleberry", "custapberry", "jabocaberry",
        "rowapberry", "keeberry", "marangaberry", "occaberry", "passhoberry",
        "wacanberry", "rindoberry", "yacheberry", "chopleberry", "kebiaberry",
        "shucaberry", "cobaberry", "payapaberry", "tangaberry", "chartiberry",
        "kasibberry", "habanberry", "colburberry", "babiriberry", "chilanberry",
        "roseliberry", "expertbelt", "metronome", "muscleband", "wiseglasses",
        "scopelens", "widelens", "zoomlens", "razorclaw", "razorfang",
        "kingsrock", "protectivepads", "safetygoggles", "utilityumbrella",
        "terrainextender", "electricseed", "grassyseed", "mistyseed", "psychicseed",
        "throatspray", "adrenalineorb", "mentalherb", "powerherb", "whiteherb",
        # Abilities
        "overgrow", "blaze", "torrent", "swarm", "guts", "sheerforce",
        "hustle", "technician", "adaptability", "download", "moody",
        "speedboost", "chlorophyll", "swiftswim", "sandrush", "slushrush",
        "surgesurfer", "unburden", "quickfeet", "defiant", "competitive",
        "moxie", "beastboost", "soulheart", "battlearmor", "shellarmor",
        "sturdy", "multiscale", "shadowshield", "filter", "solidrock",
        "furcoat", "marvelscale", "fluffy", "iceface", "disguise",
        "wonderguard", "magicguard", "regenerator", "naturalcure", "immunity",
        "waterveil", "limber", "owntempo", "innerfocus", "oblivious",
        "synchronize", "trace", "imposter", "protean", "libero",
        "colorchange", "forecast", "flowergift", "zenmode", "stancechange",
        "schooling", "battlebond", "powerconstruct", "shieldsdown", "comatose",
        "rkssystem", "disguise", "gulpmissile", "iceface", "hungerswitch",
        "asone", "curiousmedicine", "transistor", "dragonsmaw", "unseenfist",
        "flashfire", "heatproof", "thickfat", "waterabsorb", "stormdrain",
        "voltabsorb", "lightningrod", "motordrive", "sapsipper", "dryskin",
        "justified", "rattled", "telepathy", "anticipation", "forewarn",
        "frisk", "infiltrator", "pressure", "unnerve", "moldbreaker",
        "teravolt", "turboblaze", "darkaura", "fairyaura", "aurabreak",
        "prankster", "triage", "galewings", "stakeout", "stalwart",
        "propellertail", "queenlymajesty", "dazzling", "psychicsurge",
        "electricsurge", "grassysurge", "mistysurge", "screencleaner",
        "neutralizinggas", "pastelveil", "punkrock", "icescales", "ripen",
        "steelworker", "steelyspirit", "perishbody", "wanderingspirit",
        "gorillatactics", "mirrorarmor", "intrepidsword", "dauntlessshield",
        "libero", "ballfetch", "cottondown", "steamengine", "sandspit",
        "iceface", "powerspot", "mimicry", "gulpmissile", "stalwart",
        "unknown",
    ]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab_size: int = 8192,
    ):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Build vocabulary from special tokens and common Pokemon vocabulary"""
        idx = 0

        # Add special tokens first
        for token in self.SPECIAL_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Add placeholder tokens for the rest of vocabulary
        # In practice, this would be populated from the actual dataset
        while idx < self.vocab_size:
            token = f"<token_{idx}>"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary"""
        token = token.lower().replace(" ", "").replace("-", "")

        if token in self.token_to_id:
            return self.token_to_id[token]

        if len(self.token_to_id) >= self.vocab_size:
            return self.token_to_id["<unk>"]

        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs"""
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = [t.lower() for t in text]

        ids = []
        for token in tokens:
            token = token.replace(" ", "").replace("-", "")
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id.get("<unk>", 1))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append("<unk>")
        return " ".join(tokens)

    def encode_observation(self, obs_text: str, max_length: int = 87) -> torch.Tensor:
        """
        Encode a full observation text to fixed-length tensor.

        The observation format from the paper:
        <format> <choice_type> <player> ... <opponent> ... <conditions> ... <player_prev> ... <opp_prev> ...
        """
        # Handle dict/complex objects being passed as strings
        if not isinstance(obs_text, str):
            obs_text = str(obs_text)

        # If it looks like a Python dict repr, just use padding (can't parse it)
        if obs_text.startswith("{") or obs_text.startswith("{'"):
            ids = [self.pad_token_id] * max_length
            return torch.tensor(ids, dtype=torch.long)

        ids = self.encode(obs_text)

        # Clamp all IDs to valid range
        ids = [min(max(0, i), self.vocab_size - 1) for i in ids]

        # Pad or truncate to max_length
        if len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 87,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of texts"""
        encoded = [self.encode_observation(text, max_length) for text in texts]

        if return_tensors == "pt":
            return {"input_ids": torch.stack(encoded)}
        else:
            return {"input_ids": encoded}

    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, "r") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.token_to_id)


def build_vocab_from_replays(
    replay_dir: str,
    output_path: str,
    max_vocab_size: int = 8192,
) -> PokemonTokenizer:
    """
    Build vocabulary from a directory of replay files.

    This scans all replays and extracts unique tokens for:
    - Pokemon names
    - Move names
    - Item names
    - Ability names
    """
    from collections import Counter
    import glob

    tokenizer = PokemonTokenizer(vocab_size=max_vocab_size)
    token_counts = Counter()

    replay_files = glob.glob(os.path.join(replay_dir, "**/*.json"), recursive=True)

    for replay_file in replay_files:
        try:
            with open(replay_file, "r") as f:
                data = json.load(f)

            # Extract tokens from trajectory observations
            for trajectory in data.get("trajectories", []):
                for step in trajectory.get("steps", []):
                    if "text_obs" in step:
                        tokens = step["text_obs"].lower().split()
                        token_counts.update(tokens)
        except Exception as e:
            continue

    # Add most common tokens to vocabulary
    for token, _ in token_counts.most_common(max_vocab_size - len(PokemonTokenizer.SPECIAL_TOKENS)):
        tokenizer.add_token(token)

    tokenizer.save_vocab(output_path)
    return tokenizer
