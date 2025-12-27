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
        ids = self.encode(obs_text)

        # Pad or truncate to max_length
        if len(ids) < max_length:
            ids = ids + [self.token_to_id["<pad>"]] * (max_length - len(ids))
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
