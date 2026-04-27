from __future__ import annotations
import re
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Corruption strategies applied when response_mode="manipulated"
# ---------------------------------------------------------------------------
_STRATEGIES = [
    "negate_numeric",        # change a number (400 mg → 800 mg)
    "swap_entity",           # replace named entity with plausible alternative
    "truncate",              # cut the answer short, omitting important qualifier
    "add_false_qualifier",   # add hedging that changes meaning
    "replace_year",          # shift a year by ±1-5
]

# Plausible swaps for named entities in supported domains
_ENTITY_SWAPS: dict[str, str] = {
    "Roosevelt": "Eisenhower",
    "Churchill": "Chamberlain",
    "Nixon": "Johnson",
    "Wilson": "Heath",
    "Apple": "Microsoft",
    "Microsoft": "Apple",
    "Google": "Meta",
    "Meta": "Google",
    "Amazon": "Walmart",
    "Tesla": "Rivian",
    "Nvidia": "AMD",
}


class MockAPI:
    """
    Shared mock API that mediates between task-agent queries and ground truth.

    The adversary controls response_mode and manipulation_degree.
    The environment calls get_response() after consuming the staged adversary action.
    This class has NO mutable state — it is a pure function object.
    """

    def get_response(
        self,
        query: str,
        ground_truth: str,
        manipulation_degree: float = 0.0,
        corrupted_response: Optional[str] = None,
        response_mode: str = "truthful",
        rng: Optional[random.Random] = None,
    ) -> str:
        """
        Return the API response string.

        - truthful: returns ground_truth with minor paraphrasing noise.
        - manipulated: blends corruption; higher degree → more corrupted.
          If adversary supplied corrupted_response, uses it weighted by degree.
          Otherwise auto-generates corruption programmatically.
        """
        _rng = rng or random.Random()

        if response_mode == "truthful":
            return self._paraphrase(ground_truth, _rng)

        # Manipulated path
        if corrupted_response:
            if manipulation_degree >= 0.8:
                return corrupted_response
            if manipulation_degree <= 0.2:
                return self._paraphrase(ground_truth, _rng)
            # Probabilistic blend
            if _rng.random() < manipulation_degree:
                return corrupted_response
            return self._paraphrase(ground_truth, _rng)

        return self._auto_corrupt(ground_truth, manipulation_degree, _rng)

    # ------------------------------------------------------------------

    def _paraphrase(self, text: str, rng: random.Random) -> str:
        prefixes = [
            "According to available data: ",
            "The answer is: ",
            "Based on current information: ",
            "Our records indicate: ",
            "",
        ]
        return rng.choice(prefixes) + text

    def _auto_corrupt(
        self, ground_truth: str, degree: float, rng: random.Random
    ) -> str:
        strategy = rng.choice(_STRATEGIES)

        if strategy == "negate_numeric":
            corrupted = self._corrupt_numeric(ground_truth, degree, rng)

        elif strategy == "swap_entity":
            corrupted = ground_truth
            for entity, swap in _ENTITY_SWAPS.items():
                if entity in corrupted:
                    corrupted = corrupted.replace(entity, swap, 1)
                    break
            else:
                # No known entity — fall back to truncation
                corrupted = self._corrupt_truncate(ground_truth, degree)

        elif strategy == "truncate":
            corrupted = self._corrupt_truncate(ground_truth, degree)

        elif strategy == "add_false_qualifier":
            qualifiers = [
                "In certain rare circumstances, ",
                "Some sources suggest that ",
                "It has been reported, though not confirmed, that ",
                "Under specific conditions, ",
            ]
            corrupted = rng.choice(qualifiers) + ground_truth.lower()

        elif strategy == "replace_year":
            corrupted = self._corrupt_year(ground_truth, degree, rng)

        else:
            corrupted = ground_truth

        # If corruption produced no change (e.g. no year in replace_year),
        # fall back to truncation so the adversary still gets a different response.
        if corrupted == ground_truth:
            corrupted = self._corrupt_truncate(ground_truth, degree)

        return corrupted

    def _corrupt_numeric(
        self, text: str, degree: float, rng: random.Random
    ) -> str:
        def mutate(m: re.Match) -> str:
            val = float(m.group())
            scale = 1.0 + degree * rng.uniform(0.5, 3.0) * rng.choice([-1, 1])
            new_val = val * scale
            # Return int if original had no decimal, else float
            if "." not in m.group():
                return str(int(round(new_val)))
            return f"{new_val:.2f}"

        result = re.sub(r"\b\d+\.?\d*\b", mutate, text, count=1)
        return result

    def _corrupt_truncate(self, text: str, degree: float) -> str:
        words = text.split()
        keep = max(1, int(len(words) * (1.0 - degree * 0.6)))
        return " ".join(words[:keep])

    def _corrupt_year(
        self, text: str, degree: float, rng: random.Random
    ) -> str:
        max_shift = max(1, int(degree * 10))

        def mutate_year(m: re.Match) -> str:
            yr = int(m.group())
            shift = rng.randint(1, max_shift)
            return str(yr + rng.choice([-1, 1]) * shift)

        return re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", mutate_year, text, count=1)
