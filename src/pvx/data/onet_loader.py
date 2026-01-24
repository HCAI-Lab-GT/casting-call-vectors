"""O*NET Database Loader for vocational persona generation.

Parses O*NET database text files to extract occupation data, RIASEC scores,
tasks, and other descriptors needed for generating vocational personas.

Example usage:
    loader = ONETLoader("data/onet_raw")
    occupations = loader.load_occupations()
    profile = loader.get_occupation_profile("11-1011.00")
    print(profile["riasec"])  # {'R': 1.30, 'I': 3.24, 'A': 2.08, ...}
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# RIASEC Element ID to dimension mapping
RIASEC_ELEMENTS = {
    "1.B.1.a": "R",  # Realistic
    "1.B.1.b": "I",  # Investigative
    "1.B.1.c": "A",  # Artistic
    "1.B.1.d": "S",  # Social
    "1.B.1.e": "E",  # Enterprising
    "1.B.1.f": "C",  # Conventional
}

RIASEC_FULL_NAMES = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}

# High-point code mapping (IH scale value -> RIASEC letter)
HIGHPOINT_TO_RIASEC = {
    1.0: "R",
    2.0: "I",
    3.0: "A",
    4.0: "S",
    5.0: "E",
    6.0: "C",
    0.0: None,  # No third high-point
}


class ONETLoader:
    """Load and process O*NET database files.

    Provides access to occupation data, RIASEC interest profiles,
    tasks, skills, and other descriptors from the O*NET database.

    Args:
        data_dir: Path to directory containing O*NET text files.
    """

    def __init__(self, data_dir: str | Path = "data/onet_raw"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"O*NET data directory not found: {self.data_dir}\nRun: ./scripts/download_onet.sh"
            )

        # Cached dataframes
        self._occupations: Optional[pd.DataFrame] = None
        self._interests: Optional[pd.DataFrame] = None
        self._tasks: Optional[pd.DataFrame] = None
        self._riasec_by_occupation: Optional[dict] = None

    def load_occupations(self) -> pd.DataFrame:
        """Load occupation titles and descriptions.

        Returns:
            DataFrame with columns: soc_code, title, description
        """
        if self._occupations is not None:
            return self._occupations

        filepath = self.data_dir / "Occupation Data.txt"
        df = pd.read_csv(filepath, sep="\t")
        df.columns = ["soc_code", "title", "description"]
        self._occupations = df
        logger.info(f"Loaded {len(df)} occupations from O*NET")
        return df

    def load_interests(self) -> pd.DataFrame:
        """Load raw interests data (RIASEC scores).

        Returns:
            DataFrame with all interest data including OI and IH scales.
        """
        if self._interests is not None:
            return self._interests

        filepath = self.data_dir / "Interests.txt"
        df = pd.read_csv(filepath, sep="\t")
        df.columns = [
            "soc_code",
            "element_id",
            "element_name",
            "scale_id",
            "data_value",
            "date",
            "domain_source",
        ]
        self._interests = df
        logger.info(f"Loaded {len(df)} interest records from O*NET")
        return df

    def load_tasks(self) -> pd.DataFrame:
        """Load task statements for occupations.

        Returns:
            DataFrame with columns: soc_code, task_id, task, task_type, ...
        """
        if self._tasks is not None:
            return self._tasks

        filepath = self.data_dir / "Task Statements.txt"
        df = pd.read_csv(filepath, sep="\t")
        df.columns = [
            "soc_code",
            "task_id",
            "task",
            "task_type",
            "incumbents_responding",
            "date",
            "domain_source",
        ]
        self._tasks = df
        logger.info(f"Loaded {len(df)} task statements from O*NET")
        return df

    def get_riasec_scores(self) -> dict[str, dict[str, float]]:
        """Get RIASEC scores for all occupations.

        Returns:
            Dict mapping SOC code -> {R, I, A, S, E, C scores}
        """
        if self._riasec_by_occupation is not None:
            return self._riasec_by_occupation

        interests = self.load_interests()

        # Filter to OI scale (actual scores) and RIASEC elements
        oi_data = interests[
            (interests["scale_id"] == "OI") & (interests["element_id"].isin(RIASEC_ELEMENTS.keys()))
        ]

        # Build dict of SOC -> {R: score, I: score, ...}
        result = {}
        for soc_code, group in oi_data.groupby("soc_code"):
            scores = {}
            for _, row in group.iterrows():
                dim = RIASEC_ELEMENTS[row["element_id"]]
                scores[dim] = row["data_value"]
            result[soc_code] = scores

        self._riasec_by_occupation = result
        return result

    def get_highpoint_codes(self) -> dict[str, list[str]]:
        """Get RIASEC high-point codes for all occupations.

        Returns:
            Dict mapping SOC code -> [primary, secondary, tertiary] RIASEC letters
        """
        interests = self.load_interests()

        # Filter to IH scale (high-point codes)
        ih_data = interests[
            (interests["scale_id"] == "IH")
            & (interests["element_id"].isin(["1.B.1.g", "1.B.1.h", "1.B.1.i"]))
        ]

        result = {}
        for soc_code, group in ih_data.groupby("soc_code"):
            codes = []
            for element_id in ["1.B.1.g", "1.B.1.h", "1.B.1.i"]:
                row = group[group["element_id"] == element_id]
                if not row.empty:
                    val = row.iloc[0]["data_value"]
                    letter = HIGHPOINT_TO_RIASEC.get(val)
                    if letter:
                        codes.append(letter)
            result[soc_code] = codes

        return result

    def get_occupation_profile(self, soc_code: str) -> dict:
        """Get complete profile for one occupation.

        Args:
            soc_code: O*NET-SOC occupation code (e.g., "11-1011.00")

        Returns:
            Dict with keys: soc_code, title, description, riasec, highpoint_codes, tasks
        """
        occupations = self.load_occupations()
        occ_row = occupations[occupations["soc_code"] == soc_code]

        if occ_row.empty:
            raise ValueError(f"Occupation not found: {soc_code}")

        occ = occ_row.iloc[0]

        # Get RIASEC scores
        riasec_scores = self.get_riasec_scores()
        riasec = riasec_scores.get(soc_code, {})

        # Get high-point codes
        highpoint_codes = self.get_highpoint_codes()
        highpoints = highpoint_codes.get(soc_code, [])

        # Get tasks
        tasks_df = self.load_tasks()
        tasks = tasks_df[tasks_df["soc_code"] == soc_code]["task"].tolist()

        return {
            "soc_code": soc_code,
            "title": occ["title"],
            "description": occ["description"],
            "riasec": riasec,
            "riasec_primary": highpoints[0] if highpoints else None,
            "highpoint_codes": highpoints,
            "tasks": tasks,
        }

    def filter_by_riasec(self, primary: str, min_score: float = 5.0) -> list[str]:
        """Get occupations with high scores on a specific RIASEC dimension.

        Args:
            primary: RIASEC dimension letter (R, I, A, S, E, or C)
            min_score: Minimum score threshold (scale 1-7)

        Returns:
            List of SOC codes meeting the criteria
        """
        if primary not in RIASEC_FULL_NAMES:
            raise ValueError(
                f"Invalid RIASEC dimension: {primary}. "
                f"Must be one of: {list(RIASEC_FULL_NAMES.keys())}"
            )

        riasec_scores = self.get_riasec_scores()
        matching = []
        for soc_code, scores in riasec_scores.items():
            if scores.get(primary, 0) >= min_score:
                matching.append(soc_code)

        return matching

    def get_all_occupation_profiles(self) -> list[dict]:
        """Get profiles for all occupations.

        Returns:
            List of occupation profile dicts
        """
        occupations = self.load_occupations()
        profiles = []
        for soc_code in occupations["soc_code"]:
            try:
                profile = self.get_occupation_profile(soc_code)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load profile for {soc_code}: {e}")
        return profiles

    def get_riasec_distribution(self) -> dict[str, int]:
        """Get count of occupations by primary RIASEC type.

        Returns:
            Dict mapping RIASEC letter -> count of occupations
        """
        highpoints = self.get_highpoint_codes()
        counts = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
        for codes in highpoints.values():
            if codes:
                counts[codes[0]] += 1
        return counts

    def to_slug(self, title: str) -> str:
        """Convert occupation title to filesystem-safe slug.

        Args:
            title: Occupation title (e.g., "Chief Executives")

        Returns:
            Lowercase slug (e.g., "chief_executives")
        """
        import re

        slug = title.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        return slug.strip("_")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    loader = ONETLoader()

    print("\n=== O*NET Database Summary ===")
    occupations = loader.load_occupations()
    print(f"Total occupations: {len(occupations)}")

    print("\n=== RIASEC Distribution ===")
    dist = loader.get_riasec_distribution()
    for letter, count in sorted(dist.items()):
        print(f"  {RIASEC_FULL_NAMES[letter]}: {count}")

    print("\n=== Sample Occupation Profile ===")
    profile = loader.get_occupation_profile("29-1141.00")  # Registered Nurses
    print(f"Title: {profile['title']}")
    print(f"RIASEC: {profile['riasec']}")
    print(f"Primary: {profile['riasec_primary']}")
    print(f"High-points: {profile['highpoint_codes']}")
    print(f"Tasks: {len(profile['tasks'])} tasks")
