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
from tqdm import tqdm

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


# Work Styles Element ID mapping (21 personality-like traits)
WORK_STYLE_ELEMENTS = {
    "1.D.1.a": "Innovation",
    "1.D.1.b": "Achievement Orientation",
    "1.D.1.c": "Intellectual Curiosity",
    "1.D.1.d": "Tolerance for Ambiguity",
    "1.D.1.e": "Initiative",
    "1.D.1.f": "Adaptability",
    "1.D.1.g": "Self-Confidence",
    "1.D.1.h": "Perseverance",
    "1.D.1.i": "Leadership Orientation",
    "1.D.2.a": "Humility",
    "1.D.2.b": "Sincerity",
    "1.D.2.c": "Empathy",
    "1.D.2.d": "Cooperation",
    "1.D.2.e": "Optimism",
    "1.D.2.f": "Social Orientation",
    "1.D.3.a": "Cautiousness",
    "1.D.3.b": "Attention to Detail",
    "1.D.3.c": "Dependability",
    "1.D.3.d": "Integrity",
    "1.D.4.a": "Stress Tolerance",
    "1.D.4.b": "Self-Control",
}

# Big Five (OCEAN) derived from Work Styles
# Based on established Work Styles → FFM mappings
BIG_FIVE_MAPPING = {
    "O": [
        "1.D.1.f",
        "1.D.1.a",
        "1.D.1.c",
        "1.D.1.d",
    ],  # Openness: Adaptability, Innovation, Intellectual Curiosity, Tolerance for Ambiguity
    "C": [
        "1.D.1.b",
        "1.D.3.a",
        "1.D.3.b",
        "1.D.3.c",
        "1.D.1.g",
    ],  # Conscientiousness: Achievement Orientation, Cautiousness, Attention to Detail, Dependability, Self-Confidence
    "E": ["1.D.1.i", "1.D.2.f"],  # Extraversion: Leadership Orientation, Social Orientation
    "A": ["1.D.2.d", "1.D.2.c"],  # Agreeableness: Cooperation, Empathy
    "N_inv": [
        "1.D.4.a",
        "1.D.4.b",
    ],  # Emotional Stability (inverse N): Stress Tolerance, Self-Control
}

# HEXACO Mapping (Officially organized by O*Net)
# Based on Higher_Order_Styles relsease in O*Net 30
HEXACO_MAPPING = {
    "H": ["1.D.2.a", "1.D.3.d", "1.D.2.b"],  # Honesty-Humility: Humility, Integrity, Sincerity
    "E": ["1.D.4.a", "1.D.4.b"],  # Emotionality
    "X": ["1.D.1.i", "1.D.2.f"],  # Extraversion
    "A": ["1.D.2.d", "1.D.2.c"],  # Agreeableness
    "C": [
        "1.D.1.b",
        "1.D.3.a",
        "1.D.3.b",
        "1.D.3.c",
        "1.D.1.g",
    ],  # Conscientiousness
    "O": [
        "1.D.1.f",
        "1.D.1.a",
        "1.D.1.c",
        "1.D.1.d",
    ],  # Opennes to Experience
}

# Work Values Element ID mapping (6 work values)
WORK_VALUE_ELEMENTS = {
    "1.B.2.a": "Achievement",
    "1.B.2.b": "Working Conditions",
    "1.B.2.c": "Recognition",
    "1.B.2.d": "Relationships",
    "1.B.2.e": "Support",
    "1.B.2.f": "Independence",
}

# Work Value High-Point mapping (VH scale)
WORK_VALUE_HIGHPOINT = {
    1: "Achievement",
    2: "Working Conditions",
    3: "Recognition",
    4: "Relationships",
    5: "Support",
    6: "Independence",
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
        # Work Styles and Work Values caches
        self._work_styles: Optional[pd.DataFrame] = None
        self._work_values: Optional[pd.DataFrame] = None
        self._work_style_scores: Optional[dict] = None
        self._work_value_scores: Optional[dict] = None
        self._big_five_scores: Optional[dict] = None
        self._hexaco_scores: Optional[dict] = None
        self._work_contexts: Optional[pd.DataFrame] = None

    def load_occupations(self) -> pd.DataFrame:
        """Load occupation titles and descriptions.

        Returns:
            DataFrame with columns: soc_code, title, description
        """
        if self._occupations is not None:
            return self._occupations

        filepath = self.data_dir / "Occupation Data Processed.txt"
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

    def load_work_styles(self) -> pd.DataFrame:
        """Load Work Styles data (16 personality-like traits).

        Returns:
            DataFrame with columns: soc_code, element_id, element_name,
            scale_id, data_value, n, standard_error, lower_ci, upper_ci,
            recommend_suppress, date, domain_source
        """
        if self._work_styles is not None:
            return self._work_styles

        filepath = self.data_dir / "Work Styles.txt"
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
        self._work_styles = df
        logger.info(f"Loaded {len(df)} work style records from O*NET")
        return df

    def load_work_values(self) -> pd.DataFrame:
        """Load Work Values data (6 work values).

        Returns:
            DataFrame with columns: soc_code, element_id, element_name,
            scale_id, data_value, date, domain_source
        """
        if self._work_values is not None:
            return self._work_values

        filepath = self.data_dir / "Work Values.txt"
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
        self._work_values = df
        logger.info(f"Loaded {len(df)} work value records from O*NET")
        return df

    def load_work_contexts(self) -> pd.DataFrame:
        """Load Work Contexts data (environmental and social context).

        Returns:
            DataFrame with columns: soc_code, element_id, element_name,
            scale_id, data_value, date, domain_source
        """
        if self._work_contexts is not None:
            return self._work_contexts

        filepath = self.data_dir / "Work Context Processed.txt"
        df = pd.read_csv(filepath, sep="\t")
        df = df[df["Data Value"] >= 4].reset_index(drop=True)  # Filter to more relevant contexts
        df = df.loc[:, ["O*NET-SOC Code", "Element Name", "Category Description"]]
        df.columns = [
            "soc_code",
            "element_name",
            "data_value",
        ]
        logger.info(f"Loaded {len(df)} work context records from O*NET")
        self._work_contexts = df
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

    def get_work_style_scores(self) -> dict[str, dict[str, float]]:
        """Get Work Style scores (IM scale, 1-5) for all occupations.

        Returns:
            Dict mapping SOC code -> {trait_name: score, ...}
            Only includes the 16 standard Work Style traits.
        """
        if self._work_style_scores is not None:
            return self._work_style_scores

        work_styles = self.load_work_styles()

        # Filter to IM scale and known elements
        im_data = work_styles[
            (work_styles["scale_id"] == "DR")
            & (work_styles["element_id"].isin(WORK_STYLE_ELEMENTS.keys()))
        ]

        result = {}
        for soc_code, group in im_data.groupby("soc_code"):
            scores = {}
            for _, row in group.iterrows():
                element = row["element_id"]
                name = WORK_STYLE_ELEMENTS[element]
                scores[name] = row["data_value"]
            result[soc_code] = scores

        self._work_style_scores = result
        return result

    def get_big_five_scores(self) -> dict[str, dict[str, float]]:
        """Compute Big Five (OCEAN) scores derived from Work Styles.

        Returns:
            Dict mapping SOC code -> {"O": score, "C": score, "E": score,
            "A": score, "N_inv": score}

        Note:
            N_inv is Emotional Stability (inverse of Neuroticism).
            Higher values = more emotionally stable.
        """
        if self._big_five_scores is not None:
            return self._big_five_scores

        work_style_scores = self.get_work_style_scores()

        result = {}
        for soc_code, styles in work_style_scores.items():
            scores = {}
            for domain, element_ids in BIG_FIVE_MAPPING.items():
                domain_scores = []
                for elem_id in element_ids:
                    name = WORK_STYLE_ELEMENTS.get(elem_id)
                    if name and name in styles:
                        domain_scores.append(styles[name])
                if domain_scores:
                    scores[domain] = sum(domain_scores) / len(domain_scores)
            result[soc_code] = scores

        self._big_five_scores = result
        return result

    def get_hexaco_scores(self) -> dict[str, dict[str, float]]:
        """Compute HEXACO scores derived from Work Styles.

        Returns:
            Dict mapping SOC code -> {"H": score, "E": score, "X": score, "A": score, "C": score, "O": score}

        """
        if self._hexaco_scores is not None:
            return self._hexaco_scores

        work_style_scores = self.get_work_style_scores()

        result = {}
        for soc_code, styles in work_style_scores.items():
            scores = {}
            for domain, element_ids in HEXACO_MAPPING.items():
                domain_scores = []
                for elem_id in element_ids:
                    name = WORK_STYLE_ELEMENTS.get(elem_id)
                    if name and name in styles:
                        domain_scores.append(styles[name])
                if domain_scores:
                    scores[domain] = sum(domain_scores) / len(domain_scores)
            result[soc_code] = scores

        self._hexaco_scores = result
        return result

    def get_work_value_scores(self) -> dict[str, dict[str, float]]:
        """Get Work Value scores (EX scale, 1-7) for all occupations.

        Returns:
            Dict mapping SOC code -> {value_name: score, ...}
            Only includes the 6 standard Work Values.
        """
        if self._work_value_scores is not None:
            return self._work_value_scores

        work_values = self.load_work_values()

        # Filter to EX scale and known elements
        ex_data = work_values[
            (work_values["scale_id"] == "EX")
            & (work_values["element_id"].isin(WORK_VALUE_ELEMENTS.keys()))
        ]

        result = {}
        for soc_code, group in ex_data.groupby("soc_code"):
            scores = {}
            for _, row in group.iterrows():
                element = row["element_id"]
                name = WORK_VALUE_ELEMENTS[element]
                scores[name] = row["data_value"]
            result[soc_code] = scores

        self._work_value_scores = result
        return result

    def get_work_value_highpoints(self) -> dict[str, list[str]]:
        """Get Work Value high-point codes for all occupations.

        Returns:
            Dict mapping SOC code -> [primary, secondary, tertiary] value names
        """
        work_values = self.load_work_values()

        # Filter to VH scale (high-point codes)
        vh_data = work_values[
            (work_values["scale_id"] == "VH")
            & (work_values["element_id"].isin(["1.B.2.g", "1.B.2.h", "1.B.2.i"]))
        ]

        result = {}
        for soc_code, group in vh_data.groupby("soc_code"):
            codes = []
            for element_id in ["1.B.2.g", "1.B.2.h", "1.B.2.i"]:
                row = group[group["element_id"] == element_id]
                if not row.empty:
                    val = int(row.iloc[0]["data_value"])
                    name = WORK_VALUE_HIGHPOINT.get(val)
                    if name:
                        codes.append(name)
            result[soc_code] = codes

        return result

    def get_occupation_profile(self, soc_code: str) -> dict:
        """Get complete profile for one occupation.

        Args:
            soc_code: O*NET-SOC occupation code (e.g., "11-1011.00")

        Returns:
            Dict with keys: soc_code, title, description, riasec,
            riasec_primary, highpoint_codes, work_styles, big_five,
            work_values, work_value_highpoints, tasks
        """
        occupations = self.load_occupations()
        occ_row = occupations[occupations["soc_code"] == soc_code]

        if occ_row.empty:
            raise ValueError(f"Occupation not found: {soc_code}")

        occ = occ_row.iloc[0]

        # Get RIASEC scores
        riasec_scores = self.get_riasec_scores()
        riasec = riasec_scores.get(soc_code, {})

        # Get RIASEC high-point codes
        highpoint_codes = self.get_highpoint_codes()
        highpoints = highpoint_codes.get(soc_code, [])

        # Get Work Styles (16 traits, scale 1-5)
        work_style_scores = self.get_work_style_scores()
        work_styles = work_style_scores.get(soc_code, {})

        # Get Big Five (derived from Work Styles)
        big_five_scores = self.get_big_five_scores()
        big_five = big_five_scores.get(soc_code, {})

        hexaco_scores = self.get_hexaco_scores()
        hexaco = hexaco_scores.get(soc_code, {})

        # Get Work Values (6 values, scale 1-7)
        work_value_scores = self.get_work_value_scores()
        work_values = work_value_scores.get(soc_code, {})

        # Get Work Value high-points
        work_value_hp = self.get_work_value_highpoints()
        work_value_highpoints = work_value_hp.get(soc_code, [])

        # Get tasks
        tasks_df = self.load_tasks()
        tasks = tasks_df[tasks_df["soc_code"] == soc_code]["task"].tolist()

        work_contexts_df = self.load_work_contexts()
        work_contexts = (
            work_contexts_df[work_contexts_df["soc_code"] == soc_code]
            .loc[:, ["element_name", "data_value"]]
            .set_index("element_name")
            .to_dict()
        )
        work_contexts = work_contexts["data_value"]

        return {
            "soc_code": soc_code,
            "title": occ["title"],
            "description": occ["description"],
            # RIASEC (existing)
            "riasec": riasec,
            "riasec_primary": highpoints[0] if highpoints else None,
            "highpoint_codes": highpoints,
            # Work Styles (new)
            "work_styles": work_styles,
            # Big Five derived (new)
            "big_five": big_five,
            "hexaco": hexaco,
            # Work Values (new)
            "work_values": work_values,
            "work_value_highpoints": work_value_highpoints,
            # Tasks
            "tasks": tasks,
            # Work Contexts
            "work_contexts": work_contexts,
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

    def save_profiles_to_jsons(self, save_path: str):
        occupations = self.load_occupations()
        for soc_code in tqdm(occupations["soc_code"]):
            try:
                profile = self.get_occupation_profile(soc_code)
                occupation_title = profile["title"]
                # replace the spaces with underscores and lowercase the title.
                occupation_title = occupation_title.replace(" ", "_").lower()
                if "/" in occupation_title:
                    occupation_title = occupation_title.replace("/", "__")
                filename = f"{occupation_title}.json"
                with open(f"{save_path}/{filename}", "w") as f:
                    import json

                    json.dump(profile, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save profile for {soc_code}: {e}")


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

    loader.save_profiles_to_jsons("data/occupation_profiles")

    # print("\n=== Sample Occupation Profile ===")
    # profile = loader.get_occupation_profile("29-1141.00")  # Registered Nurses
    # print(f"Title: {profile['title']}")
    # print(f"RIASEC: {profile['riasec']}")
    # print(f"Primary: {profile['riasec_primary']}")
    # print(f"High-points: {profile['highpoint_codes']}")
    # print(f"Tasks: {len(profile['tasks'])} tasks")
    # print(f"Work Contexts: {len(profile['work_contexts'])} contexts")

    # print("\n=== Work Styles (21 traits) ===")
    # for trait, score in sorted(profile["work_styles"].items()):
    #     print(f"  {trait}: {score:.2f}")

    # print("\n=== Big Five (derived) ===")
    # for dim, score in sorted(profile["big_five"].items()):
    #     print(f"  {dim}: {score:.2f}")

    # print("\n=== HEXACO ===")
    # for dim, score in sorted(profile["hexaco"].items()):
    #     print(f"  {dim}: {score:.2f}")

    # print("\n=== Work Values (6 values) ===")
    # for value, score in sorted(profile["work_values"].items()):
    #     print(f"  {value}: {score:.2f}")
    # print(f"Work Value High-points: {profile['work_value_highpoints']}")

    # print("\n=== Work Contexts (sample 3)===")
    # for context, freq in profile["work_contexts"].items():
    #     print(f"  {context}: {freq}")
