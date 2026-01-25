# Phase 6 Workflow: ML-Ready Datasets Implementation

> **Status**: Ready for Implementation
>
> **Priority**: MEDIUM
>
> **Reference**: [PSYCHOMETRICS_DATA.md § Phase 6](../reference/PSYCHOMETRICS_DATA.md#phase-6-ml-ready-datasets)
>
> **Last Updated**: 2025-01-25

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why ML Datasets for LM-VECTOR](#2-why-ml-datasets-for-lm-vector)
3. [Data Sources Overview](#3-data-sources-overview)
4. [Implementation Tasks](#4-implementation-tasks)
5. [Data Artifacts](#5-data-artifacts)
6. [Code Implementation](#6-code-implementation)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Verification Commands](#9-verification-commands)

---

## 1. Executive Summary

### Goal
Implement dataset loaders for large-scale pre-scored psychometric response datasets from:
- **HuggingFace**: Pre-scored IPIP-NEO datasets
- **Open Psychometrics**: Raw response data (1M+ respondents)
- **OSF**: Authoritative validation datasets from academic research

### Scope
| In Scope | Out of Scope |
|----------|--------------|
| HuggingFace IPIP-NEO-120/300 scores | Full HuggingFace catalog scraping |
| Open Psychometrics: IPIP, HEXACO, RIASEC, SD3 | Datasets requiring paid access |
| OSF: Johnson IPIP-NEO, Schwartz PVQ-RR | Non-English language datasets |
| Dataset catalog and metadata tracking | Real-time API streaming |
| Basic EDA and summary statistics | Full preprocessing pipelines |

### Timeline Estimate
- **Data Download**: 2-4 hours (large files, network-dependent)
- **Loader Implementation**: 4-6 hours
- **Catalog Generation**: 2 hours
- **Testing**: 2-3 hours
- **Documentation**: 1-2 hours

---

## 2. Understanding ML-Ready Datasets vs. Instruments

### 2.1 The Key Distinction

**Instruments (Phases 1-5)** = **The Survey Questions**
- HEXACO-100 items: *"I would be quite bored by a visit to an art gallery"*
- IPIP-NEO-120 items: *"Worry about things"*
- RIASEC items: *"Build kitchen cabinets"*

**ML Datasets (Phase 6)** = **Thousands of People's Answers to Those Questions**
- HuggingFace: 619,000 people's responses to IPIP-NEO-120
- Open Psychometrics: 1,015,342 people's responses to IPIP Big Five
- OSF: 307,313 validated responses from academic research

### 2.2 Overlap Analysis

| Phase | What It Provides | Example | ML Dataset Equivalent |
|-------|------------------|---------|----------------------|
| **Phase 1: O*NET** | Occupation profiles (RIASEC scores) | "Software Developer: R=2.5, I=4.8, A=3.2..." | Open Psychometrics RIASEC (145K responses) |
| **Phase 2: HEXACO** | 100 personality items + scoring | Items + how to score | Open Psychometrics HEXACO (22K actual responses) |
| **Phase 3: IPIP-NEO** | 120 Big Five items + scoring | Items + how to score | HuggingFace IPIP-120 (619K actual responses) |
| **Phase 4: Schwartz** | 21 value items + scoring | Items + how to score | OSF PVQ-RR (multilingual responses) |
| **Phase 5: Dark Personality** | 28 SD4 items + scoring | Items + how to score | Open Psychometrics SD3 (18K actual responses) |
| **Phase 6: ML Datasets** | **Real human response data** | Actual filled-out surveys | **This IS the actual data** |

### 2.3 Relationship: Complementary, Not Redundant

**Instruments provide:**
- ✅ The *questions* to ask
- ✅ The *scoring algorithms*
- ✅ The *theoretical framework*

**ML Datasets provide:**
- ✅ *Real human distributions* (what do actual scores look like?)
- ✅ *Training data* for ML models
- ✅ *Validation data* (are our derived scores realistic?)
- ✅ *Population norms* (percentiles)

**Analogy:**
```
Instrument (IPIP-NEO-120) = A thermometer
ML Dataset              = 1 million temperature readings from around the world

You need the thermometer to take measurements,
but the dataset tells you what temperatures are "normal"
```

### 2.4 Specific Use Cases for LM-VECTOR

#### Use Case 1: Validating O*NET → Big Five Derivation

**Without ML Dataset:**
```python
# We derive Big Five from O*NET Work Styles
onet_profile = get_occupation_profile("15-1252.00")  # Software Developer
big_five = onet_profile["big_five"]
# {'O': 4.2, 'C': 3.8, 'E': 2.5, 'A': 3.0, 'N': 2.8}

# But is this realistic? We don't know!
```

**With ML Dataset:**
```python
# Load 619K real IPIP-NEO responses
ipip_data = load_huggingface("ipip_120_train")

# Compare our derived scores to real population
real_O_mean = ipip_data['O'].mean()  # e.g., 3.5
real_O_std = ipip_data['O'].std()    # e.g., 0.8

derived_O = 4.2
z_score = (derived_O - real_O_mean) / real_O_std  # 0.875 (87th percentile)

# Now we can say: "This occupation scores in the 87th percentile for Openness"
```

#### Use Case 2: Generating Realistic Edge Cases

**Without ML Dataset:**
```python
# Generate an "unusual" persona
persona = {
    'O': 5.0,  # Maximum Openness
    'C': 1.0,  # Minimum Conscientiousness
    'E': 5.0,  # Maximum Extraversion
}
# But does this combination exist in real humans?
```

**With ML Dataset:**
```python
# Find real humans with this profile
unusual_profiles = ipip_data[
    (ipip_data['O'] > 4.5) &
    (ipip_data['C'] < 1.5) &
    (ipip_data['E'] > 4.5)
]
print(f"Found {len(unusual_profiles)} real people with this profile")
# If 0, this combination may not be realistic!
```

#### Use Case 3: Training ML Models

**Without ML Dataset:**
```python
# Hand-code mappings
work_style_to_big_five = {
    "Innovation": {"O": +0.8, "C": +0.2},
    "Attention to Detail": {"C": +0.9, "O": -0.1},
    # ... manual mapping of 16 work styles
}
```

**With ML Dataset:**
```python
# Train a model to learn the mapping
from sklearn.ensemble import RandomForestRegressor

# Features: O*NET Work Styles (16 dimensions)
# Target: IPIP-NEO Big Five scores (5 dimensions)

model = RandomForestRegressor()
model.fit(X_work_styles, y_big_five)

# Model learns optimal mapping from 619K examples!
```

---

## 3. Why ML Datasets for LM-VECTOR

### Pre-Scored Data Benefits

1. **Training Data**: Large-scale responses for persona vector model training
2. **Validation**: Ground truth for evaluating O*NET → Big Five derivations
3. **Norming**: Population distributions for percentile conversions
4. **Edge Cases**: Extreme profiles for adversarial testing

### Data Sources Value Proposition

| Source | Volume | Quality | Use Case |
|--------|--------|---------|----------|
| **HuggingFace** | ~300K responses | Pre-scored, clean | Direct model training |
| **Open Psychometrics** | 1M+ responses | Raw, requires cleaning | Validation, norming |
| **OSF** | 300K+ responses | Research-grade | Authoritative validation |

### Integration with O*NET Pipeline

```
┌─────────────────────────────────────────────────────┐
│ O*NET Occupational Profiles                        │
│ (RIASEC, Work Styles, Work Values)                 │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│ ML Datasets: Pre-Scored Personality Responses      │
│ - IPIP-NEO (Big Five)                              │
│ - HEXACO (6-factor)                                │
│ - RIASEC (Vocational interests)                    │
│ - SD3/SD4 (Dark personality)                       │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│ Persona Vector Generation & Validation             │
│ - Compare derived vs. measured Big Five            │
│ - Population norming for percentiles               │
│ - Edge case detection for adversarial testing      │
└─────────────────────────────────────────────────────┘
```

---

## 3. Data Sources Overview

### 3.1 HuggingFace Datasets

**Repository**: `ecorbari/IPIP120-SCORES`, `ecorbari/IPIP300-SCORES`

| Dataset | Responses | Features | Format |
|---------|-----------|----------|--------|
| IPIP-120 Scores | ~619K | 120 items + 5 domain scores + 30 facet scores | Parquet/CSV |
| IPIP-300 Scores | ~307K | 300 items + 5 domain scores + 30 facet scores | Parquet/CSV |

**License**: Check dataset card (likely permissive for research)

**Citation**: Johnson, J. A. (2014). *Journal of Research in Personality, 51*, 78-89.

---

### 3.2 Open Psychometrics

**Base URL**: http://openpsychometrics.org/_rawdata/

| Dataset | Responses | Items | File Size | Description |
|---------|-----------|-------|-----------|-------------|
| **IPIP Big Five** | 1,015,342 | 50 | ~100 MB | IPIP-NEO-FFM responses |
| **HEXACO** | 22,786 | 100 | ~5 MB | HEXACO-100 responses |
| **RIASEC** | 145,828 | 48 | ~15 MB | Vocational interest responses |
| **SD3 (Dark Triad)** | 18,192 | 27 | ~2 MB | Short Dark Triad responses |

**License**: Creative Commons (attribution required)

**Data Format**: CSV with columns: `Q1, Q2, ..., Qn, age, gender, country, ...`

**Codebooks**: Included in ZIP files (define item order, reverse scoring)

---

### 3.3 OSF Repositories

**Platform**: Open Science Framework (https://osf.io)

| Repository | OSF ID | Dataset | Size | Description |
|------------|--------|---------|------|-------------|
| Johnson IPIP-NEO | `tbmh5` | 307,313 responses | ~50 MB | IPIP-NEO validation sample |
| IPIP-NEO-120 Dev | `ncmg9` | Subset of above | ~20 MB | Development/validation split |
| Schwartz PVQ-RR | `w9as3` | Multi-language | ~10 MB | 19-value refined model, 47 languages |

**License**: CC BY 4.0 (attribution required)

**Data Format**: Mixed (CSV, SPSS, Excel) - varies by repository

---

## 4. Implementation Tasks

### Task 4.1: Create Data Directories

```bash
mkdir -p data/psychometrics/ml_datasets/{huggingface,open_psychometrics,osf}/{raw,processed}
mkdir -p data/psychometrics/ml_datasets/metadata
```

**Output Structure**:
```
data/psychometrics/ml_datasets/
├── huggingface/
│   ├── raw/                          # Downloaded parquet/CSV files
│   ├── processed/                    # Cleaned, standardized datasets
│   └── metadata/
│       └── dataset_cards.json        # HF dataset metadata
├── open_psychometrics/
│   ├── raw/                          # Downloaded ZIP files + extracted CSVs
│   ├── processed/                    # Cleaned responses
│   └── codebooks/                    # Item definitions, scoring keys
├── osf/
│   ├── raw/                          # Downloaded files (mixed formats)
│   ├── processed/                    # Standardized CSVs
│   └── metadata/
│       └── osf_project_info.json     # OSF project metadata
└── catalog.json                      # Master catalog of all datasets
```

---

### Task 4.2: Download HuggingFace Datasets

**Implementation**: `src/pvx/data/loaders/huggingface_loader.py`

```python
from datasets import load_dataset
import pandas as pd
from pathlib import Path

class HuggingFaceDatasetLoader:
    """Loader for pre-scored IPIP-NEO datasets from HuggingFace"""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/psychometrics/ml_datasets/huggingface")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_ipip_scores(self, version: str = "120") -> dict[str, pd.DataFrame]:
        """
        Download IPIP-NEO pre-scored datasets

        Args:
            version: "120" or "300"

        Returns:
            dict with 'train' and 'test' DataFrames
        """
        repo_map = {
            "120": "ecorbari/IPIP120-SCORES",
            "300": "ecorbari/IPIP300-SCORES"
        }

        if version not in repo_map:
            raise ValueError(f"Invalid version: {version}. Choose '120' or '300'")

        repo = repo_map[version]
        dataset = load_dataset(repo)

        # Save to cache
        result = {}
        for split in dataset.keys():
            df = dataset[split].to_pandas()

            # Save as parquet (efficient) and CSV (human-readable)
            parquet_path = self.cache_dir / f"ipip_neo_{version}_{split}.parquet"
            csv_path = self.cache_dir / f"ipip_neo_{version}_{split}.csv"

            df.to_parquet(parquet_path, index=False)
            df.to_csv(csv_path, index=False)

            result[split] = df

        return result
```

**Execution**:
```python
loader = HuggingFaceDatasetLoader()
ipip120 = loader.download_ipip_scores("120")
print(f"Train: {len(ipip120['train'])} rows")
print(f"Columns: {list(ipip120['train'].columns)[:10]}")
```

---

### Task 4.3: Download Open Psychometrics Data

**Implementation**: `src/pvx/data/loaders/openpsycho_loader.py`

```python
import requests
import zipfile
from pathlib import Path
import pandas as pd

class OpenPsychometricsLoader:
    """Loader for raw response data from Open Psychometrics Project"""

    BASE_URL = "http://openpsychometrics.org/_rawdata/"

    DATASETS = {
        "ipip_big_five": {
            "url": "IPIP-FFM-data-8Nov2018.zip",
            "description": "IPIP Big Five - 1,015,342 responses",
            "codebook": "codebook.txt"
        },
        "hexaco": {
            "url": "HEXACO.zip",
            "description": "HEXACO - 22,786 responses",
            "codebook": "codebook.txt"
        },
        "riasec": {
            "url": "RIASEC_data12Dec2018.zip",
            "description": "RIASEC - 145,828 responses",
            "codebook": "codebook.txt"
        },
        "sd3": {
            "url": "SD3.zip",
            "description": "Short Dark Triad - 18,192 responses",
            "codebook": "codebook.txt"
        }
    }

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/psychometrics/ml_datasets/open_psychometrics")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name: str) -> Path:
        """
        Download and extract Open Psychometrics dataset

        Args:
            dataset_name: One of ["ipip_big_five", "hexaco", "riasec", "sd3"]

        Returns:
            Path to extracted directory
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        info = self.DATASETS[dataset_name]
        url = self.BASE_URL + info["url"]

        # Download
        zip_path = self.cache_dir / "raw" / info["url"]
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {dataset_name}...")
        print(f"  {info['description']}")

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract
        extract_dir = self.cache_dir / "raw" / dataset_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"  ✓ Extracted to {extract_dir}")

        return extract_dir

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load extracted dataset into DataFrame"""
        extract_dir = self.cache_dir / "raw" / dataset_name

        # Find CSV file (usually named data.csv or similar)
        csv_files = list(extract_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {extract_dir}")

        df = pd.read_csv(csv_files[0], sep='\t', low_memory=False)
        return df
```

**Execution**:
```python
loader = OpenPsychometricsLoader()
loader.download_dataset("ipip_big_five")
df = loader.load_dataset("ipip_big_five")
print(f"Loaded {len(df)} responses with {len(df.columns)} columns")
```

---

### Task 4.4: Download OSF Datasets

**Implementation**: `src/pvx/data/loaders/osf_loader.py`

```python
import requests
from pathlib import Path
import pandas as pd
import zipfile

class OSFDatasetLoader:
    """Loader for datasets from Open Science Framework"""

    REPOSITORIES = {
        "johnson_ipip_neo": {
            "osf_id": "tbmh5",
            "url": "https://osf.io/tbmh5/download",
            "description": "Johnson's IPIP-NEO validation data (307,313 cases)",
            "format": "zip"
        },
        "ipip_neo_120_validation": {
            "osf_id": "ncmg9",
            "url": "https://osf.io/ncmg9/download",
            "description": "IPIP-NEO-120 development data",
            "format": "zip"
        },
        "schwartz_pvq_rr": {
            "osf_id": "w9as3",
            "url": "https://osf.io/w9as3/download",
            "description": "PVQ-RR 19-values (47 languages)",
            "format": "zip"
        }
    }

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/psychometrics/ml_datasets/osf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_repository(self, repo_name: str) -> Path:
        """
        Download OSF repository

        Args:
            repo_name: One of REPOSITORIES keys

        Returns:
            Path to downloaded/extracted directory
        """
        if repo_name not in self.REPOSITORIES:
            raise ValueError(f"Unknown repository: {repo_name}")

        info = self.REPOSITORIES[repo_name]

        # Download
        download_path = self.cache_dir / "raw" / f"{repo_name}.{info['format']}"
        download_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_name}...")
        print(f"  OSF ID: {info['osf_id']}")
        print(f"  {info['description']}")

        response = requests.get(info['url'], stream=True, timeout=300)
        response.raise_for_status()

        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract if ZIP
        if info['format'] == 'zip':
            extract_dir = self.cache_dir / "raw" / repo_name
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"  ✓ Extracted to {extract_dir}")
            return extract_dir
        else:
            print(f"  ✓ Downloaded to {download_path}")
            return download_path
```

**Execution**:
```python
loader = OSFDatasetLoader()
path = loader.download_repository("johnson_ipip_neo")
print(f"Data available at: {path}")
```

---

### Task 4.5: Create Dataset Catalog

**Implementation**: `src/pvx/data/loaders/dataset_catalog.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class DatasetCatalog:
    """Master catalog of all ML datasets"""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("data/psychometrics/ml_datasets")
        self.catalog_path = self.base_dir / "catalog.json"

    def scan_datasets(self) -> Dict[str, Any]:
        """Scan all dataset directories and generate catalog"""
        catalog = {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "sources": {
                "huggingface": self._scan_huggingface(),
                "open_psychometrics": self._scan_openpsycho(),
                "osf": self._scan_osf()
            }
        }

        return catalog

    def _scan_huggingface(self) -> Dict[str, Any]:
        """Scan HuggingFace datasets"""
        hf_dir = self.base_dir / "huggingface"
        if not hf_dir.exists():
            return {}

        datasets = {}
        for file in hf_dir.glob("*.parquet"):
            datasets[file.name] = {
                "size_mb": file.stat().st_size / (1024 * 1024),
                "format": "parquet",
                "path": str(file.relative_to(self.base_dir))
            }

        return datasets

    def _scan_openpsycho(self) -> Dict[str, Any]:
        """Scan Open Psychometrics datasets"""
        op_dir = self.base_dir / "open_psychometrics" / "raw"
        if not op_dir.exists():
            return {}

        datasets = {}
        for dataset_dir in op_dir.iterdir():
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob("*.csv"))
                if files:
                    datasets[dataset_dir.name] = {
                        "files": [f.name for f in files],
                        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
                        "path": str(dataset_dir.relative_to(self.base_dir))
                    }

        return datasets

    def _scan_osf(self) -> Dict[str, Any]:
        """Scan OSF datasets"""
        osf_dir = self.base_dir / "osf" / "raw"
        if not osf_dir.exists():
            return {}

        datasets = {}
        for item in osf_dir.iterdir():
            if item.is_dir():
                files = list(item.rglob("*"))
                datasets[item.name] = {
                    "file_count": len([f for f in files if f.is_file()]),
                    "path": str(item.relative_to(self.base_dir))
                }
            elif item.is_file():
                datasets[item.name] = {
                    "size_mb": item.stat().st_size / (1024 * 1024),
                    "path": str(item.relative_to(self.base_dir))
                }

        return datasets

    def save_catalog(self):
        """Generate and save catalog to JSON"""
        catalog = self.scan_datasets()

        with open(self.catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        print(f"✓ Catalog saved to {self.catalog_path}")
        return catalog

    def get_summary(self) -> str:
        """Generate human-readable summary"""
        if not self.catalog_path.exists():
            return "No catalog found. Run save_catalog() first."

        with open(self.catalog_path) as f:
            catalog = json.load(f)

        lines = [
            "=== ML Dataset Catalog Summary ===",
            f"Generated: {catalog['generated']}",
            ""
        ]

        for source, datasets in catalog['sources'].items():
            lines.append(f"\n{source.upper()}:")
            if datasets:
                for name, info in datasets.items():
                    lines.append(f"  - {name}")
                    if 'size_mb' in info:
                        lines.append(f"      Size: {info['size_mb']:.2f} MB")
                    if 'file_count' in info:
                        lines.append(f"      Files: {info['file_count']}")
            else:
                lines.append("  (No datasets found)")

        return "\n".join(lines)
```

**Execution**:
```python
catalog = DatasetCatalog()
catalog.save_catalog()
print(catalog.get_summary())
```

---

## 5. Data Artifacts

### 5.1 Expected Files After Completion

```
data/psychometrics/ml_datasets/
├── huggingface/
│   ├── ipip_neo_120_train.parquet    (~50 MB)
│   ├── ipip_neo_120_train.csv        (~100 MB)
│   ├── ipip_neo_300_train.parquet    (~30 MB)
│   └── ipip_neo_300_train.csv        (~60 MB)
├── open_psychometrics/
│   ├── raw/
│   │   ├── ipip_big_five/
│   │   │   └── data.csv              (~100 MB)
│   │   ├── hexaco/
│   │   │   └── data.csv              (~5 MB)
│   │   ├── riasec/
│   │   │   └── data.csv              (~15 MB)
│   │   └── sd3/
│   │       └── data.csv              (~2 MB)
│   └── codebooks/
│       ├── ipip_big_five_codebook.txt
│       ├── hexaco_codebook.txt
│       └── riasec_codebook.txt
├── osf/
│   ├── raw/
│   │   ├── johnson_ipip_neo/         (multiple files)
│   │   └── schwartz_pvq_rr/          (multiple files)
│   └── processed/
│       └── (cleaned CSV files)
└── catalog.json                       (master catalog)
```

### 5.2 Catalog JSON Schema

```json
{
  "generated": "2025-01-25T12:00:00",
  "version": "1.0",
  "sources": {
    "huggingface": {
      "ipip_neo_120_train.parquet": {
        "size_mb": 50.5,
        "format": "parquet",
        "path": "huggingface/ipip_neo_120_train.parquet"
      }
    },
    "open_psychometrics": {
      "ipip_big_five": {
        "files": ["data.csv", "codebook.txt"],
        "total_size_mb": 102.3,
        "path": "open_psychometrics/raw/ipip_big_five"
      }
    },
    "osf": {
      "johnson_ipip_neo": {
        "file_count": 12,
        "path": "osf/raw/johnson_ipip_neo"
      }
    }
  }
}
```

---

## 6. Code Implementation

### 6.1 Unified Dataset Manager

**File**: `src/pvx/data/ml_dataset_manager.py`

```python
"""
Unified manager for all ML-ready psychometric datasets
"""
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from pvx.data.loaders.huggingface_loader import HuggingFaceDatasetLoader
from pvx.data.loaders.openpsycho_loader import OpenPsychometricsLoader
from pvx.data.loaders.osf_loader import OSFDatasetLoader
from pvx.data.loaders.dataset_catalog import DatasetCatalog


class MLDatasetManager:
    """
    Unified interface for downloading and accessing ML datasets

    Example:
        manager = MLDatasetManager()

        # Download all datasets
        manager.download_all()

        # Load specific dataset
        df = manager.load("huggingface", "ipip_neo_120_train")

        # Get catalog
        summary = manager.get_catalog_summary()
    """

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("data/psychometrics/ml_datasets")

        # Initialize loaders
        self.hf_loader = HuggingFaceDatasetLoader(self.base_dir / "huggingface")
        self.op_loader = OpenPsychometricsLoader(self.base_dir / "open_psychometrics")
        self.osf_loader = OSFDatasetLoader(self.base_dir / "osf")
        self.catalog = DatasetCatalog(self.base_dir)

    def download_huggingface(self, versions: list[str] = None):
        """Download HuggingFace datasets"""
        versions = versions or ["120", "300"]
        for version in versions:
            print(f"\nDownloading IPIP-NEO-{version} from HuggingFace...")
            self.hf_loader.download_ipip_scores(version)

    def download_openpsycho(self, datasets: list[str] = None):
        """Download Open Psychometrics datasets"""
        datasets = datasets or ["ipip_big_five", "hexaco", "riasec", "sd3"]
        for dataset in datasets:
            print(f"\nDownloading {dataset} from Open Psychometrics...")
            self.op_loader.download_dataset(dataset)

    def download_osf(self, repositories: list[str] = None):
        """Download OSF datasets"""
        repositories = repositories or ["johnson_ipip_neo", "schwartz_pvq_rr"]
        for repo in repositories:
            print(f"\nDownloading {repo} from OSF...")
            self.osf_loader.download_repository(repo)

    def download_all(self):
        """Download all datasets from all sources"""
        print("=" * 60)
        print("DOWNLOADING ALL ML DATASETS")
        print("=" * 60)

        try:
            self.download_huggingface()
        except Exception as e:
            print(f"⚠ HuggingFace download failed: {e}")

        try:
            self.download_openpsycho()
        except Exception as e:
            print(f"⚠ Open Psychometrics download failed: {e}")

        try:
            self.download_osf()
        except Exception as e:
            print(f"⚠ OSF download failed: {e}")

        # Generate catalog
        print("\nGenerating dataset catalog...")
        self.catalog.save_catalog()

        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)

    def load(self, source: str, dataset_name: str) -> pd.DataFrame:
        """
        Load a specific dataset

        Args:
            source: "huggingface", "open_psychometrics", or "osf"
            dataset_name: Dataset identifier

        Returns:
            DataFrame with dataset
        """
        if source == "huggingface":
            # Load from parquet
            path = self.base_dir / "huggingface" / f"{dataset_name}.parquet"
            return pd.read_parquet(path)

        elif source == "open_psychometrics":
            return self.op_loader.load_dataset(dataset_name)

        elif source == "osf":
            # OSF datasets vary in format - implement as needed
            raise NotImplementedError("OSF loading requires dataset-specific logic")

        else:
            raise ValueError(f"Unknown source: {source}")

    def get_catalog_summary(self) -> str:
        """Get human-readable catalog summary"""
        return self.catalog.get_summary()
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File**: `tests/unit/test_ml_dataset_manager.py`

```python
import pytest
from pathlib import Path
from pvx.data.ml_dataset_manager import MLDatasetManager

@pytest.fixture
def manager(tmp_path):
    """Create manager with temp directory"""
    return MLDatasetManager(base_dir=tmp_path)

def test_manager_initialization(manager):
    """Test MLDatasetManager initializes correctly"""
    assert manager.base_dir.exists()
    assert manager.hf_loader is not None
    assert manager.op_loader is not None
    assert manager.osf_loader is not None

def test_catalog_creation(manager):
    """Test catalog generation"""
    catalog = manager.catalog.scan_datasets()
    assert "sources" in catalog
    assert "huggingface" in catalog["sources"]
    assert "open_psychometrics" in catalog["sources"]
    assert "osf" in catalog["sources"]

# Add more tests for:
# - Download simulation
# - Loading datasets
# - Catalog summary generation
```

### 7.2 Integration Tests

Test full download → catalog → load pipeline:

```python
def test_full_pipeline():
    """Test complete dataset acquisition pipeline"""
    manager = MLDatasetManager()

    # Download one small dataset
    manager.download_openpsycho(["sd3"])

    # Generate catalog
    catalog = manager.catalog.save_catalog()

    # Verify catalog contains SD3
    assert "sd3" in catalog["sources"]["open_psychometrics"]

    # Load dataset
    df = manager.load("open_psychometrics", "sd3")
    assert len(df) > 0
```

---

## 8. Acceptance Criteria

### Phase 6 Complete When:

- [ ] All three loader classes implemented (`HuggingFaceDatasetLoader`, `OpenPsychometricsLoader`, `OSFDatasetLoader`)
- [ ] `MLDatasetManager` unified interface working
- [ ] `DatasetCatalog` generating accurate metadata
- [ ] At least one dataset successfully downloaded from each source
- [ ] Catalog JSON file generated and valid
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration test demonstrates full pipeline
- [ ] Documentation updated in `STATUS_psychometrics.md`

### Optional Enhancements (Future Work):

- [ ] Automatic dataset versioning tracking
- [ ] Data validation (schema checking)
- [ ] Preprocessing pipelines (cleaning, normalization)
- [ ] Dataset summary statistics (EDA reports)
- [ ] Integration with persona generation pipeline

---

## 9. Verification Commands

### 9.1 Download All Datasets

```bash
uv run python -c "
from pvx.data.ml_dataset_manager import MLDatasetManager
manager = MLDatasetManager()
manager.download_all()
"
```

### 9.2 Verify Catalog

```bash
cat data/psychometrics/ml_datasets/catalog.json | python -m json.tool
```

### 9.3 Test Loading Datasets

```python
from pvx.data.ml_dataset_manager import MLDatasetManager

manager = MLDatasetManager()

# Load HuggingFace dataset
df_hf = manager.load("huggingface", "ipip_neo_120_train")
print(f"HuggingFace IPIP-120: {len(df_hf)} rows, {len(df_hf.columns)} columns")

# Load Open Psychometrics dataset
df_op = manager.load("open_psychometrics", "sd3")
print(f"Open Psychometrics SD3: {len(df_op)} rows, {len(df_op.columns)} columns")
```

### 9.4 Generate Summary Report

```bash
uv run python -c "
from pvx.data.ml_dataset_manager import MLDatasetManager
manager = MLDatasetManager()
print(manager.get_catalog_summary())
"
```

### 9.5 Run Unit Tests

```bash
uv run pytest tests/unit/test_ml_dataset_manager.py -v
```

---

## 10. Usage Examples

### Example 1: Download and Explore IPIP-NEO Data

```python
from pvx.data.ml_dataset_manager import MLDatasetManager
import pandas as pd

manager = MLDatasetManager()

# Download IPIP-NEO-120 scores
manager.download_huggingface(versions=["120"])

# Load training data
df = manager.load("huggingface", "ipip_neo_120_train")

# Basic EDA
print(df.head())
print(df.describe())

# Check domain score distributions
domain_cols = [col for col in df.columns if col.startswith(('N_', 'E_', 'O_', 'A_', 'C_'))]
df[domain_cols].hist(bins=30, figsize=(15, 10))
```

### Example 2: Compare O*NET Derived Big Five with IPIP-NEO Ground Truth

```python
from pvx.data.onet_loader import ONETLoader
from pvx.data.ml_dataset_manager import MLDatasetManager
import pandas as pd

# Get O*NET derived Big Five for occupation
onet = ONETLoader()
profile = onet.get_occupation_profile("15-1252.00")  # Software Developer
onet_big_five = profile["big_five"]

# Load IPIP-NEO ground truth data
manager = MLDatasetManager()
ipip_df = manager.load("huggingface", "ipip_neo_120_train")

# Compare distributions
print("O*NET Derived Big Five (Software Developer):")
print(onet_big_five)

print("\nIPIP-NEO Population Means:")
print(ipip_df[['N', 'E', 'O', 'A', 'C']].mean())
```

### Example 3: Generate Dataset Summary Report

```python
from pvx.data.ml_dataset_manager import MLDatasetManager

manager = MLDatasetManager()

# Generate catalog
manager.catalog.save_catalog()

# Print summary
print(manager.get_catalog_summary())

# Save to markdown
summary_md = f"""# ML Datasets Summary

{manager.get_catalog_summary()}

## Dataset Purposes

- **HuggingFace IPIP-NEO**: Pre-scored Big Five for model training
- **Open Psychometrics**: Large-scale raw responses for validation
- **OSF**: Research-grade authoritative datasets

## Next Steps

1. Implement preprocessing pipelines
2. Integrate with persona generation
3. Validate O*NET → Big Five derivations
"""

with open("data/psychometrics/ml_datasets/README.md", "w") as f:
    f.write(summary_md)
```

---

## 11. Implementation Checklist

### Pre-Implementation
- [ ] Review [PSYCHOMETRICS_DATA.md § Phase 6](../reference/PSYCHOMETRICS_DATA.md#phase-6-ml-ready-datasets)
- [ ] Check network connectivity and disk space (~500 MB needed)
- [ ] Install dependencies: `datasets`, `requests`, `pandas`

### Implementation
- [ ] Task 4.1: Create directory structure
- [ ] Task 4.2: Implement `HuggingFaceDatasetLoader`
- [ ] Task 4.3: Implement `OpenPsychometricsLoader`
- [ ] Task 4.4: Implement `OSFDatasetLoader`
- [ ] Task 4.5: Implement `DatasetCatalog`
- [ ] Task 6.1: Implement `MLDatasetManager`

### Testing
- [ ] Write unit tests for each loader
- [ ] Write integration test for full pipeline
- [ ] Verify catalog generation
- [ ] Test loading from each source

### Documentation
- [ ] Update `STATUS_psychometrics.md` with Phase 6 status
- [ ] Create dataset README with usage examples
- [ ] Document any dataset-specific quirks or issues

### Verification
- [ ] Run all verification commands (Section 9)
- [ ] Generate and review catalog summary
- [ ] Confirm at least 1 dataset from each source is accessible

---

## 12. Notes & Caveats

### Network Requirements
- Total download size: ~500 MB - 1 GB
- Open Psychometrics downloads can be slow
- OSF files may require authentication for some repos

### Data Quality
- **HuggingFace**: Pre-scored, clean, ready for ML
- **Open Psychometrics**: Raw responses, requires validation
- **OSF**: Mixed formats, may need custom parsers

### Future Enhancements
1. **Preprocessing**: Standardize response formats across sources
2. **Validation**: Check for duplicate responses, invalid data
3. **Norming**: Generate percentile tables for score interpretation
4. **Integration**: Connect datasets to persona generation pipeline
5. **Caching**: Implement smart caching to avoid re-downloads

---

★ **Insight ─────────────────────────────────────**

**Why Phase 6 Matters for LM-VECTOR:**

1. **Ground Truth Validation**: Open Psychometrics' 1M+ IPIP responses provide population norms to validate our O*NET → Big Five derivations. We can check if Software Developer work styles map to realistic Big Five distributions.

2. **Training Data at Scale**: HuggingFace's pre-scored datasets enable supervised learning for persona vector models. Instead of hand-crafting mappings, we can learn them from data.

3. **Edge Case Discovery**: Large datasets reveal rare personality combinations (e.g., high Openness + low Agreeableness) useful for adversarial testing. These edge cases help ensure models handle unusual personas robustly.

4. **Benchmark Establishment**: By comparing our generated personas against real human responses, we establish quality benchmarks. A "realistic persona" should fall within observed human distributions.

────────────────────────────────────────────────
