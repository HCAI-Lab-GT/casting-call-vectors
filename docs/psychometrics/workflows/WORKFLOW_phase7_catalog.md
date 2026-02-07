# Phase 7 Workflow: Final Catalog & Verification

> **Status**: Ready for Implementation
>
> **Priority**: MEDIUM
>
> **Reference**: [PSYCHOMETRICS_DATA.md § Phase 7](../reference/PSYCHOMETRICS_DATA.md#phase-7-final-catalog--verification)
>
> **Last Updated**: 2025-01-25

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why a Unified Registry](#2-why-a-unified-registry)
3. [Architecture Overview](#3-architecture-overview)
4. [Implementation Tasks](#4-implementation-tasks)
5. [Data Artifacts](#5-data-artifacts)
6. [Code Implementation](#6-code-implementation)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Verification Commands](#9-verification-commands)

---

## 1. Executive Summary

### Goal
Implement a `PsychometricRegistry` class that provides:
- Unified access to all psychometric instrument loaders
- Complete project catalog and summary generation
- Comprehensive verification and validation tooling
- Ready-to-use API for downstream persona vector research

### Scope
| In Scope | Out of Scope |
|----------|--------------|
| PsychometricRegistry class | New psychometric instruments |
| Unified loader interface | External data downloads |
| Project summary generation | ML model training |
| Comprehensive verification | Production deployment |
| Documentation generation | Web API endpoints |

### Dependencies
This phase integrates all previous phases:
- **Phase 1**: ONETLoader (COMPLETE)
- **Phase 2**: HEXACOLoader (workflow ready)
- **Phase 3**: IPIPNEOLoader (workflow ready)
- **Phase 4**: SchwartzLoader (workflow ready)
- **Phase 5**: DarkPersonalityLoader (workflow ready)
- **Phase 6**: ML Dataset loaders (workflow ready)

---

## 2. Why a Unified Registry

### The Integration Challenge

After implementing phases 1-6, we'll have multiple standalone loaders:

```python
from pvx.data.onet_loader import ONETLoader
from pvx.data.hexaco_loader import HEXACOLoader
from pvx.data.ipip_neo_loader import IPIPNEOLoader
from pvx.data.schwartz_loader import SchwartzLoader
from pvx.data.dark_personality_loader import DarkPersonalityLoader

# This is verbose and error-prone:
onet = ONETLoader()
hexaco = HEXACOLoader()
ipip = IPIPNEOLoader()
schwartz = SchwartzLoader()
dark = DarkPersonalityLoader()
```

### The Registry Solution

A unified registry provides:

```python
from pvx.data.psychometric_registry import PsychometricRegistry

# Single entry point:
registry = PsychometricRegistry()

# Discover available instruments
instruments = registry.list_instruments()
# ['onet', 'hexaco-100', 'hexaco-60', 'ipip-neo-120', 'ipip-neo-300', ...]

# Access any loader
hexaco_items = registry.get_items('hexaco-100')
riasec = registry.get_loader('onet').get_riasec_scores('11-1011.00')

# Generate comprehensive summaries
summary = registry.generate_summary()
```

### Benefits for LM-VECTOR

1. **Simplified Research Workflow**: Single import for all psychometric data
2. **Consistency**: Standardized interface across all instruments
3. **Discoverability**: Easy to find available instruments and their capabilities
4. **Validation**: Built-in verification that all loaders work correctly
5. **Documentation**: Auto-generated catalogs and usage guides

---

## 3. Architecture Overview

### Component Structure

```
src/pvx/data/
├── onet_loader.py              # Phase 1 (COMPLETE)
├── hexaco_loader.py            # Phase 2
├── ipip_neo_loader.py          # Phase 3
├── schwartz_loader.py          # Phase 4
├── dark_personality_loader.py  # Phase 5
├── ml_dataset_loader.py        # Phase 6
└── psychometric_registry.py    # Phase 7 (THIS WORKFLOW)
```

### Registry Class Design

```python
class PsychometricRegistry:
    """
    Unified registry for all psychometric instruments.

    Provides:
    - Lazy loading of instrument loaders
    - Standardized interface for item access
    - Comprehensive project summaries
    - Validation and verification
    """

    def __init__(self, data_dir: Path | None = None):
        """Initialize registry with optional data directory override."""

    def list_instruments(self) -> list[str]:
        """Get list of available instrument IDs."""

    def get_loader(self, instrument_id: str) -> Any:
        """Get the loader instance for a specific instrument."""

    def get_items(self, instrument_id: str) -> list[dict]:
        """Get items for a specific instrument."""

    def get_instrument_info(self, instrument_id: str) -> dict:
        """Get metadata about an instrument."""

    def generate_summary(self) -> dict:
        """Generate comprehensive project summary."""

    def verify_all(self) -> dict:
        """Run verification on all loaders."""

    def export_catalog(self, output_path: Path) -> None:
        """Export complete catalog as JSON."""
```

### Instrument ID Convention

Standardized naming for all instruments:

| Instrument ID | Loader Class | Items | Description |
|---------------|--------------|-------|-------------|
| `onet` | ONETLoader | Variable | O*NET occupational data |
| `hexaco-100` | HEXACOLoader | 100 | HEXACO-PI-R full version |
| `hexaco-60` | HEXACOLoader | 60 | HEXACO brief version |
| `ipip-neo-120` | IPIPNEOLoader | 120 | IPIP-NEO medium version |
| `ipip-neo-300` | IPIPNEOLoader | 300 | IPIP-NEO full version |
| `schwartz-pvq21` | SchwartzLoader | 21 | Portrait Values Questionnaire |
| `schwartz-pvq40` | SchwartzLoader | 40 | Extended PVQ |
| `sd3` | DarkPersonalityLoader | 27 | Short Dark Triad |
| `sd4` | DarkPersonalityLoader | 28 | Short Dark Tetrad |

---

## 4. Implementation Tasks

### Task 4.1: Create Registry Class Skeleton

**File**: `src/pvx/data/psychometric_registry.py`

```python
"""
Psychometric Instrument Registry

Unified access point for all psychometric data loaders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

# Import all loaders (will be implemented in phases 2-6)
from pvx.data.onet_loader import ONETLoader

try:
    from pvx.data.hexaco_loader import HEXACOLoader
except ImportError:
    HEXACOLoader = None

try:
    from pvx.data.ipip_neo_loader import IPIPNEOLoader
except ImportError:
    IPIPNEOLoader = None

try:
    from pvx.data.schwartz_loader import SchwartzLoader
except ImportError:
    SchwartzLoader = None

try:
    from pvx.data.dark_personality_loader import DarkPersonalityLoader
except ImportError:
    DarkPersonalityLoader = None


class PsychometricRegistry:
    """Central registry for all psychometric instruments."""

    # Instrument metadata catalog
    INSTRUMENTS = {
        "onet": {
            "name": "O*NET Occupational Database",
            "loader_class": "ONETLoader",
            "data_type": "occupational",
            "available": True,
        },
        "hexaco-100": {
            "name": "HEXACO-PI-R 100",
            "loader_class": "HEXACOLoader",
            "data_type": "personality",
            "available": HEXACOLoader is not None,
            "items": 100,
        },
        "hexaco-60": {
            "name": "HEXACO-60",
            "loader_class": "HEXACOLoader",
            "data_type": "personality",
            "available": HEXACOLoader is not None,
            "items": 60,
        },
        "ipip-neo-120": {
            "name": "IPIP-NEO-120",
            "loader_class": "IPIPNEOLoader",
            "data_type": "personality",
            "available": IPIPNEOLoader is not None,
            "items": 120,
        },
        "ipip-neo-300": {
            "name": "IPIP-NEO-300",
            "loader_class": "IPIPNEOLoader",
            "data_type": "personality",
            "available": IPIPNEOLoader is not None,
            "items": 300,
        },
        "schwartz-pvq21": {
            "name": "Schwartz PVQ-21",
            "loader_class": "SchwartzLoader",
            "data_type": "values",
            "available": SchwartzLoader is not None,
            "items": 21,
        },
        "schwartz-pvq40": {
            "name": "Schwartz PVQ-40",
            "loader_class": "SchwartzLoader",
            "data_type": "values",
            "available": SchwartzLoader is not None,
            "items": 40,
        },
        "sd3": {
            "name": "Short Dark Triad (SD3)",
            "loader_class": "DarkPersonalityLoader",
            "data_type": "dark_personality",
            "available": DarkPersonalityLoader is not None,
            "items": 27,
        },
        "sd4": {
            "name": "Short Dark Tetrad (SD4)",
            "loader_class": "DarkPersonalityLoader",
            "data_type": "dark_personality",
            "available": DarkPersonalityLoader is not None,
            "items": 28,
        },
    }

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize psychometric registry.

        Args:
            data_dir: Base directory for psychometric data.
                      Defaults to data/psychometrics/
        """
        if data_dir is None:
            self.data_dir = Path("data/psychometrics")
        else:
            self.data_dir = Path(data_dir)

        # Lazy-loaded loader instances
        self._loaders: dict[str, Any] = {}
```

---

### Task 4.2: Implement Core Registry Methods

Add these methods to the `PsychometricRegistry` class:

```python
def list_instruments(
    self,
    available_only: bool = True,
    data_type: Literal["all", "personality", "values", "occupational", "dark_personality"] = "all"
) -> list[str]:
    """
    List available instrument IDs.

    Args:
        available_only: If True, only return instruments with loaders installed
        data_type: Filter by data type category

    Returns:
        List of instrument IDs
    """
    instruments = []
    for inst_id, info in self.INSTRUMENTS.items():
        # Filter by availability
        if available_only and not info.get("available", False):
            continue

        # Filter by data type
        if data_type != "all" and info.get("data_type") != data_type:
            continue

        instruments.append(inst_id)

    return instruments

def get_loader(self, instrument_id: str) -> Any:
    """
    Get loader instance for an instrument.

    Args:
        instrument_id: Instrument identifier (e.g., 'hexaco-100')

    Returns:
        Loader instance

    Raises:
        ValueError: If instrument_id is unknown
        RuntimeError: If loader class is not available
    """
    if instrument_id not in self.INSTRUMENTS:
        valid = ", ".join(self.INSTRUMENTS.keys())
        raise ValueError(f"Unknown instrument: {instrument_id}. Valid: {valid}")

    # Return cached loader if exists
    if instrument_id in self._loaders:
        return self._loaders[instrument_id]

    info = self.INSTRUMENTS[instrument_id]

    if not info.get("available", False):
        raise RuntimeError(
            f"Loader for '{instrument_id}' is not available. "
            f"Implement {info['loader_class']} first."
        )

    # Instantiate loader
    loader_class_name = info["loader_class"]

    if loader_class_name == "ONETLoader":
        loader = ONETLoader()
    elif loader_class_name == "HEXACOLoader":
        loader = HEXACOLoader(data_dir=self.data_dir / "hexaco")
    elif loader_class_name == "IPIPNEOLoader":
        loader = IPIPNEOLoader(data_dir=self.data_dir / "ipip_neo")
    elif loader_class_name == "SchwartzLoader":
        loader = SchwartzLoader(data_dir=self.data_dir / "schwartz")
    elif loader_class_name == "DarkPersonalityLoader":
        loader = DarkPersonalityLoader(data_dir=self.data_dir / "dark_personality")
    else:
        raise RuntimeError(f"Unknown loader class: {loader_class_name}")

    # Cache and return
    self._loaders[instrument_id] = loader
    return loader

def get_items(self, instrument_id: str, **kwargs) -> list[dict]:
    """
    Get items for an instrument.

    Args:
        instrument_id: Instrument identifier
        **kwargs: Additional arguments passed to loader's get_items()

    Returns:
        List of item dictionaries
    """
    loader = self.get_loader(instrument_id)

    # Handle version-specific instruments
    if instrument_id == "hexaco-100":
        return loader.get_items(version="100")
    elif instrument_id == "hexaco-60":
        return loader.get_items(version="60")
    elif instrument_id == "ipip-neo-120":
        return loader.get_items(version="120")
    elif instrument_id == "ipip-neo-300":
        return loader.get_items(version="300")
    elif instrument_id in ["schwartz-pvq21", "schwartz-pvq40"]:
        version = "21" if "21" in instrument_id else "40"
        return loader.get_items(version=version)
    elif instrument_id in ["sd3", "sd4"]:
        return loader.get_items(version=instrument_id.upper())
    else:
        # Generic case
        if hasattr(loader, 'get_items'):
            return loader.get_items(**kwargs)
        else:
            raise NotImplementedError(f"{instrument_id} does not support get_items()")

def get_instrument_info(self, instrument_id: str) -> dict:
    """
    Get metadata about an instrument.

    Args:
        instrument_id: Instrument identifier

    Returns:
        Dictionary with instrument metadata
    """
    if instrument_id not in self.INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {instrument_id}")

    return self.INSTRUMENTS[instrument_id].copy()
```

---

### Task 4.3: Implement Summary Generation

```python
def generate_summary(self) -> dict:
    """
    Generate comprehensive summary of all instruments.

    Returns:
        Dictionary with:
        - instruments: Dict mapping instrument_id to stats
        - totals: Aggregate statistics
        - status: Availability status for each instrument
    """
    from datetime import datetime

    summary = {
        "generated": datetime.now().isoformat(),
        "project": "LM-VECTOR Psychometrics Data Repository",
        "instruments": {},
        "totals": {
            "total_instruments": len(self.INSTRUMENTS),
            "available_instruments": 0,
            "total_items": 0,
        },
        "by_category": {}
    }

    for inst_id, info in self.INSTRUMENTS.items():
        inst_summary = {
            "name": info["name"],
            "data_type": info["data_type"],
            "available": info.get("available", False),
        }

        if info.get("available", False):
            summary["totals"]["available_instruments"] += 1

            try:
                items = self.get_items(inst_id)
                item_count = len(items)
                inst_summary["item_count"] = item_count
                summary["totals"]["total_items"] += item_count
            except Exception as e:
                inst_summary["error"] = str(e)
                inst_summary["item_count"] = 0
        else:
            inst_summary["item_count"] = 0

        summary["instruments"][inst_id] = inst_summary

        # Aggregate by category
        category = info["data_type"]
        if category not in summary["by_category"]:
            summary["by_category"][category] = {
                "instruments": [],
                "total_items": 0
            }
        summary["by_category"][category]["instruments"].append(inst_id)
        summary["by_category"][category]["total_items"] += inst_summary.get("item_count", 0)

    return summary

def export_catalog(self, output_path: Path | str) -> None:
    """
    Export complete catalog as JSON.

    Args:
        output_path: Path to save catalog JSON
    """
    summary = self.generate_summary()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Catalog exported to {output_path}")

def export_markdown_summary(self, output_path: Path | str) -> None:
    """
    Export human-readable markdown summary.

    Args:
        output_path: Path to save markdown file
    """
    summary = self.generate_summary()
    output_path = Path(output_path)

    md = f"""# LM-VECTOR Psychometrics Data Repository

**Generated**: {summary['generated']}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Instruments | {summary['totals']['total_instruments']} |
| Available Instruments | {summary['totals']['available_instruments']} |
| Total Items | {summary['totals']['total_items']} |

## Instruments by Category

"""

    for category, cat_info in summary['by_category'].items():
        md += f"### {category.replace('_', ' ').title()}\n\n"
        md += f"- **Total Items**: {cat_info['total_items']}\n"
        md += "- **Instruments**:\n"

        for inst_id in cat_info['instruments']:
            inst = summary['instruments'][inst_id]
            status = "✓" if inst['available'] else "✗"
            md += f"  - {status} `{inst_id}`: {inst['name']} ({inst.get('item_count', 0)} items)\n"

        md += "\n"

    md += """## Usage

```python
from pvx.data.psychometric_registry import PsychometricRegistry

# Initialize registry
registry = PsychometricRegistry()

# List available instruments
instruments = registry.list_instruments()

# Get items from any instrument
hexaco_items = registry.get_items('hexaco-100')
ipip_items = registry.get_items('ipip-neo-120')

# Access specific loader
onet = registry.get_loader('onet')
profile = onet.get_occupation_profile('11-1011.00')
```

## Data Sources

See individual workflow documents for data source citations and licensing.
"""

    with open(output_path, 'w') as f:
        f.write(md)

    print(f"✓ Markdown summary exported to {output_path}")
```

---

### Task 4.4: Implement Verification

```python
def verify_all(self, verbose: bool = False) -> dict:
    """
    Run verification on all available loaders.

    Args:
        verbose: If True, print detailed verification output

    Returns:
        Dictionary with verification results for each instrument
    """
    results = {}

    for inst_id in self.list_instruments(available_only=True):
        if verbose:
            print(f"Verifying {inst_id}...")

        try:
            # Try to load items
            items = self.get_items(inst_id)

            result = {
                "status": "pass",
                "item_count": len(items),
                "sample_item": items[0] if items else None
            }

            if verbose:
                print(f"  ✓ {len(items)} items loaded")

        except Exception as e:
            result = {
                "status": "fail",
                "error": str(e)
            }

            if verbose:
                print(f"  ✗ Failed: {e}")

        results[inst_id] = result

    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "pass")
    total = len(results)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Verification: {passed}/{total} passed")
        print(f"{'='*50}")

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed
        },
        "details": results
    }
```

---

### Task 4.5: Update Package Exports

**File**: `src/pvx/data/__init__.py`

```python
"""Psychometric data loaders for LM-VECTOR."""

from pvx.data.onet_loader import ONETLoader
from pvx.data.psychometric_registry import PsychometricRegistry

# Optional imports (may not be available yet)
try:
    from pvx.data.hexaco_loader import HEXACOLoader
except ImportError:
    HEXACOLoader = None

try:
    from pvx.data.ipip_neo_loader import IPIPNEOLoader
except ImportError:
    IPIPNEOLoader = None

try:
    from pvx.data.schwartz_loader import SchwartzLoader
except ImportError:
    SchwartzLoader = None

try:
    from pvx.data.dark_personality_loader import DarkPersonalityLoader
except ImportError:
    DarkPersonalityLoader = None


__all__ = [
    "ONETLoader",
    "PsychometricRegistry",
    "HEXACOLoader",
    "IPIPNEOLoader",
    "SchwartzLoader",
    "DarkPersonalityLoader",
]
```

---

## 5. Data Artifacts

### 5.1 Generated Catalog Files

| Artifact | Location | Format | Purpose |
|----------|----------|--------|---------|
| JSON Catalog | `data/psychometrics/catalog.json` | JSON | Machine-readable summary |
| Markdown Summary | `data/psychometrics/README.md` | Markdown | Human-readable overview |
| Verification Report | `data/psychometrics/verification_report.json` | JSON | Loader test results |

### 5.2 Catalog Schema

**File**: `data/psychometrics/catalog.json`

```json
{
  "generated": "2025-01-25T10:00:00",
  "project": "LM-VECTOR Psychometrics Data Repository",
  "instruments": {
    "onet": {
      "name": "O*NET Occupational Database",
      "data_type": "occupational",
      "available": true,
      "item_count": 0
    },
    "hexaco-100": {
      "name": "HEXACO-PI-R 100",
      "data_type": "personality",
      "available": true,
      "item_count": 100
    }
  },
  "totals": {
    "total_instruments": 9,
    "available_instruments": 2,
    "total_items": 100
  },
  "by_category": {
    "personality": {
      "instruments": ["hexaco-100", "hexaco-60", "ipip-neo-120"],
      "total_items": 280
    },
    "values": {
      "instruments": ["schwartz-pvq21"],
      "total_items": 21
    }
  }
}
```

---

## 6. Code Implementation

See Tasks 4.1-4.5 above for complete implementation.

### Integration Pattern

The registry acts as a facade over all loaders:

```
User Code
    ↓
PsychometricRegistry
    ├→ ONETLoader
    ├→ HEXACOLoader
    ├→ IPIPNEOLoader
    ├→ SchwartzLoader
    └→ DarkPersonalityLoader
```

### Lazy Loading

Loaders are instantiated only when first accessed:

```python
registry = PsychometricRegistry()  # No loaders instantiated yet

hexaco = registry.get_loader('hexaco-100')  # HEXACOLoader instantiated
hexaco2 = registry.get_loader('hexaco-100')  # Returns cached instance
```

---

## 7. Testing Strategy

### 7.1 Test File: `tests/unit/test_psychometric_registry.py`

```python
"""Unit tests for PsychometricRegistry."""

import pytest
from pathlib import Path

from pvx.data.psychometric_registry import PsychometricRegistry


class TestPsychometricRegistry:
    """Tests for PsychometricRegistry class."""

    @pytest.fixture
    def registry(self) -> PsychometricRegistry:
        """Create registry instance."""
        return PsychometricRegistry()

    def test_init(self, registry: PsychometricRegistry):
        """Test registry initialization."""
        assert registry.data_dir.name == "psychometrics"
        assert isinstance(registry._loaders, dict)
        assert len(registry._loaders) == 0  # Lazy loading

    def test_list_instruments_all(self, registry: PsychometricRegistry):
        """Test listing all instruments."""
        instruments = registry.list_instruments(available_only=False)
        assert "onet" in instruments
        assert "hexaco-100" in instruments
        assert len(instruments) == 9  # Total defined instruments

    def test_list_instruments_available_only(self, registry: PsychometricRegistry):
        """Test listing only available instruments."""
        instruments = registry.list_instruments(available_only=True)
        assert "onet" in instruments  # ONETLoader is always available
        # Other instruments depend on implementation status

    def test_list_instruments_by_type(self, registry: PsychometricRegistry):
        """Test filtering instruments by type."""
        personality = registry.list_instruments(data_type="personality", available_only=False)
        assert "hexaco-100" in personality
        assert "ipip-neo-120" in personality
        assert "onet" not in personality

        occupational = registry.list_instruments(data_type="occupational", available_only=False)
        assert "onet" in occupational
        assert "hexaco-100" not in occupational

    def test_get_loader_onet(self, registry: PsychometricRegistry):
        """Test getting ONETLoader."""
        loader = registry.get_loader("onet")
        assert loader is not None
        assert hasattr(loader, 'get_occupation_profile')

    def test_get_loader_caching(self, registry: PsychometricRegistry):
        """Test that loaders are cached."""
        loader1 = registry.get_loader("onet")
        loader2 = registry.get_loader("onet")
        assert loader1 is loader2  # Same instance

    def test_get_loader_invalid(self, registry: PsychometricRegistry):
        """Test error for invalid instrument ID."""
        with pytest.raises(ValueError, match="Unknown instrument"):
            registry.get_loader("invalid-instrument")

    def test_get_loader_unavailable(self, registry: PsychometricRegistry):
        """Test error when loader is not implemented."""
        # This test will fail once loaders are implemented
        # Adjust based on which loaders are available
        try:
            registry.get_loader("hexaco-100")
        except RuntimeError as e:
            assert "not available" in str(e)

    def test_get_instrument_info(self, registry: PsychometricRegistry):
        """Test getting instrument metadata."""
        info = registry.get_instrument_info("hexaco-100")
        assert info["name"] == "HEXACO-PI-R 100"
        assert info["data_type"] == "personality"
        assert info["items"] == 100

    def test_get_instrument_info_invalid(self, registry: PsychometricRegistry):
        """Test error for invalid instrument."""
        with pytest.raises(ValueError):
            registry.get_instrument_info("invalid")

    def test_generate_summary(self, registry: PsychometricRegistry):
        """Test summary generation."""
        summary = registry.generate_summary()

        assert "generated" in summary
        assert "instruments" in summary
        assert "totals" in summary
        assert "by_category" in summary

        assert summary["totals"]["total_instruments"] == 9
        assert "onet" in summary["instruments"]

    def test_export_catalog(self, registry: PsychometricRegistry, tmp_path: Path):
        """Test catalog export."""
        output = tmp_path / "catalog.json"
        registry.export_catalog(output)

        assert output.exists()

        import json
        with open(output) as f:
            data = json.load(f)

        assert "instruments" in data
        assert "totals" in data

    def test_export_markdown_summary(self, registry: PsychometricRegistry, tmp_path: Path):
        """Test markdown summary export."""
        output = tmp_path / "README.md"
        registry.export_markdown_summary(output)

        assert output.exists()

        content = output.read_text()
        assert "LM-VECTOR Psychometrics Data Repository" in content
        assert "Summary Statistics" in content
        assert "onet" in content

    def test_verify_all(self, registry: PsychometricRegistry):
        """Test verification of all loaders."""
        results = registry.verify_all(verbose=False)

        assert "summary" in results
        assert "details" in results

        # At minimum, ONET should pass
        assert results["details"]["onet"]["status"] == "pass"
        assert results["summary"]["passed"] >= 1


class TestRegistryIntegration:
    """Integration tests with actual loaders."""

    @pytest.fixture
    def registry(self) -> PsychometricRegistry:
        """Create registry with real data directory."""
        # Use actual data directory if available
        return PsychometricRegistry()

    @pytest.mark.skipif(
        not Path("data/psychometrics/hexaco").exists(),
        reason="HEXACO data not available"
    )
    def test_get_items_hexaco(self, registry: PsychometricRegistry):
        """Test getting HEXACO items."""
        items = registry.get_items("hexaco-100")
        assert len(items) == 100
        assert "item_number" in items[0]
        assert "text" in items[0]

    def test_get_items_onet(self, registry: PsychometricRegistry):
        """Test that ONET loader integration works."""
        # ONET doesn't have get_items, so should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            registry.get_items("onet")
```

### 7.2 Test Commands

```bash
# Run all registry tests
uv run pytest tests/unit/test_psychometric_registry.py -v

# Run with coverage
uv run pytest tests/unit/test_psychometric_registry.py --cov=pvx.data.psychometric_registry --cov-report=term-missing

# Run integration tests only
uv run pytest tests/unit/test_psychometric_registry.py::TestRegistryIntegration -v
```

---

## 8. Acceptance Criteria

### 8.1 Code Quality

- [ ] `PsychometricRegistry` class implemented in `src/pvx/data/psychometric_registry.py`
- [ ] All public methods have docstrings
- [ ] Type hints on all parameters and return types
- [ ] No `ty` type checker errors
- [ ] Follows existing code patterns from `ONETLoader`

### 8.2 Functionality

- [ ] `list_instruments()` returns correct instrument IDs
- [ ] `get_loader()` instantiates and caches loaders correctly
- [ ] `get_items()` works for all available instruments
- [ ] `generate_summary()` produces valid summary data
- [ ] `verify_all()` correctly tests all loaders
- [ ] Lazy loading works (loaders not instantiated until needed)

### 8.3 Data Artifacts

- [ ] `data/psychometrics/catalog.json` generated correctly
- [ ] `data/psychometrics/README.md` is human-readable
- [ ] JSON catalog passes schema validation

### 8.4 Testing

- [ ] Unit tests cover all public methods
- [ ] Tests pass for unavailable loaders (graceful degradation)
- [ ] Integration tests work with available loaders
- [ ] Test coverage > 80%

### 8.5 Documentation

- [ ] Docstrings complete for all classes/methods
- [ ] README in `data/psychometrics/` documents usage
- [ ] Workflow document (this file) is complete

---

## 9. Verification Commands

### 9.1 Quick Verification

```bash
# Test registry import
python3 -c "from pvx.data.psychometric_registry import PsychometricRegistry; print('✓ Import successful')"

# List instruments
python3 -c "
from pvx.data.psychometric_registry import PsychometricRegistry
registry = PsychometricRegistry()
print('Available instruments:', registry.list_instruments())
"

# Generate summary
python3 -c "
from pvx.data.psychometric_registry import PsychometricRegistry
import json
registry = PsychometricRegistry()
summary = registry.generate_summary()
print(json.dumps(summary['totals'], indent=2))
"
```

### 9.2 Full Verification Script

```bash
#!/bin/bash
echo "=== Phase 7: Registry Verification ==="

echo -e "\n--- Import Check ---"
uv run python -c "from pvx.data.psychometric_registry import PsychometricRegistry; print('✓ Registry imported')"

echo -e "\n--- List Instruments ---"
uv run python -c "
from pvx.data.psychometric_registry import PsychometricRegistry
registry = PsychometricRegistry()
instruments = registry.list_instruments(available_only=False)
print(f'Total instruments defined: {len(instruments)}')
available = registry.list_instruments(available_only=True)
print(f'Available instruments: {len(available)}')
print(f'Available: {available}')
"

echo -e "\n--- Generate Catalog ---"
uv run python -c "
from pvx.data.psychometric_registry import PsychometricRegistry
from pathlib import Path
registry = PsychometricRegistry()
registry.export_catalog(Path('data/psychometrics/catalog.json'))
registry.export_markdown_summary(Path('data/psychometrics/README.md'))
"

echo -e "\n--- Verify All Loaders ---"
uv run python -c "
from pvx.data.psychometric_registry import PsychometricRegistry
registry = PsychometricRegistry()
results = registry.verify_all(verbose=True)
"

echo -e "\n--- Run Unit Tests ---"
uv run pytest tests/unit/test_psychometric_registry.py -v --tb=short

echo -e "\n--- Type Check ---"
uv run ty check src/pvx/data/psychometric_registry.py

echo -e "\n=== Verification Complete ==="
```

### 9.3 Catalog Generation Script

```python
"""
Script: generate_catalog.py
Generate complete project catalog and summary
"""
from pathlib import Path
from pvx.data.psychometric_registry import PsychometricRegistry

def main():
    registry = PsychometricRegistry()
    base_dir = Path("data/psychometrics")

    # Generate JSON catalog
    catalog_path = base_dir / "catalog.json"
    registry.export_catalog(catalog_path)
    print(f"✓ Catalog: {catalog_path}")

    # Generate README
    readme_path = base_dir / "README.md"
    registry.export_markdown_summary(readme_path)
    print(f"✓ README: {readme_path}")

    # Run verification
    print("\n--- Running Verification ---")
    results = registry.verify_all(verbose=True)

    # Save verification report
    import json
    report_path = base_dir / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Verification report: {report_path}")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
uv run python scripts/generate_catalog.py
```

---

## 10. Integration with Persona Generation

### Example: Using Registry in Persona Vector Pipeline

```python
"""
Example: Using PsychometricRegistry for persona generation
"""
from pvx.data.psychometric_registry import PsychometricRegistry

# Initialize registry
registry = PsychometricRegistry()

# Generate a persona profile using multiple instruments
def generate_persona_profile(occupation_code: str) -> dict:
    """Generate comprehensive persona profile."""

    profile = {}

    # Get O*NET data
    onet = registry.get_loader('onet')
    profile['occupation'] = onet.get_occupation_profile(occupation_code)

    # Get personality instruments
    try:
        hexaco_items = registry.get_items('hexaco-100')
        profile['hexaco_available'] = True
        profile['hexaco_item_count'] = len(hexaco_items)
    except:
        profile['hexaco_available'] = False

    try:
        ipip_items = registry.get_items('ipip-neo-120')
        profile['ipip_available'] = True
        profile['ipip_item_count'] = len(ipip_items)
    except:
        profile['ipip_available'] = False

    # Get values instruments
    try:
        schwartz_items = registry.get_items('schwartz-pvq21')
        profile['schwartz_available'] = True
        profile['schwartz_item_count'] = len(schwartz_items)
    except:
        profile['schwartz_available'] = False

    return profile

# Test
profile = generate_persona_profile('11-1011.00')
print(profile)
```

---

## Related Documents

- [PSYCHOMETRICS_DATA.md § Phase 7](../reference/PSYCHOMETRICS_DATA.md#phase-7-final-catalog--verification) - Reference implementation
- [STATUS_psychometrics.md](./STATUS_psychometrics.md) - Progress tracking
- [WORKFLOW_overview_phases_2_7.md](./WORKFLOW_overview_phases_2_7.md) - Phase overview
- [onet_loader.py](../../src/pvx/data/onet_loader.py) - Pattern reference
- [WORKFLOW_phase2_hexaco.md](./WORKFLOW_phase2_hexaco.md) - HEXACO workflow
- [WORKFLOW_phase3_ipip_neo.md](./WORKFLOW_phase3_ipip_neo.md) - IPIP-NEO workflow
- [WORKFLOW_phase4_schwartz.md](./WORKFLOW_phase4_schwartz.md) - Schwartz workflow
- [WORKFLOW_phase5_dark_personality.md](./WORKFLOW_phase5_dark_personality.md) - Dark Personality workflow
- [WORKFLOW_phase6_ml_datasets.md](./WORKFLOW_phase6_ml_datasets.md) - ML Datasets workflow

---

## Appendix A: Complete Registry API

### Quick Reference

```python
from pvx.data.psychometric_registry import PsychometricRegistry

registry = PsychometricRegistry()

# Discovery
instruments = registry.list_instruments()
personality = registry.list_instruments(data_type="personality")
info = registry.get_instrument_info("hexaco-100")

# Access
loader = registry.get_loader("hexaco-100")
items = registry.get_items("hexaco-100")

# Summaries
summary = registry.generate_summary()
registry.export_catalog("data/psychometrics/catalog.json")
registry.export_markdown_summary("data/psychometrics/README.md")

# Verification
results = registry.verify_all(verbose=True)
```

---

## Appendix B: Troubleshooting

### Common Issues

**Issue**: `RuntimeError: Loader for 'hexaco-100' is not available`

**Solution**: The loader hasn't been implemented yet. Complete Phase 2 workflow first.

---

**Issue**: `FileNotFoundError: HEXACO-100 items not found`

**Solution**: Data files haven't been extracted. Follow Phase 2 workflow to download and extract data.

---

**Issue**: `ImportError: No module named 'pvx.data.hexaco_loader'`

**Solution**: The loader module doesn't exist. Implement it following the phase workflow.

---

**Issue**: Catalog shows `item_count: 0` for all instruments

**Solution**: Data hasn't been processed. Run data extraction workflows for phases 2-6.

---

*End of Phase 7 Workflow Document*
