#!/usr/bin/env python
"""
Extract a RIASEC persona vector for a single trait on a given model.

Uses the existing RIASECPersonaModel pipeline. The pregenerated positive/negative
responses in configs/riasec.yaml are model-agnostic text; only the activations
change per model.

Usage:
  python scripts/marin/run_riasec_extraction.py --trait realistic
  python scripts/marin/run_riasec_extraction.py --model_id marin-community/marin-8b-instruct --trait artistic
  python scripts/marin/run_riasec_extraction.py --trait social --layer 10
"""

import argparse

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-riasec-extraction")


def get_num_layers(model_id: str) -> int:
    """Get the number of decoder layers for a model without loading weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    return config.num_hidden_layers


def main():
    parser = argparse.ArgumentParser(description="Extract RIASEC persona vector for one trait.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        choices=sorted(RIASECHelpers.RIASEC_TRAITS),
        help="RIASEC trait to extract.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for activation extraction. Default: num_layers // 2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./persona_data/model_inits/",
        help="Directory to save persona vectors.",
    )
    args = parser.parse_args()

    # Auto-compute middle layer if not specified
    if args.layer is None:
        num_layers = get_num_layers(args.model_id)
        layer = num_layers // 2
        logger.info("Auto-computed middle layer: %d (of %d total)", layer, num_layers)
    else:
        layer = args.layer

    logger.info("Extracting RIASEC vector: model=%s, trait=%s, layer=%d", args.model_id, args.trait, layer)

    # load_or_create will either load existing safetensors or extract fresh vectors
    model = RIASECPersonaModel.load_or_create(
        target_model_id=args.model_id,
        trait=args.trait,
        layer=layer,
        safetensors_dir=args.output_dir,
    )

    logger.info("Persona vector shape: %s", tuple(model.response_persona_vector.shape))
    logger.info("Saved to: %s", args.output_dir)

    model.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
