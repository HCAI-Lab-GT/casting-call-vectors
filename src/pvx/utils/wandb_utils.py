import json
import wandb
from pvx.utils.logging_utils import setup_logging

logger = setup_logging(name="wandb_utils")

'''
# Note:
# Do not attempt hook based custom wandb logging.
# It's really brutal getting hook to work as dependent on inspect-wandb hook. No way to order and race condition hell.
# Tried and gave up, not worth it.

from inspect_ai.hooks import hooks, Hooks, RunStart, SampleEnd
'''

def create_table(json_path: str) -> wandb.Table:
    '''
    Rehydrates table from JSON output file and uploads to wandb.
    
    Args:
        json_path (str): Path to JSON file containing samples.
        
    Returns:
        wandb.Table: populated Wandb table.
    '''
    # Load JSON (fallback to JSONL if needed)
    results = {}
    try:
        with open(json_path, "r") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        samples = []
        with open(json_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        results = {"samples": samples}

    # Create wandb Table
    table = wandb.Table(columns=[
        "sample_id",
        "input_json",
        "output_json",
        "scores_json",
    ])

    # Helper to resolve attachments
    def resolve_attachments(obj, attachments):
        """
        Replace strings like 'attachment://<id>' with attachments['<id>'] if present.
        Works recursively on dict/list/str.
        
        Args:
            obj: object to resolve
            attachments: dict of attachments
            
        Returns:
            Resolved object
        """
        if attachments is None:
            attachments = {}

        if isinstance(obj, str) and obj.startswith("attachment://"):
            key = obj.split("attachment://", 1)[1]
            return attachments.get(key, obj)

        if isinstance(obj, dict):
            return {k: resolve_attachments(v, attachments) for k, v in obj.items()}

        if isinstance(obj, list):
            return [resolve_attachments(v, attachments) for v in obj]

        return obj

    # Process each sample
    samples = results.get("samples", results if isinstance(results, list) else [])
    for s in samples:
        # attachments can be per-sample (Inspect logs this) and/or top-level
        attachments = {}
        if isinstance(results, dict) and isinstance(results.get("attachments"), dict):
            attachments.update(results["attachments"])
        if isinstance(s, dict) and isinstance(s.get("attachments"), dict):
            attachments.update(s["attachments"])

        sample_id = s.get("id")

        # Build input payload
        inp = resolve_attachments(s.get("input"), attachments)
        input_payload = {
            "input": inp,
            "target": s.get("target"),
            "metadata": s.get("metadata", {}),
        }

        # Build output payload (resolve any attachment:// in completion/message content)
        out = resolve_attachments(s.get("output"), attachments)

        # Scores payload
        scores = s.get("scores", {})

        # Add to table
        table.add_data(
            sample_id,
            json.dumps(input_payload, ensure_ascii=False, default=str),
            json.dumps(out, ensure_ascii=False, default=str),
            json.dumps(scores, ensure_ascii=False, default=str),
        )

    # Returns the table
    return table