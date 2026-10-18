"""Import a supported ai-toolkit YAML recipe without loading models or starting training."""
from util.import_util import script_imports

script_imports(allow_zluda=False)

import argparse
import json
from pathlib import Path

from modules.util.aitoolkit_config_import import IMPORT_LIMITATIONS, import_aitoolkit_config

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source ai-toolkit YAML recipe")
    parser.add_argument("destination", type=Path, help="New native OneTrainer JSON config; must not exist")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    try:
        source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
        config = import_aitoolkit_config(source)
        data = config.to_dict()
        data.pop("secrets", None)  # Export only recipe settings; never inspect a user's secrets.json.
        serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
        # Exclusive create preserves both the source recipe and any existing destination.
        with args.destination.open("x", encoding="utf-8") as output:
            output.write(serialized)
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Created {args.destination}")
    for limitation in IMPORT_LIMITATIONS:
        print(f"Review: {limitation}")


if __name__ == "__main__":
    main()
