from pvx import setup_logging

from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel

logger = setup_logging(name="identity-experiments")

pvx_identity = RoleLayersPersonaModel.load_or_create(
                        target_model_id="allenai/Olmo-3-7B-Instruct",
                        concept="activist",
                        layer=16,
                        target_pairs=40,
                        safetensors_dir="./persona_data/model_inits/",
                    )

pvx_behavior = RoleLayersPersonaModel.load_or_create(
                        target_model_id="allenai/Olmo-3-7B-Instruct",
                        concept="activist",
                        layer=16,
                        target_pairs=40,
                        safetensors_dir="./src/pvx/experiments/identity_experiments/role_behavior_inits",
                        dataset_dirpath="./src/pvx/experiments/identity_experiments/role_behavior_datasets"
                    )

