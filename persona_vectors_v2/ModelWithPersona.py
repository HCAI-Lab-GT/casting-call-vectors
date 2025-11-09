from Dataset import PersonaDataset
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS
import torch
import ollama
from typing import List, Tuple, Dict, Optional
import os
from openai import OpenAI
import re

class ModelWithPersona:

    def __init__(self, target_model_id: str = "qwen2.5:7b-instruct", dataset: Optional[PersonaDataset] = None, layer: float = -1):
        self.target_model_id = target_model_id
        self.dataset = dataset
        self.layer_steering = layer