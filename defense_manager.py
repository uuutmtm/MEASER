import torch
import os
import importlib
from typing import Dict, Any

class DefenseManager:
    def __init__(self):
        pass

    def run_defense(self, defense_name: str, model, tokenizer, **kwargs):
        """
        Run a defense by dynamically loading the corresponding module.
        Expected structure: LLM_sanitizer/defenses/{defense_name}.py
        Expected entry function: run_{defense_name}_defense(model, tokenizer, **kwargs)
        Returns:
            The defended model.
        """
        defense_name = defense_name.lower().strip()
        
        try:
             # Dynamic import: defenses.peft
            # Assuming this file is in the same package 'LLM_sanitizer'
            try:
                module = importlib.import_module(f".defenses.{defense_name}", package="LLM_sanitizer")
            except ImportError:
                 # Try absolute if relative fails
                module = importlib.import_module(f"defenses.{defense_name}")
            
            # Find entry function: run_{defense_name}_defense
            func_name = f"run_{defense_name}_defense"
            if hasattr(module, func_name):
                run_func = getattr(module, func_name)
                return run_func(model, tokenizer, **kwargs)
            else:
                print(f"Error: Module 'defenses.{defense_name}' does not have function '{func_name}'.")
                return model

        except ImportError as e:
            print(f"Defense '{defense_name}' not found or failed to load: {e}")
            return model
        except Exception as e:
            print(f"An error occurred during defense '{defense_name}': {e}")
            import traceback
            traceback.print_exc()
            return model

