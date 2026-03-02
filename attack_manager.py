import torch
import os
import time
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional
from validator import defense_effective_val

class AttackManager:
    def __init__(self, payloads_dir: str = "/home/tanming/2026/LLM_sanitizer/payloads"):
        self.payloads_dir = Path(payloads_dir)
        self.payloads_dir.mkdir(parents=True, exist_ok=True)
        # Ensure a default payload exists
        self.default_payload_path = self.payloads_dir / "test_payload.bin"
        if not self.default_payload_path.exists():
            with open(self.default_payload_path, "wb") as f:
                f.write(os.urandom(1024)) # 1KB payload

    def run_attack(self, attack_name: str, model: torch.nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Run a specific attack by dynamically loading the corresponding module.
        Expected structure: LLM_sanitizer/attacks/{attack_name}.py
        Expected entry function: run_{attack_name}_attack(model, **kwargs)
        """
        attack_name = attack_name.lower().strip()
        
        # Prepare standard arguments
        if "payload_path" not in kwargs:
             kwargs["payload_path"] = self.default_payload_path

        try:
            # Dynamic import: attacks.freezer
            # Assuming this file is in the same package 'LLM_sanitizer'
            # We try relative import if possible, or absolute
            try:
                module = importlib.import_module(f".attacks.{attack_name}", package="LLM_sanitizer")
            except ImportError:
                 # Try absolute if relative fails (e.g. running script directly)
                module = importlib.import_module(f"attacks.{attack_name}")
            
            # Find entry function: run_{attack_name}_attack
            func_name = f"run_{attack_name}_attack"
            if hasattr(module, func_name):
                run_func = getattr(module, func_name)
                return run_func(model, **kwargs)
            else:
                print(f"Error: Module 'attacks.{attack_name}' does not have function '{func_name}'.")
                return {}

        except ImportError as e:
            print(f"Attack '{attack_name}' not found or failed to load: {e}")
            return {}
        except Exception as e:
            print(f"An error occurred during attack '{attack_name}': {e}")
            import traceback
            traceback.print_exc()
            return {}

    def verify_attack(self, model: torch.nn.Module, attack_meta: Dict[str, Any], output_dir: str, original_payload_path: str = None) -> Dict[str, Any]:
        """Verify attack success by extracting payload."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        attack_name = attack_meta.get("attack", "")
        if not attack_name:
            return {"error": "No attack metadata provided."}
            
        try:
            # Dynamic import
            try:
                module = importlib.import_module(f".attacks.{attack_name}", package="LLM_sanitizer")
            except ImportError:
                module = importlib.import_module(f"attacks.{attack_name}")
                
            # Find entry function: verify_{attack_name}_attack
            func_name = f"verify_{attack_name}_attack"
            if hasattr(module, func_name):
                verify_func = getattr(module, func_name)
                
                # Dynamically pass original_payload_path if the function accepts it
                sig = inspect.signature(verify_func)
                if "original_payload_path" in sig.parameters:
                    result = verify_func(model, attack_meta, str(output_dir), original_payload_path=original_payload_path)
                else:
                    result = verify_func(model, attack_meta, str(output_dir))
                
                # Check for content correctness if payload provided
                if original_payload_path and os.path.exists(original_payload_path):
                    try:
                        with open(original_payload_path, "rb") as f:
                            orig_bytes = f.read()

                        ext_bytes = b""
                        if "extracted_bytes" in result:
                            ext_bytes = result["extracted_bytes"]
                        elif "extracted_path" in result and os.path.exists(result["extracted_path"]):
                            with open(result["extracted_path"], "rb") as f:
                                ext_bytes = f.read()
                        
                        if ext_bytes:
                            min_l = min(len(orig_bytes), len(ext_bytes))
                            if min_l > 0:
                                val_ber = defense_effective_val.calculate_ber(orig_bytes[:min_l], ext_bytes[:min_l])
                                result["ber"] = val_ber
                                # The user specifically requested success=1 if extraction runs, regardless of BER
                                result["success"] = 1
                    except Exception as e:
                        print(f"Generic BER Check failed: {e}")
                
                # Cleanup to prevent console flooding
                if "extracted_bytes" in result:
                    del result["extracted_bytes"]
                    
                return result
            else:
                print(f"Error: Module 'attacks.{attack_name}' does not have function '{func_name}'.")
                return {"error": f"Verification function not found for {attack_name}"}

                
        except Exception as e:
            print(f"Verification failed/not found for {attack_name}: {e}")
            import traceback
            traceback.print_exc()
            return {"success": 0, "error": str(e)}
