import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
import logging

# Add current dir to path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our managers and validators
try:
    from attack_manager import AttackManager
    from defense_manager import DefenseManager
    from validator.mmlu_val import test_mmlu
except ImportError:
    sys.path.append(os.path.join(current_dir, ".."))
    from attack_manager import AttackManager
    from defense_manager import DefenseManager
    from validator.mmlu_val import test_mmlu

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}\n{title}\n{'='*60}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}[INFO] {msg}{Colors.ENDC}")

def print_result(msg):
    print(f"{Colors.OKGREEN}[RESULT] {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}[WARNING] {msg}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="LLM Sanitizer Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to the target LLM")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--attacks", nargs='+', default=[], help="List of attacks to run (e.g. freezer)")
    parser.add_argument("--defenses", nargs='+', default=[], help="List of defenses to run (e.g. peft)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for defense training")
    parser.add_argument("--payload", type=str, default=None, help="Path to payload file (for attacks that require it)")
    
    # Allow extra args for specific attacks/defenses
    args, unknown = parser.parse_known_args()
    
    # Parse unknown args into dictionary
    extra_args = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip('-')
        if i + 1 < len(unknown):
            val = unknown[i+1]
            extra_args[key] = val
        else:
             extra_args[key] = True # Flag
             
    print_info(f"Extra Args Parsed: {extra_args}")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_log = output_dir / "pipeline_log.txt"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(results_log),
            # logging.StreamHandler(sys.stdout) # Removed duplicate stream handler to avoid double printing with our custom prints
        ]
    )
    logger = logging.getLogger(__name__)

    print_section(f"Starting Pipeline\nModel: {args.model}\nAttacks: {args.attacks}\nDefenses: {args.defenses}")

    # Load Model (Float16)
    print_info("Loading Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        # Handle pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16, 
            device_map=args.device,
            trust_remote_code=True
        )
    except Exception as e:
        print_info(f"{Colors.FAIL}Error loading model: {e}{Colors.ENDC}")
        logger.error(f"Error loading model: {e}")
        return

    # 1. Test Original Model
    print_section("1. Testing Original Model (MMLU)")
    base_res = test_mmlu(model, tokenizer, data_dir="/home/tanming/2026/datasets/mmlu/data/val", num_samples=5) 
    print_result(f"Original Results: {base_res}")
    logger.info(f"Original Results: {base_res}")

    attacker = AttackManager(payloads_dir=str(Path(current_dir) / "payloads"))
    attack_meta = {}

    # 2. Attacks
    if args.attacks:
        print_section("2. Running Attacks")
        for attack_name in args.attacks:
            print_info(f"Running Attack: {attack_name}")
            # Run Attack
            # Note: run_attack modifies model in-place
            try:
                kwargs = {"redundancy": 5, "interval": 100}
                current_payload_path = args.payload if args.payload else str(attacker.default_payload_path)
                kwargs["payload_path"] = current_payload_path
                
                # Pass tokenizer for attacks like AWQ-aware watermarking
                kwargs["tokenizer"] = tokenizer
                
                # Merge extra args from command line
                kwargs.update(extra_args)
                    
                meta = attacker.run_attack(attack_name, model, **kwargs)
                if meta:
                    attack_meta.update(meta) # Keep tracking of last attack for now or merge
                    print_result(f"Attack '{attack_name}' completed.")
                    logger.info(f"Attack '{attack_name}' Meta: {meta}")
                    
                    # Verify immediate
                    print_info(f"Verifying {attack_name}...")
                    verify_res = attacker.verify_attack(model, meta, str(output_dir / f"{attack_name}_extraction"), original_payload_path=current_payload_path)
                    print_result(f"Immediate Verification: {verify_res}")
                    logger.info(f"Attack '{attack_name}' Verification: {verify_res}")
                else:
                    print_warning(f"Attack '{attack_name}' returned no metadata (failed?)")
            except Exception as e:
                 print_info(f"{Colors.FAIL}Error running attack {attack_name}: {e}{Colors.ENDC}")

        # 3. Test Attacked Model
        print_section("3. Testing Attacked Model (MMLU)")
        attacked_res = test_mmlu(model, tokenizer, data_dir="/home/tanming/2026/datasets/mmlu/data/val", num_samples=5)
        print_result(f"Attacked Model Perf: {attacked_res}")
        logger.info(f"Attacked Model Perf: {attacked_res}")
    else:
        print_info("No attacks specified, skipping attack phase.")

    # 4. Defenses
    if args.defenses:
        print_section("4. Running Defenses")
        defender = DefenseManager()
        
        # Prepare target modules
        # User requested to target ALL linear layers common in Llama
        default_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if "quant_targets" in extra_args:
            target_modules = extra_args["quant_targets"].split(",")
        else:
            target_modules = default_targets
        
        # Also ensure the specific attacked module is included
        target_param_name = attack_meta.get("target_name", "") # e.g. model.layers.0.self_attn.q_proj.weight
        
        if target_param_name:
            # crude parsing: model.embed_tokens.weight -> embed_tokens
            # model.layers.0.self_attn.q_proj.weight -> q_proj
            parts = target_param_name.split(".")
            if len(parts) >= 2:
                # -1 is weight/bias, -2 is module name
                module_name = parts[-2]
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        for defense_name in args.defenses:
            print_info(f"Running Defense: {defense_name}")
            print_info(f"Target Modules (inferred): {target_modules}")
            
            try:
                model = defender.run_defense(defense_name, model, tokenizer, 
                                            output_dir=str(output_dir / f"{defense_name}_checkpoints"), 
                                            epochs=args.epochs, 
                                            dataset_path="/home/tanming/2026/datasets/mmlu", 
                                            target_modules=target_modules,
                                            **extra_args)
                print_result(f"Defense '{defense_name}' completed.")
            except Exception as e:
                print_info(f"{Colors.FAIL}Error running defense {defense_name}: {e}{Colors.ENDC}")

        # 5. Test Defended Model
        print_section("5. Testing Defended Model")
        
        # 5a. Defense Effectiveness (Try to extract payload again if there was an attack)
        if attack_meta:
            print_info("Verifying if attack persists...")
            current_payload_path = args.payload if args.payload else str(attacker.default_payload_path)
            post_defense_verify = attacker.verify_attack(model, attack_meta, str(output_dir / "defense_extraction"), original_payload_path=current_payload_path)
            print_result(f"Post-Defense Extraction: {post_defense_verify}")
            logger.info(f"Post-Defense Extraction: {post_defense_verify}")
        
        # 5b. Model Performance
        print_info("Evaluating model performance (MMLU)...")
        defended_res = test_mmlu(model, tokenizer, data_dir="/home/tanming/2026/datasets/mmlu/data/val", num_samples=5)
        print_result(f"Defended Model Perf: {defended_res}")
        logger.info(f"Defended Model Perf: {defended_res}")

    else:
        print_info("No defenses specified, skipping defense phase.")

    print_section("Pipeline Completed")

if __name__ == "__main__":
    main()

