import hashlib
import os
from typing import Union, List, Tuple, Dict, Any

def bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def calculate_ber(data1: Union[bytes, List[int]], data2: Union[bytes, List[int]]) -> float:
    """
    Calculate Bit Error Rate (BER) between two sequences.
    Handles both bytes and list of bits.
    """
    # Convert to bits if bytes provided
    if isinstance(data1, bytes):
        bits1 = bytes_to_bits(data1)
    else:
        bits1 = data1
        
    if isinstance(data2, bytes):
        bits2 = bytes_to_bits(data2)
    else:
        bits2 = data2
        
    len1, len2 = len(bits1), len(bits2)
    
    # If both are empty, match
    if len1 == 0 and len2 == 0:
        return 0.0
        
    # Count differences in overlapping part
    min_len = min(len1, len2)
    diffs = sum(b1 != b2 for b1, b2 in zip(bits1[:min_len], bits2[:min_len]))
    
    # Add penalty for length mismatch (missing bits are errors)
    diffs += abs(len1 - len2)
    
    # Use max length as denominator to strictly penalize length mismatch
    total_bits = max(max(len1, len2), 1) 
    
    return diffs / total_bits

def verify_content(original_payload_path: str, extracted_path: str) -> Dict[str, Any]:
    """
    Verify the integrity of extracted content against the original payload.
    Returns a dictionary with verification results suitable for the pipeline.
    """
    result = {
        "verification_mode": "content_match",
        "status": "UNKNOWN",
        "ber": 1.0,
        "success": 0, # Default to fail until proven pass
        "message": ""
    }

    try:
        # Check file existence
        if not original_payload_path or not os.path.exists(original_payload_path):
             result["message"] = f"Original payload not found at {original_payload_path}"
             result["status"] = "ERROR"
             return result
             
        if not extracted_path or not os.path.exists(extracted_path):
             result["message"] = f"Extracted file not found at {extracted_path}"
             result["status"] = "ERROR"
             return result

        # Read files
        with open(original_payload_path, "rb") as f1:
            d1 = f1.read()
            
        with open(extracted_path, "rb") as f2:
            d2 = f2.read()
            
        # Calculate BER
        ber = calculate_ber(d1, d2)
        result["ber"] = ber
        
        # Verify success criteria
        # Strictly require BER = 0.0 for success (perfect extraction)
        if ber == 0.0:
            # Double check with hash for safety
            h1 = hashlib.sha256(d1).hexdigest()
            h2 = hashlib.sha256(d2).hexdigest()
            
            if h1 == h2:
                result["status"] = "PASS"
                result["success"] = 1
                result["message"] = "Content match successful (BER=0.0)"
            else:
                # Should not happen if BER is 0, but technically possible if bits logic is flawed
                result["status"] = "FAIL"
                result["success"] = 0
                result["message"] = "Hash mismatch despite BER=0.0 (Check bit logic)"
        else:
            result["status"] = "FAIL"
            result["success"] = 0
            result["message"] = f"Extraction completed with BER: {ber:.4f}"
                
    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = f"Error during verification: {str(e)}"
        
    return result

# Legacy class wrapper if needed strictly for backward compatibility with previous scripts
class DefenseEvaluator:
    @staticmethod
    def evaluate(extracted_path, original_path):
        res = verify_content(original_path, extracted_path)
        return res["success"], res["ber"]
    
    @staticmethod
    def calculate_ber(bits1, bits2):
        return calculate_ber(bits1, bits2)
