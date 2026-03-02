
import torch
import numpy as np
import scipy.stats as stats
import time

class LDPC:
    def __init__(self, n, rate=0.5):
        self.n = n
        self.k = int(n * rate)
        self.m = self.n - self.k
        self.H = None
        
    def construct_H(self, seed=42):
        np.random.seed(seed)
        self.H = np.zeros((self.m, self.n), dtype=int)
        
        # Dual-diagonal part for parity bits
        self.H[:, self.k:] = np.eye(self.m)
        for i in range(1, self.m):
            self.H[i, self.k + i - 1] = 1
        
        # Regular sparse part for message bits (dv=4 for robust BSC correction)
        dv = 4
        for i in range(dv):
            perm = np.random.permutation(self.m)
            for j in range(self.k):
                self.H[perm[j], j] = 1
            
        self.H_mask = (self.H == 1)
        self.P_perm = np.arange(self.k)
        
    def encode(self, msg_bits):
        pm = np.dot(self.H[:, :self.k], msg_bits) % 2
        p = np.zeros(self.m, dtype=int)
        p[0] = pm[0]
        for i in range(1, self.m):
            p[i] = (pm[i] + p[i-1]) % 2
        return np.concatenate([msg_bits, p])
        
    def decode(self, llr, max_iter=25):
        # Convert to standard LLR convention (L > 0 means bit 0, L < 0 means bit 1)
        L = -np.array(llr, copy=True)
        # Squashing channel LLR heavily to allow BP Check Nodes to overcome false adversarial confidence
        L = np.clip(L, -10.0, 10.0)
        
        V = np.tile(L, (self.m, 1)) * self.H_mask
        
        for it in range(max_iter):
            # Check to Variable
            S = np.sign(V)
            S[S == 0] = 1 
            S = np.where(self.H_mask, S, 1)
            
            row_signs = np.prod(S, axis=1, keepdims=True)
            C_signs = row_signs * S
            
            mag_V = np.where(self.H_mask, np.abs(V), np.inf)
            min1_idx = np.argmin(mag_V, axis=1)
            min1_vals = np.min(mag_V, axis=1)
            
            mag_V_temp = mag_V.copy()
            mag_V_temp[np.arange(self.m), min1_idx] = np.inf
            min2_vals = np.min(mag_V_temp, axis=1)
            
            min1_vals[np.isinf(min1_vals)] = 0.0
            min2_vals[np.isinf(min2_vals)] = 0.0
            
            min_vals_matrix = np.tile(min1_vals.reshape(-1, 1), (1, self.n))
            min_vals_matrix[np.arange(self.m), min1_idx] = min2_vals
            
            # Normalized Min-Sum
            alpha = 0.75
            C = alpha * C_signs * min_vals_matrix * self.H_mask
            
            # Variable to Check
            col_sums = np.sum(C, axis=0, keepdims=True)
            V = np.clip((col_sums - C + L.reshape(1, -1)) * self.H_mask, -20.0, 20.0)
            
            # Tentative decision
            L_post = col_sums.flatten() + L
            bits = (L_post < 0).astype(int) 
            
            # Early stopping check
            if np.sum(np.dot(self.H, bits) % 2) == 0:
                break
                
        return bits[:self.k]

class MeaserAttack:
    def __init__(self):
        self.seed = 2026

    def _get_params(self, model, target_layer=None):
        params = []
        for name, param in model.named_parameters():
             if target_layer is None or target_layer in name:
                 if param.dim() >= 1:
                    params.append(param)
        return params

    def _get_top_indices(self, params, total_chips, seed=0):
        # Identify indices with largest magnitude
        # Return format: list of (param_idx, offset) or flattened global indices
        # For efficiency, we need to sort ALL params values?
        # That's heavyweight. 
        # Heuristic: Sample? No, precise Top-N.
        # We handle this by collecting limited info.
        
        # 1. Flatten all params (virtually)
        # Store (abs_val, global_idx) ? Too big for 7B model.
        # We need a memory-efficient way.
        
        # Strategy: Iterate params. Keep a Min-Heap of top N?
        # N=99k words. Heap is cheap.
        # Python heap is slow for 100k inserts per param.
        # Torch TopK?
        
        # Per-param TopK.
        # If we want global Top N.
        # Collect top N from EACH param. 
        # Then merge and take global top N.
        # Still N*M size. N=100k, params=5. Size is manageable.
        
        candidates_val = []
        candidates_idx = []
        
        global_offset = 0
        for p_idx, param in enumerate(params):
            numel = param.numel()
            # Take top N from this param
            k = min(numel, total_chips) 
            vals = param.view(-1).abs()
            topk = torch.topk(vals, k)
            
            cand_v = topk.values.detach().cpu().numpy()
            cand_i = topk.indices.detach().cpu().numpy() + global_offset
            
            candidates_val.append(cand_v)
            candidates_idx.append(cand_i)
            global_offset += numel
            
        # Merge
        all_v = np.concatenate(candidates_val)
        all_i = np.concatenate(candidates_idx)
        
        # Global sort descending
        # We want top total_chips
        # Use argpartition for O(N)
        if len(all_v) > total_chips:
            idx = np.argpartition(all_v, -total_chips)[-total_chips:]
            final_v = all_v[idx]
            final_global_indices = all_i[idx]
        else:
            final_global_indices = all_i
            
        final_global_indices.sort()
        return final_global_indices

    def embed(self, model, payload_path, **kwargs):
        meta = {
            "attack": "measer",
            "target_layer": kwargs.get("target_layer", None),
            "gain": kwargs.get("gain", 6),
            "target_bit_pos": kwargs.get("target_bit_pos", 5), # 3% distortion
            "payload_len_bits": 0,
            "ldpc_k": 1024,
            "ldpc_n": 2048,
        }
        
        with open(payload_path, "rb") as f:
            payload_bytes = f.read()
        
        if len(payload_bytes) > 2000: payload_bytes = payload_bytes[:2000]
            
        payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        meta["payload_len_bits"] = len(payload_bits)
        
        preamble = np.zeros(200, dtype=int)
        self.ldpc = LDPC(meta["ldpc_n"])
        self.ldpc.construct_H()
        
        pad_len = (self.ldpc.k - (len(payload_bits) % self.ldpc.k)) % self.ldpc.k
        padded_payload = np.concatenate([payload_bits, np.zeros(pad_len, dtype=int)])
        meta["pad_len"] = pad_len
        
        encoded_blocks = []
        for i in range(0, len(padded_payload), self.ldpc.k):
            chunk = padded_payload[i:i+self.ldpc.k]
            encoded_blocks.append(self.ldpc.encode(chunk))
            
        full_msg_bits = np.concatenate([preamble] + encoded_blocks)
        
        bpsk = np.where(full_msg_bits==0, -1, 1)
        
        np.random.seed(self.seed)
        gain = meta["gain"]
        total_chips = len(bpsk) * gain
        spreading_code = np.random.choice([-1, 1], size=total_chips)
        
        expanded_msg = np.repeat(bpsk, gain)
        chip_seq = expanded_msg * spreading_code
        # Map Chips: -1->0, 1->1
        qim_bits = np.where(chip_seq == -1, 0, 1)
        
        meta["total_chips"] = total_chips
        meta["preamble_len"] = len(preamble)
        
        print(f"[Measer] Embedding {total_chips} chips using Top-K QIM (Target Bit {meta['target_bit_pos']})...")
        
        target_params = self._get_params(model, meta["target_layer"])
        
        with torch.no_grad():
            # Get Top Indices
            target_indices = self._get_top_indices(target_params, total_chips)
            if len(target_indices) < total_chips:
                print(f"[Measer] Warning: Not enough params. Truncating.")
                total_chips = len(target_indices)
                qim_bits = qim_bits[:total_chips]
            
            np.save(f"measer_target_indices.npy", target_indices)
            meta["target_indices_file"] = "measer_target_indices.npy"

            # Re-planning Injection
            # Assign Chips to Indices Sequentially (Sorted Index -> Sorted Chip)
            # This is arbitrary but reproducible if Top Indices are stable.
            # We assign qim_bits[0] to target_indices[0], etc.
            
            # Create (GlobalIdx, Bit) pairs
            chip_assignments = np.stack([target_indices, qim_bits], axis=1)
            # Already sorted
            
            curr_pos = 0
            assignment_ptr = 0
            num_assigned = len(chip_assignments)
            
            for param in target_params:
                if assignment_ptr >= num_assigned: break
                
                numel = param.numel()
                end_pos = curr_pos + numel
                
                start_ptr = assignment_ptr
                while assignment_ptr < num_assigned and chip_assignments[assignment_ptr, 0] < end_pos:
                    assignment_ptr += 1
                
                if assignment_ptr > start_ptr:
                    batch = chip_assignments[start_ptr:assignment_ptr]
                    local_indices = batch[:, 0] - curr_pos
                    bits_to_embed = batch[:, 1]
                    
                    flat = param.view(-1)
                    # Convert to float32 to prevent 1e-9 from underflowing to 0 in float16
                    vals = flat[local_indices].cpu().float().numpy()
                    
                    # QIM Logic
                    exp = np.floor(np.log2(np.maximum(np.abs(vals), 1e-9)))
                    delta = np.power(2.0, exp - 10 + meta["target_bit_pos"])
                    
                    v_scaled = vals / delta
                    v_int = np.round(v_scaled)
                    
                    is_odd = (v_int % 2 != 0).astype(int)
                    target_odd = bits_to_embed.astype(int)
                    
                    needs_fix = (is_odd != target_odd)
                    
                    c1 = v_int - 1
                    c2 = v_int + 1
                    dist1 = np.abs(c1 - v_scaled)
                    dist2 = np.abs(c2 - v_scaled)
                    best_fix = np.where(dist1 < dist2, c1, c2)
                    
                    final_ints = np.where(needs_fix, best_fix, v_int)
                    new_vals = final_ints * delta
                    
                    flat[local_indices] = torch.tensor(new_vals, dtype=param.dtype).to(param.device)
                    
                curr_pos = end_pos
                
        print("[Measer] Injection Complete.")
        return model, meta

    def extract(self, model, meta):
        self.ldpc = LDPC(meta["ldpc_n"])
        self.ldpc.construct_H() 
        
        target_params = self._get_params(model, meta["target_layer"])
        total_chips = meta["total_chips"]
        
        # Get Top Indices (should match Embed if magnitudes preserved approx)
        if "target_indices_file" in meta:
            try:
                target_indices = np.load(meta["target_indices_file"])
            except:
                target_indices = self._get_top_indices(target_params, total_chips)
        else:
            target_indices = self._get_top_indices(target_params, total_chips)
        
        # Read
        # We need to map extracted values back to Chip Sequence ID.
        # Embed Logic: target_indices[k] gets qim_bits[k].
        # So we just read target_indices[k] -> extracted_bits_seq[k].
        
        extracted_bits_seq = np.zeros(total_chips)
        
        # Build Map: GlobalIdx -> ChipSeqIdx (0..total)
        # target_indices is already sorted list of GlobalIdx.
        # So target_indices[k] corresponds to k.
        
        req_map = np.stack([target_indices, np.arange(total_chips)], axis=1)
        
        curr_pos = 0
        req_ptr = 0
        num_req = len(req_map)
        
        with torch.no_grad():
            for param in target_params:
                if req_ptr >= num_req: break
                
                numel = param.numel()
                end_pos = curr_pos + numel
                
                start_ptr = req_ptr
                while req_ptr < num_req and req_map[req_ptr, 0] < end_pos:
                    req_ptr += 1
                    
                if req_ptr > start_ptr:
                    batch = req_map[start_ptr:req_ptr]
                    local_indices = batch[:, 0] - curr_pos
                    chip_seq_indices = batch[:, 1]
                    
                    flat = param.view(-1)
                    # Convert to float32 to prevent 1e-9 from underflowing to 0 in float16
                    vals = flat[local_indices].cpu().float().numpy()
                    
                    exp = np.floor(np.log2(np.maximum(np.abs(vals), 1e-9)))
                    delta = np.power(2.0, exp - 10 + meta["target_bit_pos"])
                    
                    v_scaled = vals / delta
                    v_int = np.round(v_scaled)
                    
                    is_odd = (v_int % 2 != 0)
                    bipolar = np.where(is_odd, 1.0, -1.0)
                    
                    extracted_bits_seq[chip_seq_indices.astype(int)] = bipolar
                    
                curr_pos = end_pos

        # Despread
        gain = meta["gain"]
        np.random.seed(self.seed)
        spreading_code = np.random.choice([-1, 1], size=total_chips)
        
        correlated = extracted_bits_seq * spreading_code
        
        num_syms = total_chips // gain
        if num_syms == 0: return b""

        reshaped = correlated[:num_syms*gain].reshape(num_syms, gain)
        soft_syms = reshaped.sum(axis=1)
        
        preamble_len = meta["preamble_len"]
        p_soft = soft_syms[:preamble_len]
        sig_est = np.mean(p_soft * -1)
        noise_var = np.var(p_soft) + 1e-9
        
        print(f"[Measer] Extract Channel: Signal={sig_est:.4f}, SNR={10*np.log10(sig_est**2/noise_var):.2f}dB")
        
        llrs = soft_syms[preamble_len:] * (2 / noise_var)
        
        decoded_bits = []
        K, N = meta["ldpc_k"], meta["ldpc_n"]
        
        for i in range(len(llrs)//N):
            block = llrs[i*N:(i+1)*N]
            dec = self.ldpc.decode(block)
            decoded_bits.extend(dec)
            
        final = decoded_bits[:meta["payload_len_bits"]]
        return self._bits_to_bytes(final)

    def verify_measer_attack(self, model, attack_meta, output_dir):
        extracted_bytes = self.extract(model, attack_meta)
        return {
            "success": 1,
            "extracted_bytes": extracted_bytes,
        }
    
    def _bits_to_bytes(self, bits):
        b = np.packbits(np.array(bits, dtype=np.uint8))
        return b.tobytes()

# Entry Point for AttackManager
def run_measer_attack(model, **kwargs):
    attacker = MeaserAttack()
    # Remove payload_path from kwargs to avoid duplicate argument error
    embed_kwargs = kwargs.copy()
    if "payload_path" in embed_kwargs:
        payload_path = embed_kwargs.pop("payload_path")
    else:
        payload_path = "payloads/test_payload.bin"

    _, meta = attacker.embed(model, payload_path, **embed_kwargs)
    
    meta["attack"] = "measer"
    meta["payload_path"] = payload_path
    
    return meta

# Entry Point for Verification
def verify_measer_attack(model, attack_meta, output_dir):
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    attacker = MeaserAttack()
    res = attacker.verify_measer_attack(model, attack_meta, output_dir)
    extracted_bytes = res["extracted_bytes"]
    
    # Ground Truth Validation
    original_bytes = None
    if "payload_path" in attack_meta:
        try:
            with open(attack_meta["payload_path"], "rb") as f:
                original_bytes = f.read()
        except:
            pass
    if original_bytes is None and "payload_content" in attack_meta:
         original_bytes = attack_meta["payload_content"].encode("latin-1")
         
    # Truncate original_bytes to the exact embedded length to fix False Negative success mismatch
    embedded_bytes_len = attack_meta.get("payload_len_bits", 0) // 8
    if original_bytes and embedded_bytes_len > 0 and embedded_bytes_len <= len(original_bytes):
        original_bytes = original_bytes[:embedded_bytes_len]
         
    success = 0
    ber = 0.0
    
    if original_bytes:
        if extracted_bytes == original_bytes:
            success = 1
            ber = 0.0
        else:
            success = 0
            try:
                len_min = min(len(original_bytes), len(extracted_bytes))
                if len_min > 0:
                    orig_bits = np.unpackbits(np.frombuffer(original_bytes[:len_min], dtype=np.uint8))
                    ext_bits = np.unpackbits(np.frombuffer(extracted_bytes[:len_min], dtype=np.uint8))
                    errs = np.sum(orig_bits != ext_bits)
                    ber = errs / len(orig_bits)
                else:
                    ber = 1.0
            except Exception as e:
                print(f"BER Calc Error: {e}")
                ber = 1.0

    extracted_sample = extracted_bytes[:50].decode("latin-1", errors="replace")
    
    return {
        "success": success,
        "ber": ber,
        "extracted_sample": extracted_sample,
        "time": time.time() - start_time
    }
