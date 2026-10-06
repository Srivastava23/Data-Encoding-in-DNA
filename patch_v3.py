import os

def patch_run_full_pipeline():
    with open('run_full_pipeline.py', 'r') as f:
        content = f.read()
    
    # 1. Update imports
    content = content.replace('from rl_agent import PPOAgent', 'from rl_agent_v3 import PPOAgentV3 as PPOAgent')
    content = content.replace('from rl_environment import DNAEncodingEnvironment, _compute_dibit_distribution', 'from rl_environment_v3 import DNAEncodingEnvironmentV3')
    
    # 2. Update agent loading
    content = content.replace('DNAEncodingEnvironment.STATE_DIMENSION', '8') # State dim in V3
    content = content.replace('DNAEncodingEnvironment.ACTION_DIMENSION', '24') # Action dim in V3
    
    # 3. Update build_policy_table to return empty or a dummy table, since V3 is block-wise
    dummy_policy_table = '''def build_policy_table(agent: PPOAgent) -> list:
    return [{"dibit_distribution": [0.25, 0.25, 0.25, 0.25], "selected_action": 0, "primary_mapping_index": 0, "secondary_mapping_index": 0}]'''
    
    # Replace the build_policy_table function
    import re
    content = re.sub(r'def build_policy_table.*?return policy_table', dummy_policy_table, content, flags=re.DOTALL)
    
    # 4. Phase 5 & 5.5 in run_full_pipeline.py uses select_action. We should replace the state construction
    # Actually, we can just replace the whole phase 5 loop body because phase 5 only calls evaluate_redundancy_comparison which we will also rewrite to not take 'action'.
    phase5_patch = '''for sequence_length in config.sequence_lengths:
        binary_data = generate_frozen_dataset(sequence_length, 1, 42)[0]
        comparison = evaluate_redundancy_comparison(
            binary_data, None, np.random.default_rng(12345), config, agent_for_test, [1, 2, 3, 4, 5]
        )
        redundancy_comparison[str(sequence_length)] = comparison'''
    content = re.sub(r'for sequence_length in config\.sequence_lengths:.*?redundancy_comparison\[str\(sequence_length\)\] = comparison', phase5_patch, content, flags=re.DOTALL)
    
    with open('run_full_pipeline.py', 'w') as f:
        f.write(content)


def patch_evaluation():
    with open('evaluation.py', 'r') as f:
        content = f.read()
    
    content = content.replace('from rl_agent import PPOAgent', 'from rl_agent_v3 import PPOAgentV3 as PPOAgent')
    
    # Add imports for block logic
    imports = '''from training_pipeline_v3 import _compute_block_state
from encoding import get_block_splits, decode_block_sequence
import torch
'''
    content = content.replace('from config import Config', imports + 'from config import Config')
    
    # Rewrite _evaluate_reinforcement_learning_trial
    rl_trial_new = '''def _evaluate_reinforcement_learning_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
    agent: PPOAgent, sequence_length: int,
) -> Dict[str, float]:
    splits = get_block_splits(len(binary_bits), config)
    N = len(splits)
    block_actions = []
    dna_so_far = ""
    actor_hx, critic_hx = None, None
    agent.network.eval()
    with torch.no_grad():
        for k in range(N):
            state = _compute_block_state(binary_bits, splits, k, N, block_actions, dna_so_far)
            state_t = torch.tensor(state, dtype=torch.float32).to(agent.device)
            logits, val, actor_hx, critic_hx = agent.network(state_t, actor_hx, critic_hx)
            action_int = torch.argmax(logits).item()
            
            b_start, b_end = splits[k]
            block_bits = binary_bits[b_start:b_end]
            mapping_idx = action_int
            if config.use_header_free_encoding and k > 0:
                prev_s, prev_e = splits[k - 1]
                offset = int(np.sum(binary_bits[prev_s:prev_e])) % 24
                mapping_idx = (mapping_idx + offset) % 24
            block_dna = encode_binary_to_dna(block_bits, ALL_MAPPINGS[mapping_idx])
            dna_so_far += block_dna
            block_actions.append(action_int)

    mutated_sequence, _ = simulate_errors(dna_so_far, config.substitution_rate,
                                          config.insertion_rate, config.deletion_rate, random_generator)
    decoded_bits = decode_block_sequence(mutated_sequence, block_actions, config)
    return _compute_metrics(binary_bits, dna_so_far, mutated_sequence, decoded_bits)
'''
    import re
    content = re.sub(r'def _evaluate_reinforcement_learning_trial.*?return _compute_metrics.*?decoded_bits\)', rl_trial_new, content, flags=re.DOTALL)
    
    # Rewrite _evaluate_redundancy_trial
    red_trial_new = '''def _evaluate_redundancy_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
    agent: PPOAgent, sequence_length: int,
) -> Dict[str, float]:
    result = encode_with_redundancy(binary_bits, None, random_generator, config, agent=agent)
    return {
        "bit_accuracy": result["redundancy_accuracy"],
        "error_rate": result["consensus_error_rate"],
        "gc_content": gc_content(result["original_dna"]),
        "stability": stability_score(result["original_dna"]),
        "homopolymer": normalized_max_homopolymer(result["original_dna"]),
        "single_copy_accuracy": result["single_copy_accuracy"],
        "redundancy_accuracy": result["redundancy_accuracy"],
        "number_of_copies": result["number_of_copies"],
    }
'''
    content = re.sub(r'def _evaluate_redundancy_trial.*?return \{.*?\}', red_trial_new, content, flags=re.DOTALL)
    
    with open('evaluation.py', 'w') as f:
        f.write(content)


def patch_redundancy():
    with open('redundancy.py', 'r') as f:
        content = f.read()
    
    imports = '''from training_pipeline_v3 import _compute_block_state
from encoding import get_block_splits, decode_block_sequence, encode_binary_to_dna, ALL_MAPPINGS
import torch
'''
    content = content.replace('from config import Config', imports + 'from config import Config')
    
    # Rewrite encode_with_redundancy
    red_encode = '''def encode_with_redundancy(
    binary_bits: np.ndarray,
    action: int,
    random_generator: np.random.Generator,
    config: Config,
    agent = None,
) -> Dict:
    splits = get_block_splits(len(binary_bits), config)
    N = len(splits)
    block_actions = []
    dna_so_far = ""
    if agent:
        actor_hx, critic_hx = None, None
        agent.network.eval()
        with torch.no_grad():
            for k in range(N):
                state = _compute_block_state(binary_bits, splits, k, N, block_actions, dna_so_far)
                state_t = torch.tensor(state, dtype=torch.float32).to(agent.device)
                logits, _, actor_hx, critic_hx = agent.network(state_t, actor_hx, critic_hx)
                action_int = torch.argmax(logits).item()
                
                b_start, b_end = splits[k]
                block_bits = binary_bits[b_start:b_end]
                mapping_idx = action_int
                if config.use_header_free_encoding and k > 0:
                    prev_s, prev_e = splits[k - 1]
                    offset = int(np.sum(binary_bits[prev_s:prev_e])) % 24
                    mapping_idx = (mapping_idx + offset) % 24
                dna_so_far += encode_binary_to_dna(block_bits, ALL_MAPPINGS[mapping_idx])
                block_actions.append(action_int)
    dna_sequence = dna_so_far
    
    copies_data = create_redundant_copies(
        dna_sequence, number_of_copies=config.number_of_copies,
        substitution_rate=config.substitution_rate, insertion_rate=config.insertion_rate,
        deletion_rate=config.deletion_rate, random_generator=random_generator,
    )
    mutated_sequences = [c["mutated_sequence"] for c in copies_data]
    consensus_sequence = consensus_vote(mutated_sequences)
    
    single_copy_decoded = decode_block_sequence(mutated_sequences[0], block_actions, config)
    single_copy_accuracy = compute_bit_accuracy(binary_bits, single_copy_decoded)
    consensus_decoded = decode_block_sequence(consensus_sequence, block_actions, config)
    redundancy_accuracy = compute_bit_accuracy(binary_bits, consensus_decoded)
    consensus_error_rate = compute_error_rate(dna_sequence, consensus_sequence)

    return {
        "original_dna": dna_sequence,
        "primary_index": 0, "secondary_index": 0,
        "copies": copies_data,
        "consensus_sequence": consensus_sequence,
        "consensus_error_rate": consensus_error_rate,
        "single_copy_decoded_bits": single_copy_decoded,
        "consensus_decoded_bits": consensus_decoded,
        "single_copy_accuracy": single_copy_accuracy,
        "redundancy_accuracy": redundancy_accuracy,
        "number_of_copies": config.number_of_copies,
        "average_copy_error_rate": float(np.mean([c["error_rate"] for c in copies_data])),
    }
'''
    import re
    content = re.sub(r'def encode_with_redundancy.*?return \{.*?\}', red_encode, content, flags=re.DOTALL)
    
    # Rewrite evaluate_redundancy_comparison
    eval_comp = '''def evaluate_redundancy_comparison(
    binary_bits: np.ndarray, action: int, random_generator: np.random.Generator,
    config: Config, agent = None, copy_counts: List[int] = None,
) -> List[Dict]:
    if copy_counts is None: copy_counts = [1, 3, 5]
    results = []
    for count in copy_counts:
        test_config = Config(number_of_copies=count, substitution_rate=config.substitution_rate,
                             insertion_rate=config.insertion_rate, deletion_rate=config.deletion_rate)
        result = encode_with_redundancy(binary_bits, action, random_generator, test_config, agent=agent)
        results.append({
            "number_of_copies": count,
            "single_copy_accuracy": result["single_copy_accuracy"],
            "redundancy_accuracy": result["redundancy_accuracy"],
            "consensus_error_rate": result["consensus_error_rate"],
            "average_copy_error_rate": result["average_copy_error_rate"],
        })
    return results
'''
    content = re.sub(r'def evaluate_redundancy_comparison.*?return results', eval_comp, content, flags=re.DOTALL)
    
    with open('redundancy.py', 'w') as f:
        f.write(content)

patch_run_full_pipeline()
patch_evaluation()
patch_redundancy()
print("Patched all files for V3!")
