import re

try:
    import lm_eval.tasks.ifeval.utils as ifeval_utils
except ImportError:
    raise ImportError("Could not import lm_eval.tasks.ifeval.utils. Please ensure lm-eval is installed.")

def process_results(doc, results):
    """
    Proxies the generation results to lm_eval's native IFEval processor,
    and ensures all returned metrics are scalar floats to prevent aggregation crashes.
    """
    pred = results[0].strip()
    
    # Strip conversational fillers
    match = re.match(r'(?i)(?:the answer is\s*:?\s*|answer\s*:?\s*|it is\s*:?\s*)?(.*)', pred, flags=re.DOTALL)
    if match:
        pred = match.group(1)
        
    # 1. Get the raw metrics from lm_eval's IFEval processor
    raw_metrics = ifeval_utils.process_results(doc, [pred])
    
    safe_metrics = {}
    
    # 2. Flatten any lists into scalar floats
    for key, value in raw_metrics.items():
        if isinstance(value, list):
            # Instruction-level metrics return lists of booleans (one per instruction)
            if len(value) > 0:
                safe_metrics[key] = float(sum(value)) / len(value)
            else:
                safe_metrics[key] = 0.0
        else:
            # Prompt-level metrics return booleans/ints
            safe_metrics[key] = float(value)
            
    return safe_metrics