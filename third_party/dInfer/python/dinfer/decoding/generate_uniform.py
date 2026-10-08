import torch
import numpy as np
import logging

from transformers.models.layoutlmv2.modeling_layoutlmv2 import relative_position_bucket

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num, BlockLoc

logger = logging.getLogger(__name__)

def align_exp2(x: torch.Tensor):
    assert x.ndim == 0 and x.item() >= 0
    shift = 0 if x == 0 else int(torch.floor(torch.log2(x.to(torch.float64)))) + 1
    return 1 << shift

class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

def select_undecoded(seq_idx, orig_x, x, block, block_loc, mask_id, writeback=False):
    if x.batch_size == 1:
        return seq_idx, x
    bool_idx = torch.all(block != mask_id, dim=1)

    if writeback:
        # Write the decoded tokens back
        finished_idx = seq_idx[bool_idx]
        orig_x[finished_idx, block_loc.start:block_loc.end] = block[bool_idx]

    # Select the undecoded sequences
    return seq_idx, x

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

            # If there are more than one sequence, we should filter the sequences and only decode
            # on the sequences that still have masked tokens.
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class BlockDiffusionRunner(BlockRunner):
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : BlockDiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf, backend):
        super().__init__(diff_iteration, early_stop, maximum_unroll, expected_tpf)
        self.backend = backend
        self.cache_update_count = 0
        self.hidden_cache_update_count = 0
        self.need_cross_block_update = False

    def prefill(self, model, prefilling_x, kv_cache, pos_ids, attn_mask, prefilling_limit, block_length):
        """ Prefill for KV Cache
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        prefilling_x : torch.Tensor
            The input IDs of the tokens in the prefilling range.
        kv_cache: KVCache
            The KV-cache
        pos_ids: torch.Tensor
            The position IDs of the tokens in the prefilling range.
        attn_mask: torch.Tensor
            The attention mask of the tokens in the prefilling range.
        prefilling_limit: int
            The limit of the first prefilling step.
        block_length: int
            The block length, used for the following prefilling steps if needed.
        """
        if kv_cache is None:
            return
        else:
            if prefilling_limit > prefilling_x.shape[1]:
                output = model(prefilling_x.clone(memory_format=torch.contiguous_format), use_cache=True, attention_mask=attn_mask, position_ids=pos_ids.clone(memory_format=torch.contiguous_format))
                if self.backend == 'vllm':
                    kv_cache.update(output.past_key_values)
                else:
                    kv_cache.range_update(output.past_key_values, 0, prefilling_x.size(1), 0)
            else:
                # limit prefilling length to avoid OOM
                # first prefill partial prompt
                output = model(prefilling_x[:, :prefilling_limit].clone(memory_format=torch.contiguous_format), use_cache=True, attention_mask=attn_mask[:, :prefilling_limit, :prefilling_limit], 
                               position_ids=pos_ids[:, :prefilling_limit].clone(memory_format=torch.contiguous_format))
                if self.backend == 'vllm':
                    kv_cache.update(output.past_key_values)
                else:
                    kv_cache.range_update(output.past_key_values, 0, prefilling_limit, 0)
                # continue prefilling other parts, using block length to be able to replay already captured cuda graph
                for block_start in range(prefilling_limit, prefilling_x.shape[1], block_length):
                    block_end = block_start+block_length
                    kv_cache.extend_cache(block_end)
                    past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
                    output = model(prefilling_x[:, block_start:block_end].clone(memory_format=torch.contiguous_format), use_cache=True, past_key_values=past_key_values,
                                position_ids=pos_ids[:, block_start:block_end].clone(memory_format=torch.contiguous_format))     
                    if self.backend == 'vllm':
                        kv_cache.update(output.past_key_values)
                    else:
                        kv_cache.range_update(output.past_key_values, 0, block_end, block_length)     

            self.diff_iteration.num_forwards +=1
            self.diff_iteration.iter_no +=1
        self.need_cross_block_update = False

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        is_shifted = isinstance(self.diff_iteration, BlockShiftDiffusionIteration)
        # We MUST loop until this specific block has no masks left
        while True:
            current_block = x.data[:, block_loc.start:block_loc.end]
            input_block_mask_number = (current_block == decoder.mask_id).sum().item()
            
            if input_block_mask_number == 0:
                # Return a boolean tensor indicating completion for the batch
                return torch.ones(x.batch_size, dtype=torch.bool, device=x.device)

            unroll_k = int(max(min(input_block_mask_number // self.expected_tpf, self.maximum_unroll), 1))
            
            for _ in range(unroll_k):
                if block_loc.start > 0 and kv_cache is not None:
                    cross_block_loc = BlockLoc(block_loc.start-block_length, block_loc.end)
                    cross_block_x = x[:, block_loc.start-block_length:block_loc.end]
                    cross_block_replace_positions = (block_loc.start-block_length, block_loc.end)
                    
                    # --- FIX: Dynamically align cache length to shifted boundaries ---
                    target_cache_len = cross_block_loc.end - 1 if is_shifted else cross_block_loc.end
                    
                    if hasattr(kv_cache, 'past_key_values') and kv_cache.past_key_values is not None:
                        cache_data = kv_cache.past_key_values._data if hasattr(kv_cache.past_key_values, '_data') else kv_cache.past_key_values
                        
                        if isinstance(cache_data, torch.Tensor):
                            last_block_past_key_values = cache_data[:, :, :, :, :target_cache_len, :]
                        else:
                            last_block_past_key_values = []
                            for layer_cache in cache_data:
                                if isinstance(layer_cache, (tuple, list)):
                                    last_block_past_key_values.append((layer_cache[0][..., :target_cache_len, :], layer_cache[1][..., :target_cache_len, :]))
                                else:
                                    last_block_past_key_values.append(layer_cache[..., :target_cache_len, :])
                    else:
                        last_block_past_key_values = None

                    self.diff_iteration.forward(model, decoder, x, kv_cache, cross_block_x, cross_block_loc,
                                                block_id, pos_ids, attn_mask, last_block_past_key_values, cross_block_replace_positions, self.backend, is_cross_block=True, block_length=block_length)
                else:
                    # --- FIX: Dynamically align cache length to shifted boundaries ---
                    target_cache_len = block_loc.end - 1 if is_shifted else block_loc.end
                    
                    if hasattr(kv_cache, 'past_key_values') and kv_cache.past_key_values is not None:
                        cache_data = kv_cache.past_key_values._data if hasattr(kv_cache.past_key_values, '_data') else kv_cache.past_key_values
                        
                        if isinstance(cache_data, torch.Tensor):
                            past_key_values = cache_data[:, :, :, :, :target_cache_len, :]
                        else:
                            past_key_values = []
                            for layer_cache in cache_data:
                                if isinstance(layer_cache, (tuple, list)):
                                    past_key_values.append((layer_cache[0][..., :target_cache_len, :], layer_cache[1][..., :target_cache_len, :]))
                                else:
                                    past_key_values.append(layer_cache[..., :target_cache_len, :])
                    else:
                        past_key_values = kv_cache.past_key_values if hasattr(kv_cache, 'past_key_values') else kv_cache
                    
                    replace_position = (block_loc.start, block_loc.end)
                    
                    self.diff_iteration.forward(model, decoder, x, kv_cache, current_block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self.backend)
        
class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        cache_update_kv = None
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            cache_update_kv = output.past_key_values
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return cache_update_kv, logits

class BlockDiffusionIteration(BaseDiffusionIteration):
    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend, is_cross_block=False, block_length=32):
        
        # Slice the FULL mask down to the current block_loc boundaries
        if attn_mask is not None:
            sliced_attn_mask = attn_mask[..., block_loc.start:block_loc.end, :block_loc.end]
        else:
            sliced_attn_mask = None

        if kv_cache is None:
            output = model(x.data[:, :block_loc.end],
                           attention_mask=sliced_attn_mask if not is_cross_block else attn_mask[..., :block_loc.end, :block_loc.end],
                           position_ids=pos_ids[:, :block_loc.end])
            logits = output.logits[:, block_loc.start:block_loc.end]
        else:
            output = model(block.clone(memory_format=torch.contiguous_format),
                           attention_mask=sliced_attn_mask,
                           position_ids=pos_ids[:, block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                           use_cache=True, 
                           past_key_values=past_key_values,
                           replace_position=(0,0) if backend=='sglang' else replace_position)
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
                
            if is_cross_block:
                logits = output.logits[:, block_length:]
            else:
                logits = output.logits

        target_start = block_loc.start + block_length if is_cross_block else block_loc.start
        decoder.decode(logits, target_start, block_loc.end, x)

        self.num_forwards += 1
        return output


class ShiftDiffusionIteration(DiffusionIteration):
    """ A shift implementation of diffusion iteration to decode.
    """
    def __init__(self, use_shift = False):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        block_start, block_end = block_loc.start-1, block_loc.end-1
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_start, block_end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            # TODO(dulun): need to improve efficiency
            x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
            decoder.decode(output.logits[:, block_start:block_end], block_start, block_end, x_shifted)
            x.data[:, 1:] = x_shifted.data
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_start:block_end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_end - block_start
            logits = logits[:, :block_length]
        else:
            # cache position is the position between current_block_start and current_block_end
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:block_end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        # TODO(dulun): need to improve efficiency
        x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
        decoder.decode(logits, block_start, block_end, x_shifted)
        x.data[:, 1:] = x_shifted.data
        self.num_forwards += 1
        self.iter_no += 1

class BlockShiftDiffusionIteration(DiffusionIteration):
    """ Block diffusion iteration handling shifted labels (i-1 predicts i). """
    def __init__(self):
        super().__init__()
        self.num_forwards = 0
        self.cache_updates = 0
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend, is_cross_block=False, block_length=32):
        
        b_end = block_loc.end - 1

        if kv_cache is None:
            # Prefill phase
            feed_len = b_end
            prefill_mask = attn_mask[..., :feed_len, :feed_len] if attn_mask is not None else None
            
            output = model(x.data[:, :feed_len],
                           attention_mask=prefill_mask,
                           position_ids=pos_ids[:, :feed_len])
            
            if block_loc.start == 0:
                dummy_logit = torch.zeros_like(output.logits[:, 0:1])
                logits = torch.cat([dummy_logit, output.logits], dim=1)
            else:
                b_start = block_loc.start - 1
                logits = output.logits[:, b_start:b_end]
        else:
            # Decoding phase
            b_start = block_loc.start - 1
            if b_start < 0:
                shifted_block = torch.cat([x.data[:, 0:1], x.data[:, 0:b_end]], dim=1)
                sliced_pos_ids = torch.cat([pos_ids[:, 0:1], pos_ids[:, 0:b_end]], dim=1)
                if attn_mask is not None:
                    # FIX: Slice exactly to b_end to match the shifted KV cache length
                    sliced_attn_mask = torch.cat([attn_mask[..., 0:1, :b_end], attn_mask[..., 0:b_end, :b_end]], dim=-2)
                else:
                    sliced_attn_mask = None
            else:
                shifted_block = x.data[:, b_start:b_end]
                sliced_pos_ids = pos_ids[:, b_start:b_end]
                if attn_mask is not None:
                    # FIX: Slice exactly to b_end to match the shifted KV cache length
                    sliced_attn_mask = attn_mask[..., b_start:b_end, :b_end]
                else:
                    sliced_attn_mask = None

            output = model(shifted_block.clone(memory_format=torch.contiguous_format),
                           attention_mask=sliced_attn_mask,
                           position_ids=sliced_pos_ids.clone(memory_format=torch.contiguous_format),
                           use_cache=True, 
                           past_key_values=past_key_values,
                           replace_position=(0,0) if backend=='sglang' else replace_position)
            
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
                
            if is_cross_block:
                logits = output.logits[:, block_length:]
            else:
                logits = output.logits

        # --- DECODING TARGET ALIGNMENT ---
        target_start = block_loc.start + block_length if is_cross_block else block_loc.start
        decoder.decode(logits, target_start, block_loc.end, x)

        self.num_forwards += 1
        self.iter_no += 1
        return output

class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, use_shift=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        if use_shift:
            self.diff_iteration = ShiftDiffusionIteration()
        else:
            self.diff_iteration = BaseDiffusionIteration()
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class IterationSmooth(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            # update KV-cache
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if kv_cache is None:
            logits = model(inputs_embeds=self.inputs_embeds).logits
            decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:block_loc.end] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
        self.num_forwards += 1
        self.iter_no += 1

class IterSmoothDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.diff_iteration = IterationSmooth(self.model, cont_weight, cont_weight_init, cont_weight_growth, threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates
    
    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        self.diff_iteration.reset_input_embeds(x)
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class VicinityCacheIteration(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, prefix_look, after_look, warmup_steps):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        if self.iter_no < self.warmup_steps:
            out_full = model(x.data)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(x.data, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        window_input = x.data[:, left_start:right_end]
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(window_input, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
        self.num_forwards += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x)
        self.iter_no += 1

class VicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with Vicinity Cache Update.

    The decoding algorithm defines a window to update KV-cache in each diffusion iteration.
    The window can be larger than the decoding block.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = VicinityCacheIteration(prefix_look, after_look, warmup_steps)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class IterSmoothWithVicinityCache(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, prefix_look, after_look, warmup_steps,
            cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)

        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0
    
    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.iter_no < self.warmup_steps:
            out_full = model(inputs_embeds=self.inputs_embeds)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(
                inputs_embeds=self.inputs_embeds[:, left_start:right_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
        )

        self.num_forwards += 1
        self.iter_no += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
        mask_index = (x.data[:, left_start:right_end] == decoder.mask_id)
        self.inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

class IterSmoothWithVicinityCacheDiffusionLLM(IterSmoothDiffusionLLM):
    """ This diffusion LLM inference generates tokens with vicinity cache and iteration smoothing.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3,
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = IterSmoothWithVicinityCache(model, prefix_look, after_look, warmup_steps,
                cont_weight=cont_weight, cont_weight_init=cont_weight_init, cont_weight_growth=cont_weight_growth,
                threshold_decay=threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates


class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[:, (self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[:, self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()

class BlockDiffusionLLMAttnmask(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of Attention Mask.

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        assert prompt.shape[0] == 1, "We currently only support batch size = 1."
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length
        
        
        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0)


        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        # We don't need kv_cache for the implementation of attention mask
        kv_cache = None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, 
                pos_ids, bd_attn_mask)
            if decode_compl:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()


def gather_blocks(x: torch.Tensor, idx: torch.Tensor, block_length: int) -> torch.Tensor:
    # Gather blocks from a batch to form a mini-batch input ids, select on block from each sequence,
    # idx is a 1-d tensor indicating the block start position of each sequence, block length indecates the length of a block
    n, L = x.shape
    offsets = torch.arange(block_length, device=x.device).unsqueeze(0)          # (1, block_length)
    indices = idx.unsqueeze(1) + offsets                                       # (n, block_length)
    blocks = torch.gather(x, dim=1, index=indices)
    return blocks

def select_batch_sequences_by_mask_number(x, valid_flag, mask_id, batch_size):
    # Select sequences to build a mini-batch, using sequences with most mask tokens
    cand_idx = torch.nonzero(valid_flag, as_tuple=False).squeeze(1)  # shape (N,)
    _, sorted_order = torch.sort(-(x.data[cand_idx]==mask_id).sum(dim=1), stable=True)  
    top_order = sorted_order[:batch_size] 
    return cand_idx[top_order]

def select_batch_sequences_by_order(x, valid_flag, mask_id, batch_size):
    # Select sequences to build a mini-batch, selecting simply by sequence order
    return torch.nonzero(valid_flag, as_tuple=True)[0][:batch_size]

select_prefilling_batch_sequences = select_batch_sequences_by_mask_number
select_decoding_batch_sequences = select_batch_sequences_by_mask_number

class BlockDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of KV-Cache

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : CacheFactory
        The factory class that creates the KV-cache.
    early_stop : bool, default=True
        If True, generation of each sequence stops as soon as it has generated the EOS token.
    maximum_unroll : int, default=1
        Maximum number of forwards to unroll at a time, to reduce cuda graph overhead.
    expected_tpf : int, default=15
        Expected tokens per forward pass, used for controling unrolling).
    backend : str, default='vllm'
        Inference backend. Options include 'vllm', 'sglang'
    mini_batch_size : int, default=4
        Size of mini-batches used in dynamic batching.
    prefilling_limit : int, default=128
        Maximum length for prefilling the KV-cache, rest of the input prompt will be prefilled block by block.
    use_naive_batching : bool, default=True
        If True, uses naive batching; otherwise, uses dynamic batching.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, early_stop=True, maximum_unroll=1, expected_tpf=15, backend='vllm', 
                 mini_batch_size=4, prefilling_limit=128, use_naive_batching=True, use_shift=False):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.cache_factory = cache_factory
        self.use_shift = use_shift
        # Route to the correct iteration handler
        if self.use_shift:
            self.diff_iteration = BlockShiftDiffusionIteration()
        else:
            self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)        
        self.early_stop = early_stop
        self.backend = backend
        self.mini_batch_size = mini_batch_size
        self.prefilling_limit = prefilling_limit
        self.use_naive_batching = use_naive_batching
        is_phase_moe = getattr(self.model.model.config, "phase_moe_mode", "none") != "none"
        if is_phase_moe:
            assert self.backend == 'sglang' and not self.use_naive_batching, \
                "FATAL: Phase-MoE requires backend='sglang' and use_naive_batching=False to handle dynamic block alignments!"

        if self.use_naive_batching or self.backend != 'sglang':
            self.generate = self.naive_batching_generate
        else:
            self.generate = self.dynamic_batching_generate

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def naive_batching_generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        batch_size = prompt.shape[0]
        prompt_length = prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length = total_length-prompt_length
        
        mask_length = (max(self.cache_factory.max_length, prompt_length + gen_length)+block_length-1)//block_length*block_length
        attn_mask_num_blocks = mask_length // block_length

        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(attn_mask_num_blocks, attn_mask_num_blocks, device=self.model.device, dtype=torch.bool))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0).repeat(batch_size, 1)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        prompt_length = it._get_first_block_start()
        kv_cache = self.cache_factory.create()

        # prefill for kv_cache
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        prefill_length = max(prefill_length, block_length)
        self.block_runner.prefill(self.model, x[:, :prefill_length], kv_cache, pos_ids[:, :prefill_length], bd_attn_mask[:,:prefill_length,:prefill_length], self.prefilling_limit, block_length)
        
        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            if self.backend == 'vllm':
                cross_block_attn_mask = bd_attn_mask[:,block_loc.start-block_length:block_loc.end, :block_loc.end]
            else:
                cross_block_attn_mask = torch.ones(batch_size, 2*block_length, kv_cache.past_key_values._data.shape[4], device=prompt.device, dtype=torch.bool)
                cross_block_attn_mask[:, :block_length, -block_length:].fill_(False)
            # Pass bd_attn_mask to both slots; the BlockRunner ignores the second slot now.
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, pos_ids, bd_attn_mask, block_length, bd_attn_mask)
            if torch.all(decode_compl) and self.early_stop:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

    @ torch.no_grad()
    def dynamic_batching_generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with dynamic batching
        '''
        assert self.cache_factory is not None
        device = self.model.device
        mask_id = self.decoder.mask_id
        batch_size = prompt.shape[0]
        prompt_length = prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length = total_length-prompt_length
        
        mask_length = (max(self.cache_factory.max_length, prompt_length + gen_length)+block_length-1)//block_length*block_length
        attn_mask_num_blocks = mask_length // block_length
        mini_batch_size = self.mini_batch_size

        # Prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(attn_mask_num_blocks, attn_mask_num_blocks, device=device, dtype=torch.bool))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(mini_batch_size, 1, 1)
        pos_ids = torch.arange(total_length, device=device).unsqueeze(0).repeat(batch_size, 1)

        x = TokenArray(prompt, new_gen_length, mask_id, self.decoder.eos_id, device)

        prefilling_limit = self.prefilling_limit
        non_mask_number = (prompt != mask_id).sum(dim=-1)
        
        decoding_start = (non_mask_number // block_length) * block_length
        decoding_start = decoding_start.clip(0, prefilling_limit)

        prefilling_lengths = decoding_start.clip(0, prefilling_limit)

        # Initialize KV cache
        num_layers = self.model.model.config.num_hidden_layers
        num_kv_heads = self.model.model.config.num_key_value_heads
        num_heads = self.model.model.config.num_attention_heads
        head_dim = self.model.model.config.hidden_size // num_heads
        self.past_key_values = torch.zeros((num_layers, 2, batch_size, max(1, num_kv_heads//torch.distributed.get_world_size()), 
                                            self.model.max_length, head_dim), dtype=torch.bfloat16, device=device)

        # === PREFILLING PHASE ===
        # Prefill the KV cache with the initial prompt tokens up to prefilling_limit
        prefilling_flag = prefilling_lengths>0
        while torch.any(prefilling_flag):
            prefilling_seq_ids = select_prefilling_batch_sequences(x, prefilling_flag, mask_id, mini_batch_size)
            prefilling_length = max(prefilling_lengths[prefilling_seq_ids])

            prefilling_x = x.select_seqs(prefilling_seq_ids)
            
            # EXPLICIT PHASE 0 FOR PREFILL (Prevents SGLang positional arg bugs)
            phase_indices = torch.zeros_like(prefilling_x[:, :prefilling_length], dtype=torch.long)

            output = self.model(
                prefilling_x[:, :prefilling_length].clone(memory_format=torch.contiguous_format), 
                use_cache=True, 
                attention_mask=bd_attn_mask[:len(prefilling_seq_ids), :prefilling_length, :prefilling_length].clone(memory_format=torch.contiguous_format), 
                position_ids=pos_ids[prefilling_seq_ids,:prefilling_length].clone(memory_format=torch.contiguous_format),
                phase_indices=phase_indices, 
            )
            inner_shape = output.past_key_values[0].shape
            prefilling_kv = torch.stack(output.past_key_values, dim=0).reshape(num_layers, 2, *inner_shape)
            for id, sample_prefilling_length in enumerate(prefilling_lengths[prefilling_seq_ids]):
                self.past_key_values[:, :, prefilling_seq_ids[id], :, :sample_prefilling_length] = prefilling_kv[:, :, id, :, :sample_prefilling_length]
                self.past_key_values[:, :, prefilling_seq_ids[id], :, sample_prefilling_length:] = 0
            self.diff_iteration.num_forwards +=1
            prefilling_flag[prefilling_seq_ids] = False
            
        # === DECODING PHASE ===
        # This loop performs block-by-block block diffusion generation using dynamic batching over sequences that are still active.
        # The outer loop iterates over varying cache lengths, progressively expanding the attention context.
        # The inner loop handles dynamic batching for sequences that fit within the current cache capacity.
        # In each inner iteration:
        #   we first select sequences and build batch for forward pass.
        #   Then logits are used to update sequences, and sequences with finished blocks update the KV cache.
        #   Completed sequences (generated EOS) are marked to exit early if enabled.
        decoding_flag = (decoding_start+block_length)<=total_length
        
        while torch.any(decoding_flag):
            # Dynamically adjust cache size based on the earliest decoding position
            current_cache_length = max(128, align_exp2(min(decoding_start[decoding_flag])+block_length))
            current_cache_flag = decoding_flag & ((decoding_start+block_length)<=current_cache_length)
            
            while torch.any(current_cache_flag):
                decoding_seq_ids = select_decoding_batch_sequences(x, current_cache_flag, mask_id, mini_batch_size)
                decoding_x = x.select_seqs(decoding_seq_ids)
                bsz = len(decoding_seq_ids)
                
                # --- ROBUST SHIFT FIX ---
                shift_offset = 1 if self.use_shift else 0
                feed_start = torch.clamp(decoding_start[decoding_seq_ids] - shift_offset, min=0)
                
                decoding_block = gather_blocks(decoding_x.data, feed_start, block_length)
                decoding_past_key_values = self.past_key_values[:, :, decoding_seq_ids, :, :current_cache_length]
                
                decoding_pos_ids = torch.arange(block_length, device=device, dtype=torch.long).repeat(bsz, 1)
                decoding_pos_ids = decoding_pos_ids + feed_start.unsqueeze(1)
                
                # --- ATTENTION MASK FIX ---
                # Create mask to block out padding in SGLang's statically-sized KV cache graph
                attn_mask = torch.zeros((bsz, block_length, current_cache_length), dtype=torch.bool, device=device)
                for i_batch in range(bsz):
                    fs = feed_start[i_batch].item()
                    
                    # 1. All tokens can attend to the clean prefix
                    attn_mask[i_batch, :, :fs] = True
                    
                    if self.use_shift:
                        # 2. The first token in the query block is the clean x_{P-1}. 
                        # It ONLY attends to itself and the prefix. It MUST NOT attend to the noisy tokens.
                        attn_mask[i_batch, 0, current_cache_length - block_length] = True
                        
                        # 3. The remaining tokens are the noisy mask tokens. 
                        # They attend to x_{P-1} and to each other.
                        attn_mask[i_batch, 1:, current_cache_length - block_length : current_cache_length] = True
                    else:
                        # Standard MDLM: all tokens are noisy, they attend to each other
                        attn_mask[i_batch, :, current_cache_length - block_length : current_cache_length] = True
                # --------------------------
                phase_block_size = getattr(self.model.model.config, "phase_block_size", 32)
                assert block_length >= phase_block_size and block_length % phase_block_size == 0, \
                    f"Inference block_length ({block_length}) must be a multiple of phase_block_size ({phase_block_size})."
                num_phase_blocks = decoding_block.shape[1] // phase_block_size

                # Forward Pass (Now protected by the compiled CUDA graph mask)
                # EXPLICIT PHASE DEDUCTION (Bypasses SGLang internal wrapper tracking bugs)
                phase_block_size = getattr(self.model.model.config, "phase_block_size", 32)
                assert block_length >= phase_block_size and block_length % phase_block_size == 0, \
                    f"Inference block_length ({block_length}) must be a multiple of phase_block_size ({phase_block_size})."

                phase_indices = torch.zeros_like(decoding_block, dtype=torch.long)
                for i in range(0, block_length, phase_block_size):
                    chunk = decoding_block[:, i:i+phase_block_size]
                    mask_counts = (chunk == mask_id).sum(dim=1, keepdim=True)
                    phase_indices[:, i:i+phase_block_size] = mask_counts.expand(-1, chunk.shape[1])

                # Forward Pass (Now protected by the compiled CUDA graph mask)
                output = self.model(
                    decoding_block, 
                    use_cache=True, 
                    position_ids=decoding_pos_ids, 
                    past_key_values=decoding_past_key_values, 
                    attention_mask=attn_mask,
                    phase_indices=phase_indices,
                )
                
                logits = output.logits[:bsz]
                self.decoder.batch_decode(logits, decoding_start[decoding_seq_ids], decoding_x, block_length)
                
                # Check completion on actual target block
                actual_target_block = gather_blocks(decoding_x.data, decoding_start[decoding_seq_ids], block_length)
                block_finished = (actual_target_block == mask_id).sum(dim=1) == 0
                
                inner_shape = output.past_key_values[0].shape
                decoding_kv = torch.stack(output.past_key_values, dim=0).reshape(num_layers, 2, *inner_shape)[:, :, :block_finished.shape[0]]
                
                block_idx_matrix = feed_start[block_finished].unsqueeze(1) + torch.arange(block_length, device=device)
                
                # SGLang scatters to the end regardless of shift_offset; extract from the end and write contiguously
                self.past_key_values[:, :, decoding_seq_ids[block_finished].unsqueeze(1), :, block_idx_matrix.long()] = \
                    decoding_kv.permute(2, 4, 0, 1, 3, 5)[block_finished, current_cache_length - block_length : current_cache_length]
                # --- END SHIFT FIX ---

                # 1. Evaluate EOS detection using the CURRENT decoding_start before incrementing it
                global_positions = decoding_start[decoding_seq_ids].unsqueeze(1) + torch.arange(block_length, device=device).unsqueeze(0)
                is_generated_pos = global_positions >= non_mask_number[decoding_seq_ids].unsqueeze(1)
                eos_detected = (actual_target_block == self.decoder.eos_id) & is_generated_pos
                
                eos_mask = (torch.any(eos_detected, dim=1) & block_finished)
                
                eos_indices = eos_mask.nonzero(as_tuple=True)[0]
                
                # 2. Increment decoding_start now that index math is done
                decoding_start[decoding_seq_ids] += block_finished.long()*block_length
                x[decoding_seq_ids] = decoding_x.data
                
                # 3. Handle early stop
                if self.early_stop and eos_indices.numel() > 0:
                    stop_seq_ids = decoding_seq_ids[eos_indices]
                    decoding_start[stop_seq_ids] = total_length 
                    decoding_flag[stop_seq_ids] = False

                self.diff_iteration.num_forwards +=1
                decoding_flag = decoding_flag & ((decoding_start+block_length)<=total_length)
                current_cache_flag = decoding_flag & ((decoding_start+block_length)<=current_cache_length)
                
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

