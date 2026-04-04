import argparse
import time

import torch
from model_triton import model_triton
from reference import ModelBuffers, ModelParams, model_ref
from tokenizers.decoders import DecodeStream
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class HFGenerator:
    def __init__(self, model_id: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="cuda").eval()
        self.eos_token_id = AutoTokenizer.from_pretrained(model_id).eos_token_id

    @torch.no_grad()
    def generate(self, token_ids: list[int], max_tokens: int = 1024):
        input_ids = torch.tensor(token_ids, device="cuda")
        kv_cache = DynamicCache()

        for _ in range(max_tokens):
            logits = self.model(input_ids.unsqueeze(0), past_key_values=kv_cache).logits.squeeze(0)
            input_ids = logits[-1].argmax(dim=0, keepdim=True)
            token = input_ids.item()
            yield token

            if token == self.eos_token_id:
                break


class MyGenerator:
    def __init__(self, model_id: str):
        self.params = ModelParams.from_pretrained(model_id).to("cuda")
        self.eos_token_id = AutoTokenizer.from_pretrained(model_id).eos_token_id

        embeds = self.params.input_embeds
        self.buffers = ModelBuffers.create(
            self.params.num_kv_heads,
            self.params.num_layers,
            device=embeds.device,
            kv_dtype=embeds.dtype,
        )

    def generate(self, token_ids: list[int], max_tokens: int = 1024):
        input_ids = torch.tensor(token_ids, device="cuda")
        self.buffers.position = 0

        # prefill
        input_ids = model_ref(input_ids, self.params, self.buffers).unsqueeze(0)
        token = input_ids.item()
        yield token

        if token == self.eos_token_id:
            return

        # decode
        for _ in range(max_tokens - 1):
            input_ids = model_triton(input_ids, self.params, self.buffers).unsqueeze(0)
            token = input_ids.item()
            yield token

            if token == self.eos_token_id:
                return


class VllmGenerator:
    def __init__(self, model_id: str):
        from vllm import EngineArgs, LLMEngine

        self.llm = LLMEngine.from_engine_args(EngineArgs(model_id))

    def generate(self, token_ids: list[int], max_tokens: int = 1024):
        from vllm import SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        llm = self.llm
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            output_kind=RequestOutputKind.DELTA,
            detokenize=False,
        )
        llm.add_request("1234", TokensPrompt(prompt_token_ids=token_ids), sampling_params)
        while llm.has_unfinished_requests():
            for output in llm.step():
                yield from output.outputs[0].token_ids


def main(args: argparse.Namespace):
    model_id = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    decoder = dict(
        hf=HFGenerator,
        vllm=VllmGenerator,
        my=MyGenerator,
    )[args.impl](model_id)

    messages = []
    do_warmup = True

    while True:
        prompt = "hi" if do_warmup else input("> ")
        messages.append(dict(role="user", content=prompt))

        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_attention_mask=False,
        )
        # tokenizer changes behavior after certain version
        if not isinstance(token_ids, list):
            token_ids = token_ids["input_ids"]

        stream = DecodeStream(skip_special_tokens=True)
        outputs = []

        gen = decoder.generate(token_ids, max_tokens=100)

        # prefill
        t0 = time.perf_counter()
        token = next(gen)
        to_print = stream.step(tokenizer._tokenizer, token)
        if not do_warmup and to_print is not None:
            outputs.append(to_print)
            print(to_print, end="", flush=True)
        t1 = time.perf_counter()

        # decode
        for token in gen:
            to_print = stream.step(tokenizer._tokenizer, token)
            if not do_warmup and to_print is not None:
                outputs.append(to_print)
                print(to_print, end="", flush=True)
        t2 = time.perf_counter()

        if not do_warmup:
            print()
            print(f"Prefill: {len(token_ids)} tokens, {len(token_ids) / (t1 - t0):,.2f} tok/s")
            print(f"Decode: {len(outputs)} tokens, {len(outputs) / (t2 - t1):,.2f} tok/s")

            # update chat history
            messages.append(dict(role="assistant", content="".join(outputs)))

        do_warmup = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--impl", choices=["hf", "vllm", "my"], default="my")
    args = parser.parse_args()

    main(args)
