import argparse
import time

import torch
from model_triton import model_triton
from reference import ModelBuffers, ModelParams, model_ref
from tokenizers.decoders import DecodeStream
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class HFDecoder:
    def __init__(self, model):
        self.model = model.to("cuda")
        self.reset()

    def reset(self):
        self.kv_cache = DynamicCache()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor):
        logits = self.model(input_ids.unsqueeze(0), past_key_values=self.kv_cache).logits.squeeze(0)
        return logits[-1].argmax(dim=0)


class ReferenceDecoder:
    def __init__(self, model):
        self.params = ModelParams.from_state_dict(model.state_dict())
        self.params = self.params.to("cuda")

        embeds = self.params.input_embeds
        self.buffers = ModelBuffers.create(
            self.params.num_kv_heads,
            self.params.num_layers,
            device=embeds.device,
            kv_dtype=embeds.dtype,
        )
        torch.cuda.empty_cache()

    def reset(self):
        self.buffers.position = 0

    def __call__(self, input_ids: torch.Tensor):
        if input_ids.shape[0] == 1:
            return model_triton(input_ids, self.params, self.buffers)
        else:
            return model_ref(input_ids, self.params, self.buffers)


def main(args: argparse.Namespace):
    model_id = args.model
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: have to be careful NOT to duplicate model weights in VRAM
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # decoder = HFDecoder(model)
    decoder = ReferenceDecoder(model)

    messages = []
    do_warmup = True

    while True:
        prompt = "hi" if do_warmup else input("> ")
        messages.append(dict(role="user", content=prompt))

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_attention_mask=False,
        )["input_ids"]
        input_ids = input_ids.squeeze(0).to(device)
        num_prefill_tokens = input_ids.shape[0]

        # redo prefill everytime
        decoder.reset()
        stream = DecodeStream(skip_special_tokens=True)
        outputs = []

        def forward(input_ids: torch.Tensor):
            input_ids = decoder(input_ids).unsqueeze(0)
            token = input_ids.item()

            to_print = stream.step(tokenizer._tokenizer, token)
            if not do_warmup and to_print is not None:
                outputs.append(to_print)
                print(to_print, end="", flush=True)

            return input_ids, token

        t0 = time.perf_counter()
        input_ids, token = forward(input_ids)  # prefill
        t1 = time.perf_counter()

        # decode
        for _ in range(1024):
            input_ids, token = forward(input_ids)
            if token == tokenizer.eos_token_id:
                print()
                break
        t2 = time.perf_counter()

        if not do_warmup:
            print(f"Prefill: {num_prefill_tokens} tokens, {num_prefill_tokens / (t1 - t0):.2f} tok/s")
            print(f"Decode: {len(outputs)} tokens, {len(outputs) / (t2 - t1):.2f} tok/s")

            # update chat history
            messages.append(dict(role="assistant", content="".join(outputs)))

        do_warmup = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device")
    args = parser.parse_args()

    main(args)
