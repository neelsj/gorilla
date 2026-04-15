from typing import Any

from bfcl_eval.model_handler.local_inference.phi import PhiHandler
from overrides import override


class PhiReasoningHandler(PhiHandler):
    """
    Handler for Microsoft Phi-4 reasoning models (e.g., Phi-4-reasoning-vision-15B).

    These models use the same chat template as Phi-4 (<|im_start|>/<|im_sep|>/<|im_end|>)
    but produce <think>...</think> reasoning blocks that must be stripped before decoding.
    """

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        reasoning_content = ""
        if "</think>" in model_response:
            reasoning_content = model_response.split("</think>")[0]
            model_response = model_response.split("</think>")[-1]

        return {
            "model_responses": model_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            }
        )
        return inference_data
