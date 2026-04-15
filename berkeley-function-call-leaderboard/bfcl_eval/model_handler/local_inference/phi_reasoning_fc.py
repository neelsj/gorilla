import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class PhiReasoningFCHandler(OSSHandler):
    """
    Function-calling handler for Microsoft Phi-4 reasoning models
    (e.g., Phi-4-reasoning-vision-15B, Phi-4-reasoning-vision-5B).

    Chat template: <|im_start|>/<|im_sep|>/<|im_end|>
    Tool tokens: <tool>...</tool>, <tool_call>...</tool_call>, <tool_response>...</tool_response>
    Reasoning tokens: <think>...</think> or <|dummy_84|>
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        thinking_mode=False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.is_fc_model = True
        self.thinking_mode = thinking_mode

        if self.thinking_mode:

            self.SYSTEM_PROMPT_PREAMBLE = (
                "You are Phi-4-reasoning-vision, a multimodal model trained by Microsoft to help users.  Your role as an assistant is to provide accurate, coherent, and actionable responses, adapting your reasoning mode automatically based on the complexity, clarity, and confidence of each task.\n\n"
                "Structure Rules:\n"
                "1. All reasoning goes between <think> and </think> (thinking block). \n"
                "2. Whenever a tool would improve your answer, invoke it using <tool_call>...</tool_call> instead of relying solely on memory.\n"
                "3. Issue one or multiple tool calls <tool_call></tool_call>...<tool_call></tool_call> at a time; "
                "when tool calls can't be called in parallel you can sequentially interleave throughout the reasoning process "
                "(using the result of one to guide the call of the other). \n"
                "4. After each tool call or calls, the results of each tool call will be provided in a user message within the "
                "<tool_response></tool_response>...<tool_response></tool_response> tags.\n"
                "5. Stop the generation only after reaching the final answer.\n"
                "\n"
                "You can utilize the tools as many times as required. For example, "
                "<think>reasoning here</think><tool_call>tool call here</tool_call>"
                "<tool_response>output of tool call</tool_response>"
                "<think> reasoning here</think>final answer here (or more tool calls).\n"
                "\n"
                '# Format for tool calls: <tool_call>{"name": <function-name>,"arguments": <args-json-object>}</tool_call>\n'
                "\n"
                "# Available Tools\n"
                "You are provided with function signatures within <tool></tool> tags."
            )
        else:
            self.SYSTEM_PROMPT_PREAMBLE = (
                "You are Phi-4-reasoning-vision, a multimodal model trained by Microsoft to help users.  Your role as an assistant is to provide accurate, coherent, and actionable responses.\n\n"
                "Structure Rules:\n"
                "1. Whenever a tool would improve your answer, invoke it using <tool_call>...</tool_call> instead of relying solely on memory.\n"
                "2. Issue one or multiple tool calls <tool_call></tool_call>...<tool_call></tool_call> at a time; "
                "when tool calls can't be called in parallel you can sequentially interleave throughout the process "
                "(using the result of one to guide the call of the other). \n"
                "3. After each tool call or calls, the results of each tool call will be provided in a user message within the "
                "<tool_response></tool_response>...<tool_response></tool_response> tags.\n"
                "4. Stop the generation only after reaching the final answer.\n"
                "\n"
                "You can utilize the tools as many times as required. For example, "
                "<tool_call>tool call here</tool_call>"
                "<tool_response>output of tool call</tool_response>"
                "final answer here (or more tool calls).\n"
                "\n"
                '# Format for tool calls: <tool_call>{"name": <function-name>,"arguments": <args-json-object>}</tool_call>\n'
                "\n"
                "# Available Tools\n"
                "You are provided with function signatures within <tool></tool> tags."
            )

    @override
    def _format_prompt(self, messages, function):
        # sanity check
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert 0 <= len(system_messages) <= 1

        # extract any existing system message content
        extra_system = ""
        if messages[0]["role"] == "system":
            extra_system = "\n\n" + messages[0]["content"]
            messages = messages[1:]

        # format tool definitions as {"type":"function","function":{...}}
        tool_list = []
        for func in function:
            tool_list.append({"type": "function", "function": func})
        tool_contents = json.dumps(tool_list)

        system_content = (
            f"{self.SYSTEM_PROMPT_PREAMBLE}{extra_system}\n\n\n"
            f"<tool>{tool_contents}</tool>"
        )

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_content}<|im_end|>\n"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_prompt += f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant<|im_sep|>"

        if self.thinking_mode:
            formatted_prompt += "<think>"
        else:
            formatted_prompt += "<|dummy_84|>"

        return formatted_prompt

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if type(result) != list or any(type(item) != dict for item in result):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return result

    @override
    def decode_execute(self, result, has_tool_call_tag):
        if type(result) != list or any(type(item) != dict for item in result):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call(result)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_responses_message_for_chat_history = api_response.choices[0].text
        model_responses = api_response.choices[0].text

        # Strip reasoning blocks
        model_responses = self._strip_reasoning(model_responses)

        extracted_tool_calls = self._extract_tool_calls(model_responses)

        if (
            self._is_tool_call_response_format(extracted_tool_calls)
            and len(extracted_tool_calls) > 0
        ):
            model_responses = [
                {item["name"]: item.get("arguments", {})} for item in extracted_tool_calls
            ]

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        # Training data shows tool responses as role="user" with <tool_response> tags
        # Multiple results are concatenated in a single user message
        tool_response_content = "".join(
            f"<tool_response>{result}</tool_response>" for result in execution_results
        )
        inference_data["message"].append(
            {
                "role": "user",
                "content": tool_response_content,
            }
        )
        return inference_data

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

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Strip <think>...</think> blocks and <|dummy_84|> tokens from the response."""
        # Remove <think>...</think> blocks
        if "</think>" in text:
            text = text.split("</think>")[-1]
        # Remove <|dummy_84|> token
        text = text.replace("<|dummy_84|>", "")
        return text.strip()

    @staticmethod
    def _extract_tool_calls(input_string: str) -> list:
        # Match both <tool_call>...</tool_call> and bare {...}</tool_call> (missing opening tag)
        # The model often drops the opening <tool_call> on 2nd+ calls in a turn
        pattern = r"(?:<tool_call>)?(.*?)</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Fallback: <tool_call> present but missing closing tag
        if not matches:
            pattern = r"<tool_call>(.*?)(?:</tool_call>)?$"
            matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            match = match.strip()
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue

            if type(parsed) is list:
                result.extend(parsed)
            elif type(parsed) is dict:
                result.append(parsed)

        return result

    @staticmethod
    def _is_tool_call_response_format(input: list) -> bool:
        if type(input) != list:
            return False

        for item in input:
            if type(item) != dict:
                return False
            if "name" not in item:
                return False

        return True
