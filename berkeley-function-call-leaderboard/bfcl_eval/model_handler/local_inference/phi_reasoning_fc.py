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
    Reasoning tokens: <think>...</think> or <nothink>
    """

    # Supported tool-spec shapes (see _format_tool_entry below). Picks the
    # outer wrap (single vs double "function" nesting) AND the inner
    # "parameters" container shape (object/dict/props-no-type/flat). Choose
    # to match whichever shape dominates the model's training mix.
    SUPPORTED_TOOL_FORMATS = (
        "double/object",        # default — matches phi4rv training majority
        "single/object",
        "single/dict",          # BFCL's native shape (what `function` already is)
        "single/props-no-type",
        "single/flat",
    )

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        thinking_mode=False,
        # Tried single/dict (matches vulcan_bfcl training shape) but it cost
        # ~16-25pp on parallel_multiple categories vs. single/object — the
        # +10pp irrelevance bump didn't compensate. Keeping object as default.
        tool_format="single/object",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.is_fc_model = True
        self.thinking_mode = thinking_mode
        self.skip_special_tokens = False
        if tool_format not in self.SUPPORTED_TOOL_FORMATS:
            raise ValueError(
                f"tool_format={tool_format!r} not in {self.SUPPORTED_TOOL_FORMATS!r}"
            )
        self.tool_format = tool_format

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
            # Structure-Rules preamble. Tried matching vulcan_bfcl's bare
            # `# Tools` format (the BFCL-mirror training set), but it dropped
            # single_turn 81.75 -> 44.25 and multi_turn 29.38 -> 5.38 — the
            # other 9+ phi4rv tool-calling datasets all use Structure-Rules
            # preambles, so that's what the model has internalized as
            # "tool-use mode."
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

        tool_list = [self._format_tool_entry(func) for func in function]
        tool_contents = json.dumps(tool_list)

        system_content = (
            f"{self.SYSTEM_PROMPT_PREAMBLE}\n"
            f"{self._tool_format_blurb()}{extra_system}\n\n\n"
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
            formatted_prompt += "<nothink>"

        return formatted_prompt

    # --- tool-spec formatting ---------------------------------------------
    # BFCL hands `function` to us as the "single/dict" shape:
    #   {"name": ..., "description": ...,
    #    "parameters": {"type": "dict", "properties": {...}, "required": [...]}}
    # The phi4rv training mix renders the same tool spec in five different
    # shapes; the model is sensitive to which one it sees. We render to match
    # whichever shape the model saw most often during SFT.

    _FORMAT_BLURBS = {
        "double/object": 'Each tool is wrapped as {"type":"function","function":{"type":"function","function":{...}}} with JSON Schema parameters (parameters.type = "object").',
        "single/object": 'Each tool is wrapped as {"type":"function","function":{...}} with JSON Schema parameters (parameters.type = "object").',
        "single/dict":   'Each tool is wrapped as {"type":"function","function":{...}} with parameters.type = "dict".',
        "single/props-no-type": 'Each tool is wrapped as {"type":"function","function":{...}} with a parameters block containing properties and required (no top-level type field).',
        "single/flat":   'Each tool is wrapped as {"type":"function","function":{...}} with a flat parameters map: {param_name: {"description": ..., "type": "<py-type>"}}; optional params have a ", optional" suffix on the type.',
    }

    _JSONSCHEMA_TO_PY = {
        "integer": "int",
        "string":  "str",
        "boolean": "bool",
        "number":  "float",
        "array":   "List",
        "object":  "Dict",
    }

    def _tool_format_blurb(self) -> str:
        return self._FORMAT_BLURBS[self.tool_format]

    def _format_tool_entry(self, func: dict) -> dict:
        fmt = self.tool_format

        if fmt == "single/dict":
            # BFCL native — pass through.
            return {"type": "function", "function": func}

        f = dict(func)
        params = dict(func.get("parameters") or {})

        if fmt == "single/object":
            params["type"] = "object"
            f["parameters"] = params
            return {"type": "function", "function": f}

        if fmt == "double/object":
            params["type"] = "object"
            f["parameters"] = params
            return {"type": "function", "function": {"type": "function", "function": f}}

        if fmt == "single/props-no-type":
            params.pop("type", None)
            f["parameters"] = params
            return {"type": "function", "function": f}

        if fmt == "single/flat":
            f["parameters"] = self._flatten_parameters(params)
            return {"type": "function", "function": f}

        raise ValueError(f"unknown tool_format {fmt!r}")

    def _flatten_parameters(self, params: dict) -> dict:
        """Render {properties, required} as a flat {name: {description, type}} map
        using Python-style type names; optional params get a ', optional' suffix."""
        properties = params.get("properties") or {}
        required = set(params.get("required") or [])
        out: dict = {}
        for name, spec in properties.items():
            entry = dict(spec)
            t = entry.get("type")
            if isinstance(t, str):
                py = self._JSONSCHEMA_TO_PY.get(t, t)
                if py == "List":
                    inner = (spec.get("items") or {}).get("type")
                    if isinstance(inner, str):
                        py = f"List[{self._JSONSCHEMA_TO_PY.get(inner, inner)}]"
                if name not in required:
                    py = f"{py}, optional"
                entry["type"] = py
            out[name] = entry
        return out

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
        """Strip <think>...</think> blocks and <nothink> tokens from the response."""
        # Remove <think>...</think> blocks
        if "</think>" in text:
            text = text.split("</think>")[-1]
        # Remove <nothink> token
        text = text.replace("<nothink>", "")
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


# Fara CUA system prompt. Tools are inlined inside <tools>...</tools> (note the
# plural — distinct from Phi-4-reasoning's separate "# Available Tools" + <tool>
# block). Everything else (chat template, tool_call/tool_response tokens,
# reasoning tokens) matches PhiReasoningFCHandler.
FARA_BUNNY_PHI4_FN_CALL_TEMPLATE = """You are Fara, a computer use agent (CUA) specialized for web browsers. You are developed by Microsoft AI Frontiers. You assist users with completing and automating tasks that require the use of a web browser.

The model was trained in the timeframe of January - March 2026. You can effectively perform tasks even beyond this range by accessing the web browser and using the latest information on the live web. But your knowledge cutoff is limited to early 2026, so you may not be aware of events or developments that occurred after that time, without explicitly browsing and searching for latest information on the web.

This edition of the model was trained using SFT on top of Phi-4-vision-5B, using a synthetic data mixture generated and developed by Microsoft AI Frontiers.

A critical point is a situation where we must pause and request information or confirmation from the user before proceeding. There are three types:

Case 1: Missing User Information — The task requires personal information that the user has not provided (e.g., email, phone number, address, payment details). Never fabricate or assume personal information. Fill in only what the user has explicitly provided, then pause and ask for any missing required fields. (e.g., form requires phone number but user only gave name and email -> fill name and email, then ask for phone number.) If the user has provided all required information, proceed without stopping.

Case 2: Underspecified Task — The task description is ambiguous or missing details needed to make a decision at the current step. Pause and ask for clarification. (e.g., user asks to book a flight but doesn't specify destination -> ask for destination.) If the user's instructions contain all information needed for the current decision, proceed without stopping.

Case 3: Irreversible Action — We are about to perform an action that cannot be undone (e.g., submitting a form, completing a purchase, sending a message, deleting data). If the user explicitly authorized the action (e.g., "submit the form", "complete the purchase", "you have my permission to submit") -> proceed without stopping. If the user did NOT explicitly authorize the action -> stop and ask for confirmation. (e.g., "fill out a form" with no mention of submitting -> fill the form, then ask before submitting; "fill out and submit a form" -> fill and submit without stopping.)

Only stop at a critical point if (1) required information is missing, (2) the task is ambiguous, OR (3) an irreversible action lacks explicit user authorization. If the user has provided all necessary information AND explicitly authorized the action, proceed without interruption.

You are provided with function signatures within <tools></tools> XML tags:
<tools>{tool_contents}</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{{"name": <function-name>, "arguments": <args-json-object>}}</tool_call>"""


class FaraPhiReasoningFCHandler(PhiReasoningFCHandler):
    """Fara CUA variant on top of Phi-4-reasoning function-calling.

    Same chat template (<|im_start|>/<|im_sep|>/<|im_end|>) and tool-call tokens
    as PhiReasoningFCHandler; only the system message differs — it uses the
    Fara CUA preamble and inlines tools as <tools>...</tools>.
    """

    def __init__(self, *args, tool_format="single/object", **kwargs):
        super().__init__(*args, tool_format=tool_format, **kwargs)

    @override
    def _format_prompt(self, messages, function):
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert 0 <= len(system_messages) <= 1

        extra_system = ""
        if messages[0]["role"] == "system":
            extra_system = "\n\n" + messages[0]["content"]
            messages = messages[1:]

        tool_list = [self._format_tool_entry(func) for func in function]
        tool_contents = json.dumps(tool_list)

        system_content = (
            FARA_BUNNY_PHI4_FN_CALL_TEMPLATE.format(tool_contents=tool_contents)
            + "\n"
            + self._tool_format_blurb()
            + extra_system
        )

        formatted_prompt = f"<|im_start|>system<|im_sep|>{system_content}<|im_end|>\n"
        for msg in messages:
            formatted_prompt += (
                f"<|im_start|>{msg['role']}<|im_sep|>{msg['content']}<|im_end|>\n"
            )
        formatted_prompt += "<|im_start|>assistant<|im_sep|>"
        formatted_prompt += "<think>" if self.thinking_mode else "<nothink>"
        return formatted_prompt
