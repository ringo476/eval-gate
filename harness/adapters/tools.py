"""
Tools available to the tool-calling agent adapter.

The calculator deliberately does NOT use eval() on model-supplied input --
even in a demo/toy tool, evaluating an LLM's output as Python is a real
injection surface (the model is an untrusted input source here, same as
user input would be). It parses a restricted arithmetic grammar with
Python's `ast` module instead, allowing only numbers and +-*/() and
rejecting everything else, including any attribute access, name lookup,
or function call.
"""
from __future__ import annotations

import ast
import operator

from harness.core.corpus_search import search_corpus

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluates a basic arithmetic expression (numbers, + - * / and parentheses only). "
                        "Use this for any arithmetic instead of computing it yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '(12 + 8) * 3'"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "kb_lookup",
        "description": "Searches a small internal knowledge base of climate-change facts and returns the "
                        "most relevant entry. Use this for any factual question instead of guessing.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "What to look up, e.g. 'ocean acidification'"},
            },
            "required": ["topic"],
        },
    },
]

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    pass


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculatorError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    raise CalculatorError(f"disallowed expression element: {type(node).__name__}")


MAX_EXPRESSION_CHARS = 200  # model output is untrusted; bound the work a single call can cause


def calculator(expression: str) -> str:
    if not isinstance(expression, str):
        return "error: expression must be a string"
    if len(expression) > MAX_EXPRESSION_CHARS:
        return f"error: expression too long (max {MAX_EXPRESSION_CHARS} characters)"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)
        return str(result)
    except CalculatorError as e:
        return f"error: {e}"
    except RecursionError:
        # deeply nested parentheses blow the parser/evaluator stack
        return "error: expression nested too deeply"
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError, OverflowError, MemoryError) as e:
        return f"error: could not evaluate '{expression[:60]}' ({e})"


def kb_lookup(topic: str) -> str:
    matches = search_corpus(topic, top_k=1)
    if not matches:
        return "no relevant entry found"
    doc_id, text, _score = matches[0]
    return f"[{doc_id}] {text}"


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "calculator":
        return calculator(tool_input.get("expression", ""))
    if name == "kb_lookup":
        return kb_lookup(tool_input.get("topic", ""))
    return f"error: unknown tool '{name}'"
