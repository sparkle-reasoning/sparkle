#!/usr/bin/env python3
"""
Sparkle Mathematical Reasoning Score Computation

This module provides comprehensive scoring functions for evaluating mathematical reasoning
responses from language models. It includes:

- Answer extraction from various formats (\\boxed{}, <answer> tags, etc.)
- Format validation for structured responses
- Mathematical equivalence checking using both symbolic and string-based methods
- Flexible scoring schemes for different evaluation scenarios

The scoring system is designed to evaluate both the correctness of mathematical answers
and the quality of response formatting, making it suitable for training and evaluation
of mathematical reasoning models.

Main API:
    MathScorer: The primary class for mathematical response evaluation
    compute_score: Legacy function for backward compatibility

Usage:
    scorer = MathScorer(debug_probability=0.1)
    score = scorer.compute_score(response, ground_truth, scoring_mode="hierarchical") 
    We recommend using the hierarchical mode for training which achieves the best performance
    among all scoring modes for training and is the same as the sparkle paper.
    
    For augmented responses (partial format), use:
    score = scorer.compute_score(response, ground_truth, scoring_mode="hierarchical_aug", extra_info=extra_info)
"""

import re
import importlib.util
import sys
from typing import Dict, Tuple, Optional, Union, List
import random

# LaTeX parsing setup with error handling
HAS_LATEX_PARSER = False
if importlib.util.find_spec("pylatexenc") is not None:
    try:
        import pylatexenc.latexwalker
        from pylatexenc import latex2text
        
        # Patch pylatexenc to handle mismatched braces gracefully
        original_get_latex_nodes = pylatexenc.latexwalker.LatexWalker.get_latex_nodes
        
        def patched_get_latex_nodes(self, *args, **kwargs):
            try:
                return original_get_latex_nodes(self, *args, **kwargs)
            except Exception as e:
                if "Unexpected mismatching closing brace" in str(e):
                    return []
                raise
        
        pylatexenc.latexwalker.LatexWalker.get_latex_nodes = patched_get_latex_nodes
        HAS_LATEX_PARSER = True
        
    except ImportError:
        pass

# SymPy setup
try:
    import sympy
    from sympy.parsing import sympy_parser
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

# Constants for validation
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"
UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "cannot determine", 
    "unable to determine", "insufficient information"
]

# Public API
__all__ = [
    'MathScorer',
    'compute_score',  # Legacy compatibility
    'extract_answer',  # Legacy compatibility
]

class MathScorer:
    """Mathematical reasoning response scorer with comprehensive evaluation capabilities."""
    
    def __init__(self, debug_probability: float = 0.0):
        """Initialize the scorer.
        
        Args:
            debug_probability: Probability of printing debug information (0.0 to 1.0)
        """
        self.debug_probability = debug_probability
        
    def _should_debug(self) -> bool:
        """Determine if debug output should be printed."""
        return random.random() < self.debug_probability
    
    def extract_answer(self, passage: str) -> Optional[str]:
        """Extract the answer from a passage, preferring \\boxed{} notation."""
        if "\\boxed" in passage:
            return self._extract_boxed_answer(passage)
        return None
    
    def _extract_boxed_answer(self, solution_str: str) -> Optional[str]:
        """Extract content from the last \\boxed{} notation in the string."""
        idx = solution_str.rfind("\\boxed")
        if idx < 0:
            return None
        
        if solution_str[idx:idx+7] == "\\boxed{":
            # Handle standard \boxed{} notation
            i = idx + 7
            num_braces_open = 1
            right_brace_idx = None
            
            while i < len(solution_str):
                if solution_str[i] == "{":
                    num_braces_open += 1
                elif solution_str[i] == "}":
                    num_braces_open -= 1
                    if num_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1
            
            if right_brace_idx is not None:
                return solution_str[idx+7:right_brace_idx].strip()
                
        elif solution_str[idx:idx+7] == "\\boxed ":
            # Handle \boxed space notation
            content = solution_str[idx+7:].split("$")[0].strip()
            return content
        
        return None
    
    def extract_solution(self, solution_str: str) -> Optional[str]:
        """Extract the final answer from a solution string using priority rules.
        
        Priority order:
        1. \\boxed{} notation (last occurrence)
        2. <answer></answer> tags (last occurrence)
        3. "Final answer:" or "Answer:" patterns
        
        Args:
            solution_str: Raw solution string
            
        Returns:
            The extracted answer or None if not found
        """
        # Check for boxed notation first
        boxed_answer = self._extract_boxed_answer(solution_str)
        if boxed_answer:
            return boxed_answer
        
        # Check for answer tags
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
        if matches:
            answer_content = matches[-1].group(1).strip()
            
            # Try to extract structured answer patterns
            pattern_match = re.search(
                r'(?:the\s+)?(?:answer|result|value|solution)(?:\s+is|:|=)\s*(.*?)(?:\s*$|\.)', 
                answer_content, re.IGNORECASE
            )
            if pattern_match:
                return pattern_match.group(1).strip()
            
            # Check for "Final answer:" pattern
            final_match = re.search(r'(?:Final answer:|Answer:)\s*(.*?)(?:\s*$|\.)', answer_content)
            if final_match:
                return final_match.group(1).strip()
                
            return answer_content
        
        # Check for answer patterns in the full text
        pattern_matches = re.search(r'(?:Final answer:|Answer:|The answer is:?)\s*(.*?)(?:\n|$)', solution_str)
        if pattern_matches:
            return pattern_matches.group(1).strip()
        
        return None
    
    def validate_response_structure(self, response: str, do_print: bool = False, augmented: bool = False) -> bool:
        """Performs comprehensive validation of response structure.
        
        Args:
            response: Processed response string from the model
            do_print: Whether to print detailed validation information
            augmented: Whether this is an augmented response (partial format)
            
        Returns:
            Boolean indicating whether all formatting requirements are met
        """
        if do_print:
            print("\n[Structure Validation]")
        validation_passed = True

        if augmented:
            tags = {
                'think_end': ('</think>', 1),
                'answer_start': ('<answer>', 1),
                'answer_end': ('</answer>', 1)
            }
        else:
            tags = {
                'think_start': ('<think>', 1),
                'think_end': ('</think>', 1),
                'answer_start': ('<answer>', 1),
                'answer_end': ('</answer>', 1)
            }

        positions = {}
        for tag_name, (tag_str, expected_count) in tags.items():
            count = response.count(tag_str)
            positions[tag_name] = pos = response.find(tag_str)
            
            if do_print:
                print(f"\n  {tag_str}: count={count}, position={pos}")
            
            if count != expected_count:
                if do_print:
                    print(f"\n  [Error] {tag_str} appears {count} times (expected {expected_count})")
                validation_passed = False

        # Verify tag order
        if augmented:
            if (positions['think_end'] > positions['answer_start'] or
                positions['answer_start'] > positions['answer_end']):
                if do_print:
                    print("\n  [Error] Incorrect tag order: Expected </think><answer>...</answer>")
                validation_passed = False
            else:
                if do_print:
                    print("\n  Tag sequence validation passed")
        else:
            if (positions['think_start'] > positions['think_end'] or
                positions['think_end'] > positions['answer_start'] or
                positions['answer_start'] > positions['answer_end']):
                if do_print:
                    print("\n  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
                validation_passed = False
            else:
                if do_print:
                    print("\n  Tag sequence validation passed")
            
        # Check for boxed notation inside answer tags
        answer_content = None
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1)
            if '\\boxed{' not in answer_content:
                if do_print:
                    print("\n  [Error] Missing \\boxed{} notation inside <answer> tags")
                validation_passed = False
            else:
                if do_print:
                    print("\n  \\boxed{} notation found inside <answer> tags")

        return validation_passed
    
    def validate_response_structure_granular(self, response: str, do_print: bool = False) -> float:
        """Performs comprehensive validation of response structure with granular scoring.
        
        Args:
            response: Processed response string from the model
            do_print: Whether to print detailed validation information
            
        Returns:
            Float between 0.0 and 1.0 representing format quality
        """
        if do_print:
            print("\n[Structure Validation]")
        
        # Initialize with maximum score
        format_score = 1.0
        
        # Check required tags
        tags = {
            'think_start': ('<think>', 0.25),  # Each tag is worth 25% of format score
            'think_end': ('</think>', 0.25),
            'answer_start': ('<answer>', 0.25),
            'answer_end': ('</answer>', 0.25)
        }

        # Track positions for order checking
        positions = {}
        missing_tags = []
        
        # Check tag presence
        for tag_name, (tag_str, weight) in tags.items():
            count = response.count(tag_str)
            positions[tag_name] = pos = response.find(tag_str)
            
            if do_print:
                print(f"\n  {tag_str}: count={count}, position={pos}")
            
            # Deduct points for missing or duplicate tags
            if count != 1:
                format_score -= weight
                missing_tags.append(tag_name)
                if do_print:
                    print(f"\n  [Error] {tag_str} appears {count} times (expected 1)")

        # Check for boxed notation inside answer tags
        has_boxed = False
        answer_content = None
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        
        if answer_match:
            answer_content = answer_match.group(1)
            if '\\boxed{' in answer_content:
                has_boxed = True
                if do_print:
                    print("\n  \\boxed{} notation found inside <answer> tags")
            else:
                # Deduct for missing boxed notation
                format_score -= 0.2  # 20% penalty for missing boxed notation
                if do_print:
                    print("\n  [Error] Missing \\boxed{} notation inside <answer> tags")
        
        # Check tag ordering only if all tags are present
        if len(missing_tags) == 0:
            # Verify correct tag order
            if (positions['think_start'] > positions['think_end'] or
                positions['think_end'] > positions['answer_start'] or
                positions['answer_start'] > positions['answer_end']):
                
                # Deduct for incorrect tag order
                format_score -= 0.4  # 40% penalty for incorrect tag order
                if do_print:
                    print("\n  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
            else:
                if do_print:
                    print("\n  Tag sequence validation passed")
        
        # Ensure score stays in valid range
        format_score = max(0.0, min(1.0, format_score))
        
        if do_print:
            print(f"\n  Final format score: {format_score:.2f}")
        
        return format_score
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        if not answer:
            return ""
        
        answer = answer.strip()
        
        # Remove \text{} wrapping
        text_match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if text_match:
            answer = text_match.group("text").strip()
        
        # Basic cleanup
        answer = answer.replace("\\%", "%").replace("\\$", "$")
        answer = answer.replace("$", "").replace("%", "")
        answer = answer.replace(" or ", " , ").replace(" and ", " , ")
        
        # Handle large numbers
        answer = answer.replace("million", "*10^6")
        answer = answer.replace("billion", "*10^9")
        answer = answer.replace("trillion", "*10^12")
        
        # Remove units
        units = ["degree", "cm", "centimeter", "meter", "mile", "second", "minute", 
                "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard"]
        for unit in units:
            answer = re.sub(f"{unit}(es)?(s)?( *\\^[0-9]+)?", "", answer)
        
        answer = re.sub(r"\^ *\\circ", "", answer)
        
        # Remove braces
        if answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1]
        
        # Parse LaTeX if available
        if HAS_LATEX_PARSER and "\\" in answer:
            try:
                answer = self._parse_latex(answer)
            except Exception:
                pass
        
        # Final cleanup
        answer = re.sub(r"- *", "-", answer)
        answer = answer.replace(" ", "").replace("{", "").replace("}", "")
        answer = answer.lower()
        
        return answer
    
    def _parse_latex(self, expr: str) -> str:
        """Parse LaTeX expression to plain text."""
        if not HAS_LATEX_PARSER:
            return expr
        
        expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
        expr = expr.replace("\\frac", " \\frac")
        
        try:
            parsed = latex2text.LatexNodes2Text().latex_to_text(expr)
            # Replace mathematical symbols
            replacements = {
                "√": "sqrt", "π": "pi", "∞": "inf", "∪": "U", "·": "*", "×": "*"
            }
            for old, new in replacements.items():
                parsed = parsed.replace(old, new)
            return parsed.strip()
        except Exception:
            return expr
    
    def _grade_answer_symbolic(self, given: str, ground_truth: str) -> bool:
        """Grade answers using symbolic computation."""
        if not HAS_SYMPY:
            return False
        
        try:
            given_norm = self._normalize_answer(given)
            truth_norm = self._normalize_answer(ground_truth)
            
            if given_norm == truth_norm:
                return True
            
            # Try symbolic comparison
            expr = f"({truth_norm})-({given_norm})"
            if self._should_allow_symbolic_eval(expr):
                sympy_diff = sympy_parser.parse_expr(
                    expr.replace("^", "**"),
                    transformations=(
                        sympy_parser.standard_transformations + 
                        (sympy_parser.implicit_multiplication_application,)
                    )
                )
                return sympy.simplify(sympy_diff) == 0
        except Exception:
            pass
        
        return False
    
    def _should_allow_symbolic_eval(self, expr: str) -> bool:
        """Check if expression is safe for symbolic evaluation."""
        # Count unknown variables
        expr_clean = expr.replace("sqrt", "").replace("frac", "")
        letters = set(c for c in expr_clean if c.isalpha())
        if len(letters) > 2:
            return False
        
        # Check for problematic patterns
        for bad_string in BAD_SUBSTRINGS:
            if bad_string in expr:
                return False
        
        for bad_regex in BAD_REGEXES:
            if re.search(bad_regex, expr):
                return False
        
        return True
    
    def grade_answer(self, given_answer: str, ground_truth: Union[str, List[str]]) -> bool:
        """Grade a given answer against ground truth(s).
        
        Args:
            given_answer: The answer to grade
            ground_truth: Ground truth answer(s)
            
        Returns:
            True if the answer is correct, False otherwise
        """
        if not given_answer:
            return False
        
        # Ensure ground_truth is a list
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Process ground truth answers
        processed_truths = []
        for truth in ground_truth:
            truth = str(truth)
            if "\\boxed" in truth:
                extracted = self.extract_answer(truth)
                if extracted:
                    processed_truths.append(extracted)
            else:
                processed_truths.append(truth)
        
        # Check against all ground truths
        for truth in processed_truths:
            # Try string normalization first
            if self._normalize_answer(given_answer) == self._normalize_answer(truth):
                return True
            
            # Try symbolic comparison
            if self._grade_answer_symbolic(given_answer, truth):
                return True
        
        return False
    
    def extract_model_response(self, solution_str: str) -> str:
        """Extract the model's response from the full solution string."""
        if "<|im_start|>assistant" in solution_str:
            return solution_str.split("<|im_start|>assistant", 1)[1]
        elif "Assistant:" in solution_str:
            return solution_str.split("Assistant:", 1)[1]
        else:
            return solution_str

    def extract_model_response_aug(self, solution_str: str, extra_info: str = None) -> tuple[str, bool]:
        """Extract model response and detect if it's augmented (partial format).
        
        Args:
            solution_str: Raw model response string
            extra_info: Additional information that might indicate augmentation
            
        Returns:
            Tuple of (processed_response, is_augmented)
        """
        # Extract the response part first
        if "<|im_start|>assistant" in solution_str:
            processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        elif "Assistant:" in solution_str:
            processed_str = solution_str.split("Assistant:", 1)[1]
        else:
            processed_str = solution_str
        
        # Detect if response is augmented (starts with partial format)
        augmented = False
        
        # Check for augmented patterns in extra_info
        if extra_info:
            if "Assistant: <think>" in extra_info or "</think>" in extra_info:
                augmented = True
        
        # Check for augmented patterns in the processed response
        # Augmented responses typically start with </think> instead of <think>
        processed_str_stripped = processed_str.strip()
        if (processed_str_stripped.startswith("</think>") or 
            "Assistant: <think>" in solution_str or
            (not "<think>" in processed_str_stripped and "</think>" in processed_str_stripped)):
            augmented = True
            
        return processed_str, augmented
    
    def compute_score(self, 
                     solution_str: str, 
                     ground_truth: Union[str, List[str]],
                     scoring_mode: str = "standard",
                     extra_info: str = None) -> float:
        """Compute a comprehensive score for a model response.
        
        Args:
            solution_str: Raw model response string
            ground_truth: Ground truth answer(s)
            scoring_mode: Scoring mode - "standard", "hierarchical", "granular", or "hierarchical_aug"
            extra_info: Additional information for augmented response detection
            
        Returns:
            Score based on the selected scoring mode
        """
        do_print = self._should_debug()
        
        if do_print:
            print("\n" + "="*80)
            print(" SCORING EVALUATION ".center(80, '='))
        
        # Handle augmented responses
        if scoring_mode == "hierarchical_aug":
            response, augmented = self.extract_model_response_aug(solution_str, extra_info)
            if do_print:
                print(f"[Augmented Response]: {augmented}")
        else:
            response = self.extract_model_response(solution_str)
            augmented = False
        
        # Extract answer
        answer_text = self.extract_solution(response)
        
        if do_print:
            print(f"\n[Extracted Answer]: {answer_text}")
        
        # Check for uncertainty expressions
        if answer_text:
            answer_lower = answer_text.lower()
            if any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES):
                if do_print:
                    print("[Result]: UNCERTAINTY EXPRESSED")
                return 0.0
        
        # Validate format and grade answer
        if scoring_mode in ["standard", "granular"]:
            # Use granular validation for standard and granular modes
            format_score = self.validate_response_structure_granular(response, do_print=do_print)
            format_correct = (format_score >= 0.7)  # Convert to boolean for standard mode
        else:
            # Use boolean validation for hierarchical modes
            format_correct = self.validate_response_structure(response, do_print=do_print, augmented=augmented)
            format_score = 1.0 if format_correct else 0.0  # Convert to score for display
        
        answer_correct = self.grade_answer(answer_text, ground_truth)
        
        if do_print:
            print(f"[Format Score]: {format_score:.2f}")
            print(f"[Answer Correct]: {answer_correct}")
        
        # Calculate final score based on mode - strictly following original logic
        if scoring_mode == "standard":
            # format and answer both 0 or 1
            score = (1.0 if format_correct else 0.0) + (1.0 if answer_correct else 0.0)
        elif scoring_mode in ["hierarchical", "hierarchical_aug"]: 
            # exact boolean conditions
            if not answer_correct:
                score = -1.0  # wrong_answer_penalty
            elif answer_correct and not format_correct:
                score = 1.0   # partial_reward for answer correct but format incorrect
            else:  # answer_correct and format_correct
                score = 2.0   # full_reward
        elif scoring_mode == "granular":
            # reward partially if format is also correct
            if answer_correct:
                score = 0.2 + (0.8 * format_score)
            else:
                score = -3.0 + (1.5 * format_score)
        else:
            raise ValueError(f"Unknown scoring mode: {scoring_mode}")
        
        if do_print:
            print(f"[Final Score]: {score:.2f}")
            print("="*80)
        
        return score

# Legacy function for backward compatibility
def compute_score(solution_str: str, ground_truth: Union[str, List[str]], **kwargs) -> float:
    """Legacy function for backward compatibility."""
    scorer = MathScorer(debug_probability=kwargs.get('debug_probability', 0.0))
    return scorer.compute_score(solution_str, ground_truth, 
                               scoring_mode=kwargs.get('scoring_mode', 'standard'))

def extract_answer(passage: str) -> str:
    """Legacy function - use MathScorer.extract_answer instead."""
    scorer = MathScorer()
    return scorer.extract_answer(passage)

def extract_boxed_answer(solution_str: str) -> str:
    """Extract the content from the last \\boxed{} notation in the string.
    
    Args:
        solution_str: Solution string that may contain \\boxed{} notation
        
    Returns:
        Content of the boxed expression, or None if not found
    """
    # Find the last occurrence of \boxed
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return None
    
    # Check if it's \boxed{} or \boxed 
    if solution_str[idx:idx+7] == "\\boxed{":
        # Handle standard \boxed{} notation
        i = idx + 7  # Start after \boxed{
        num_braces_open = 1
        right_brace_idx = None
        
        while i < len(solution_str):
            if solution_str[i] == "{":
                num_braces_open += 1
            elif solution_str[i] == "}":
                num_braces_open -= 1
                if num_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is None:
            return None
        
        return solution_str[idx+7:right_brace_idx].strip()
    
    elif solution_str[idx:idx+7] == "\\boxed ":
        # Handle \boxed space notation
        content = solution_str[idx+7:].split("$")[0].strip()
        return content
    
    return None

def extract_solution(solution_str: str) -> str:
    """Extract the final answer from a solution string based on priority rules.
    
    Priority order:
    1. Extract from \\boxed{} notation (last one if multiple exist)
    2. Extract from <answer></answer> tags (last one if multiple exist)
    3. Extract from "Final answer:" or "Answer:" patterns
    
    Args:
        solution_str: Raw solution string 
        
    Returns:
        extracted_answer: The final answer extracted from the solution string
    """
    # Check for boxed notation first
    # boxed_answer = remove_boxed(last_boxed_only_string(solution_str))
    boxed_answer = extract_boxed_answer(solution_str)
    if boxed_answer:
        return boxed_answer
    
    # Then check for answer tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        # Get the last match if multiple exist
        answer_content = matches[-1].group(1).strip()
        # Try to extract answer if it's in a pattern like "The answer is X" or "Answer: X"
        pattern_match = re.search(r'(?:the\s+)?(?:answer|result|value|solution)(?:\s+is|:|=)\s*(.*?)(?:\s*$|\.)', answer_content, re.IGNORECASE)
        if pattern_match:
            return pattern_match.group(1).strip()
            
        # Check for "Final answer:" or "Answer:" pattern within the answer tags
        final_match = re.search(r'(?:Final answer:|Answer:)\s*(.*?)(?:\s*$|\.)', answer_content)
        if final_match:
            return final_match.group(1).strip()
        
        return answer_content
    
    # Finally check for answer patterns
    pattern_matches = re.search(r'(?:Final answer:|Answer:|The answer is:?)\s*(.*?)(?:\n|$)', solution_str)
    if pattern_matches:
        return pattern_matches.group(1).strip()
    
    # If nothing matches
    return None




# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


# def _parse_latex(expr: str) -> str:
#     """Attempts to parse latex to an expression sympy can read."""
#     expr = expr.replace("\\tfrac", "\\frac")
#     expr = expr.replace("\\dfrac", "\\frac")
#     expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
#     expr = latex2text.LatexNodes2Text().latex_to_text(expr)

#     # Replace the specific characters that this parser uses.
#     expr = expr.replace("√", "sqrt")
#     expr = expr.replace("π", "pi")
#     expr = expr.replace("∞", "inf")
#     expr = expr.replace("∪", "U")
#     expr = expr.replace("·", "*")
#     expr = expr.replace("×", "*")

#     return expr.strip()

def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    # expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    
    try:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    except Exception as e:
        print(f"LaTeX parsing error (attempt 1): {e}")
        
        # Try balancing braces by counting, but preserve \boxed{}
        balanced_expr = expr
        try:
            # Split by \boxed to preserve those sections
            parts = []
            boxed_parts = []
            
            # Split the expression by \boxed
            if "\\boxed" in balanced_expr:
                segments = balanced_expr.split("\\boxed")
                parts.append(segments[0])
                
                for i in range(1, len(segments)):
                    # Find the matching closing brace for this boxed section
                    boxed_content = segments[i]
                    if boxed_content.startswith("{"):
                        brace_count = 1
                        end_idx = 0
                        for j, char in enumerate(boxed_content[1:], 1):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                            if brace_count == 0:
                                end_idx = j
                                break
                        
                        # If we found a complete boxed expression
                        if end_idx > 0:
                            boxed_parts.append("\\boxed" + boxed_content[:end_idx+1])
                            parts.append(boxed_content[end_idx+1:])
                        else:
                            # If closing brace not found, keep as is
                            boxed_parts.append("\\boxed" + boxed_content)
                            parts.append("")
                    else:
                        # If no opening brace, keep as is
                        parts.append(segments[i])
                        boxed_parts.append("")
            else:
                parts = [balanced_expr]
                
            # Process non-boxed parts only
            processed_parts = []
            for part in parts:
                # Count and balance braces in this part
                open_count = part.count('{')
                close_count = part.count('}')
                
                # Balance the braces
                if close_count > open_count:
                    part = '{' * (close_count - open_count) + part
                elif open_count > close_count:
                    part = part + '}' * (open_count - close_count)
                    
                processed_parts.append(part)
            
            # Reconstruct the expression with boxed parts preserved
            if boxed_parts:
                final_expr = processed_parts[0]
                for i in range(len(boxed_parts)):
                    if boxed_parts[i]:
                        final_expr += boxed_parts[i]
                    if i+1 < len(processed_parts):
                        final_expr += processed_parts[i+1]
                balanced_expr = final_expr
            else:
                balanced_expr = processed_parts[0]
                
            # Try parsing with balanced braces
            parsed_expr = latex2text.LatexNodes2Text().latex_to_text(balanced_expr)
        except Exception as e:
            print(f"LaTeX parsing error (final attempt): {e}")
            # If all parsing attempts fail, just use the original expression without parsing
            parsed_expr = expr

    # Replace the specific characters that this parser uses.
    # expr = expr.replace("√", "sqrt")
    # expr = expr.replace("π", "pi")
    # expr = expr.replace("∞", "inf")
    # expr = expr.replace("∪", "U")
    # expr = expr.replace("·", "*")
    # expr = expr.replace("×", "*")
    
    parsed_expr = parsed_expr.replace("√", "sqrt")
    parsed_expr = parsed_expr.replace("π", "pi")
    parsed_expr = parsed_expr.replace("∞", "inf")
    parsed_expr = parsed_expr.replace("∪", "U")
    parsed_expr = parsed_expr.replace("·", "*")
    parsed_expr = parsed_expr.replace("×", "*")

    return parsed_expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_model_response(solution_str: str) -> str:
    if "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    else:
        print("[Error] Failed to locate model response header, returning full string")
        return solution_str
    return processed_str

def compute_score(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 format_reward: float = 1.0,
                 answer_reward: float = 1.0):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth solution text
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    do_print = random.randint(1, 32) == 1
    if do_print:
        print("\n" + "="*80)
        print(" Processing New Sample ".center(80, '='))
        
    # Extract model response
    response = extract_model_response(solution_str)

    # Extract model answer
    answer_text= extract_solution(response)
    if do_print:
        print(f"\n[Model Response]\n{response}")
        print(f"\n[Extracted Response]\n{answer_text}")

    # Validate response structure
    format_correct = validate_response_structure(response, do_print)
    format_score = format_reward if format_correct else 0
    if do_print:
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"\n  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    ground_truth_list = []
    
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
        
    if answer_text:
        # Convert single ground truth to list for uniform processing
        for truth in ground_truth:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    ground_truth_list.append(processed_truth)
            else:
                ground_truth_list.append(truth)
            
        # Check against all possible ground truths
        for gt in ground_truth_list:
            if grade_answer_mathd(answer_text, gt) or grade_answer_sympy(answer_text, gt):
                answer_score = answer_reward
                if do_print:
                    print(f"\n[Content Validation] CORRECT: {answer_text} matches acceptable answer: {gt}")
                break
        
        if do_print and answer_score == 0:
            gt_str = ", ".join([f"'{gt}'" for gt in ground_truth_list])
            print(f"\n[Content Validation] INCORRECT: '{answer_text}' doesn't match any acceptable answers: {gt_str}")
    if do_print:
        print(f"\n  Answer score: {answer_score}")

    total_score = format_score + answer_score
    if do_print:
        print(f"Final Score ".center(80, '-'))
        print(f"\n  Format: {format_score} | Answer: {answer_score} | Total: {total_score}")
        print("\n" + "="*80)

    return total_score

# Deprecated - use MathScorer.compute_score with scoring_mode="standard" instead
def compute_score_o(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 format_reward: float = 1.0,
                 answer_reward: float = 1.0):
    """Deprecated - use MathScorer.compute_score instead."""
    scorer = MathScorer(debug_probability=0.03)
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="standard")

# Deprecated - use MathScorer.compute_score with scoring_mode="hierarchical" instead
def compute_score_o2(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 full_reward: float = 2.0,
                 partial_reward: float = 1.0,
                 wrong_answer_penalty: float = -1.0):
    """Deprecated - use MathScorer.compute_score instead."""
    scorer = MathScorer(debug_probability=0.03)
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical")

def compute_score_o2_aug(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 full_reward: float = 2.0,
                 partial_reward: float = 1.0,
                 wrong_answer_penalty: float = -1.0,
                 extra_info=None):
    """Deprecated - use MathScorer.compute_score with scoring_mode="hierarchical_aug" instead.
    
    Computes score for model response with augmented format support.
    """
    scorer = MathScorer(debug_probability=0.03)
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical_aug", extra_info=extra_info)

# Deprecated - use MathScorer.compute_score with scoring_mode="hierarchical" instead
def compute_score_n5(solution_str: str, 
                 ground_truth: Union[str, List[str]]):
    """Deprecated - use MathScorer.compute_score instead."""
    scorer = MathScorer(debug_probability=1.0)  # Always debug for this legacy function
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical")

# Deprecated - use MathScorer.compute_score with scoring_mode="granular" instead
def compute_score_o_granular(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 format_reward: float = 1.0,
                 answer_reward: float = 1.0):
    """Deprecated - use MathScorer.compute_score instead."""
    scorer = MathScorer(debug_probability=1.0)  # Always debug for this legacy function
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="granular")

# Deprecated - use MathScorer.compute_score with scoring_mode="granular" instead
def compute_score_n5_granular(solution_str: str, 
                     ground_truth: Union[str, List[str]]):
    """Deprecated - use MathScorer.compute_score instead."""
    scorer = MathScorer(debug_probability=1.0)  # Always debug for this legacy function
    return scorer.compute_score(solution_str, ground_truth, scoring_mode="granular")

# Deprecated - use MathScorer.grade_answer for answer-only evaluation
def compute_val_score(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 answer_reward: float = 1.0):
    """Deprecated - use MathScorer for evaluation."""
    scorer = MathScorer(debug_probability=0.06)
    response = scorer.extract_model_response(solution_str)
    answer_text = scorer.extract_solution(response)
    return answer_reward if scorer.grade_answer(answer_text, ground_truth) else 0.0

if __name__ == "__main__":
    """
    Demo and testing of the MathScorer functionality.
    """
    print("="*60)
    print("SPARKLE MATH SCORER DEMO")
    print("="*60)
    
    # Initialize scorer with debug output
    scorer = MathScorer(debug_probability=1.0)
    
    # Test answer extraction
    print("\n1. ANSWER EXTRACTION TESTS")
    print("-" * 30)
    
    test_cases = [
        ("Boxed notation", "Some workings...\\boxed{42}"),
        ("Answer tags", "<answer>The answer is 42</answer>"),
        ("Final answer pattern", "Final answer: 42"),
        ("Multiple formats", "<answer>First</answer> \\boxed{Second}"),
        ("No recognized format", "Just some text")
    ]
    
    for name, test_str in test_cases:
        result = scorer.extract_solution(test_str)
        print(f"{name}: {result}")
    
    # Test complete scoring
    print("\n\n2. COMPLETE SCORING TEST")
    print("-" * 30)
    
    sample_response = """<|im_start|>assistant
<think>
To find the largest angle, I need to use the fact that the sum of interior angles 
in a pentagon is (5-2) × 180 = 540 degrees.

Setting up the equation:
(x+1) + 2x + 3x + 4x + (5x-1) = 540
15x = 540
x = 36

The angles are: 37°, 72°, 108°, 144°, 179°
</think>
<answer>
The measure of the largest angle is \\boxed{179} degrees.
</answer>"""
    
    ground_truth = "179"
    
    print("Testing different scoring modes:")
    for mode in ["standard", "hierarchical", "granular"]:
        score = scorer.compute_score(sample_response, ground_truth, scoring_mode=mode)
        print(f"\n{mode.upper()} mode score: {score}")
    
    # Test augmented response
    print("\n\n3. AUGMENTED RESPONSE TEST")
    print("-" * 30)
    
    augmented_response = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \boxed{} notation. User: A convex pentagon has interior angles with measures $x+1$, $2x$, $3x$, $4x$, and $5x-1$ degrees.  What is the measure of the largest angle? Assistant: <think>
To find the measure of the largest angle in the convex pentagon, <|im_end|>
<|im_start|>assistant
we start by using the fact that the sum of the interior angles of a pentagon is \(540\) degrees. The given angles are \(x+1\), \(2x\), \(3x\), \(4x\), and \(5x-1\). We can set up the equation for the sum of these angles:

\[
(x+1) + 2x + 3x + 4x + (5x-1) = 540
\]

Next, we combine like terms:

\[
x + 1 + 2x + 3x + 4x + 5x - 1 = 540
\]

\[
15x = 540
\]

Solving for \(x\), we divide both sides by 15:

\[
x = \frac{540}{15} = 36
\]

Now that we have the value of \(x\), we can find the measure of each angle:
 
\[
x + 1 = 36 + 1 = 37
\]
\[
2x = 2 \cdot 36 = 72
\]
\[
3x = 3 \cdot 36 = 108
\]
\[
4x = 4 \cdot 36 = 144
\]
\[
5x - 1 = 5 \cdot 36 - 1 = 179
\]

The largest angle is \(179\) degrees.
</think>
<answer>
The measure of the largest angle is \(\\boxed{179}\) degrees.
</answer><|im_end|>"""
    
    print("Testing augmented response (partial format):")
    score = scorer.compute_score(augmented_response, ground_truth, scoring_mode="hierarchical_aug")
    print(f"Hierarchical_aug mode score: {score}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)