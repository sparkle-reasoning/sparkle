import re

import importlib.util
import sys

# Check if pylatexenc is installed
if importlib.util.find_spec("pylatexenc") is not None:
    import pylatexenc.latexwalker
    
    # Look for the correct parse function
    # The error shows that '_parse_latex_node' doesn't exist
    # Let's find the right method to patch
    
    # Option 1: Try to patch the get_latex_nodes method instead
    original_get_latex_nodes = pylatexenc.latexwalker.LatexWalker.get_latex_nodes
    
    def patched_get_latex_nodes(self, *args, **kwargs):
        try:
            return original_get_latex_nodes(self, *args, **kwargs)
        except Exception as e:
            if "Unexpected mismatching closing brace" in str(e):
                # Log the error
                print(f"Ignoring LaTeX parsing error: {e}")
                # Return an empty node structure
                return []
            # Re-raise other types of errors
            raise
    
    # Replace the original function with our patched version
    pylatexenc.latexwalker.LatexWalker.get_latex_nodes = patched_get_latex_nodes
    
    print("Successfully patched pylatexenc to ignore mismatched brace errors")

from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Dict, Tuple, Optional, Union, List
import random
from collections import Counter

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

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


def validate_response_structure(processed_str: str, do_print=False) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"\n  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"\n  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
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
    answer_match = re.search(r'<answer>(.*?)</answer>', processed_str, re.DOTALL)
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

def validate_response_structure_granular(processed_str: str, do_print=False) -> float:
    """Performs comprehensive validation of response structure with granular scoring.
    
    Args:
        processed_str: Processed response string from the model
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
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
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
    answer_match = re.search(r'<answer>(.*?)</answer>', processed_str, re.DOTALL)
    
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
    # try:
    #     expr = f"({ground_truth_normalized})-({given_normalized})"
    #     if should_allow_eval(expr):
    #         sympy_diff = _sympy_parse(expr)
    #         simplified = sympy.simplify(sympy_diff)
    #         if simplified == 0:
    #             are_equal = True
    # except:
    #     pass
    # return are_equal
    
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            try:
                sympy_diff = _sympy_parse(expr)
                simplified = sympy.simplify(sympy_diff)
                if simplified == 0:
                    are_equal = True
            except Exception as e:
                # If parsing fails, try a fallback comparison approach
                # For example, direct string comparison after more aggressive normalization
                cleaned_gt = re.sub(r'[{}\\]', '', ground_truth_normalized)
                cleaned_given = re.sub(r'[{}\\]', '', given_normalized)
                are_equal = cleaned_gt == cleaned_given
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

def compute_val_score(solution_str: str, 
                 ground_truth: Union[str, List[str]],
                 answer_reward: float = 1.0):
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: ground truth solution text
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    do_print = random.randint(1, 16) == 1
    if do_print:
        print("\n" + "="*80)
        print(" Start evaluation ".center(80, '='))

    # # Extract model response
    # response = extract_model_response(solution_str)
    
    answer_text= extract_solution(solution_str)
    if do_print:
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Extracted Response]\n{answer_text}")
    # Validate answer content
    answer_score = 0
    ground_truth_list = []
    
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, int):
        ground_truth = [str(ground_truth)]
    elif isinstance(ground_truth, float):
        ground_truth = [str(ground_truth)]
        
    if answer_text:
        for truth in ground_truth:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    ground_truth_list.append(processed_truth)
            else:
                ground_truth_list.append(truth)
            
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

    return answer_score

def doc_to_text(doc: dict) -> str:
    instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"{instruction}\n\nUser: {question} Assistant:"
    return prompt

def doc_to_text_srl(doc: dict) -> str:
    instruction = """Please reason step by step, and put your final answer within \\boxed{}."""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"User: {question}\n{instruction}\nAssistant:"
    return prompt

def doc_to_text_plan(doc: dict) -> str:
    instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"{instruction}\n\nUser: {question}\nConsider the following planning skeleton to guide your reasoning. You may adapt or extend this outline as needed based on your analysis of the problem:\n{doc['planning']}\nAssistant:"
    return prompt

def doc_to_text_plan_srl(doc: dict) -> str:
    instruction = """Please reason step by step, and put your final answer within \\boxed{}."""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"User: {question}\nConsider the following planning skeleton to guide your reasoning. You may adapt or extend this outline as needed based on your analysis of the problem:\n{doc['planning']}\n{instruction}\nAssistant:"
    return prompt

def doc_to_text_knowledge(doc: dict) -> str:
    instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"{instruction}\n\nUser: {question}\nThe following is a reference knowledge base for concepts you might not be familiar with. If you already understand these concepts, you don't need to rely on this information. Otherwise, you can refer to this as needed:\n{doc['knowledge']}\nAssistant:"
    return prompt

def doc_to_text_knowledge_srl(doc: dict) -> str:
    instruction = """Please reason step by step, and put your final answer within \\boxed{}."""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"User: {question}\nThe following is a reference knowledge base for concepts you might not be familiar with. If you already understand these concepts, you don't need to rely on this information. Otherwise, you can refer to this as needed:\n{doc['knowledge']}\n{instruction}\nAssistant:"
    return prompt

def doc_to_text_subproblem(doc: dict) -> str:
    instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""
    if "augmented_problem" in doc.keys():
        question = doc['augmented_problem']
    prompt = f"{instruction}\n\nUser: {question} Assistant:"
    return prompt

def doc_to_text_distill(doc: dict) -> str:
    instruction = """Please reason step by step, and put your final answer within \\boxed{}."""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"{instruction}\n\nUser: {question} Assistant: <think>\n"
    return prompt

def doc_to_text_direct_cot(doc: dict) -> str:
    instruction = """Please reason step by step, and put your final answer within \\boxed{}."""
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"{question}\n{instruction}"
    return prompt

def doc_to_text_simple_cot(doc: dict) -> str:
    if "problem" in doc.keys():
        question = doc['problem']
    elif "question" in doc.keys():
        question = doc['question']
    prompt = f"Question:{question}\nAnswer:\nLet's think step by step."
    return prompt

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": 0, "extracted_answers": []}
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"pass@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }
    else:
        n_res_list = []
    if "answer" in doc.keys():
        ground_truth = doc['answer']
    elif "final_answer" in doc.keys():
        ground_truth = doc['final_answer']
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, int):
        ground_truth = [str(ground_truth)]
    elif isinstance(ground_truth, float):
        ground_truth = [str(ground_truth)]
    ground_truth_list = []
    answer_reward = 1.0
    
    for truth in ground_truth:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                ground_truth_list.append(processed_truth)
        else:
            ground_truth_list.append(truth)
    
    for i, result in enumerate(results, start=1):
        answer_score = 0
        answer_text= extract_solution(result)
            
        if answer_text:
            metrics["extracted_answers"].append(answer_text)
                
            for gt in ground_truth_list:
                if grade_answer_mathd(answer_text, gt) or grade_answer_sympy(answer_text, gt):
                    answer_score = answer_reward
                    break
        if i == 1:
            metrics["exact_match"] = answer_score
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(answer_score)
        elif i > 1 and "exact_matches" in metrics:
            metrics["exact_matches"].append(answer_score)
            
        if i in n_res_list:
            metrics[f"pass@{i}"] = int(any(score == 1 for score in metrics["exact_matches"][:i]))
            if metrics["extracted_answers"]:
                most_common_answer = Counter(metrics["extracted_answers"][:i]).most_common(1)[0][0]
                majority_correct = any(
                    grade_answer_mathd(most_common_answer, gt) or 
                    grade_answer_sympy(most_common_answer, gt) 
                    for gt in ground_truth_list
                )
                metrics[f"maj@{i}"] = int(majority_correct)
        
    return metrics