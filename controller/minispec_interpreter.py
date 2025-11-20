import ast
import re
import time
import queue
from enum import Enum
from threading import Thread, Lock
from typing import List, Tuple, Union, Optional, Any
from queue import Queue

from openai import ChatCompletion, Stream

from .skillset import SkillSet
from .utils.general_utils import split_args, print_t

# ------------------------------------------------------------------------------
# Type Definitions & Utils
# ------------------------------------------------------------------------------

MiniSpecValueType = Union[int, float, bool, str, list, None]

def print_debug(*args):
    """Wrapper for debug prints. Comment out print to silence."""
    print(*args)

def evaluate_value(value: str) -> MiniSpecValueType:
    """Safely evaluates a string literal into a Python type."""
    value = str(value).strip()
    if not value or value == 'None':
        return None
    if value == 'True':
        return True
    if value == 'False':
        return False
    
    # Try numeric conversion (handles negatives and decimals correctly)
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # Remove quotes if present
    return value.strip('\'"')

# ------------------------------------------------------------------------------
# Core Data Structures
# ------------------------------------------------------------------------------

class MiniSpecReturnValue:
    """Wrapper for return value + control flow flags (replan/wait)."""
    def __init__(self, value: MiniSpecValueType, replan: bool, wait_user_answer: bool = False):
        self.value = value
        self.replan = replan
        self.wait_user_answer = wait_user_answer

    @staticmethod
    def from_tuple(t: Union[Tuple[Any, bool], Tuple[Any, bool, bool]]):
        if len(t) == 3:
            return MiniSpecReturnValue(t[0], t[1], t[2])
        return MiniSpecReturnValue(t[0], t[1], False)

    @staticmethod
    def default():
        return MiniSpecReturnValue(None, False, False)

    def __repr__(self) -> str:
        return f'Val(value={self.value}, replan={self.replan})'

class ParsingState(Enum):
    CODE           = 0
    CONDITION      = 1
    LOOP_COUNT     = 2
    SUB_STATEMENTS = 3

# ------------------------------------------------------------------------------
# Statement
# ------------------------------------------------------------------------------

class Statement:
    # Static context shared across statements (injected by Interpreter)
    execution_queue: Queue['Statement'] = None   
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    interpreter: 'MiniSpecInterpreter' = None

    def __init__(self, env: dict) -> None:
        self.env = env
        self.code_buffer = ''
        self.action = ''
        self.parsing_state = ParsingState.CODE
        
        # Control flow storage
        self.condition: str | None = None
        self.loop_count: int | None = None
        self.sub_statements: MiniSpecProgram | None = None
        
        # Parsing state flags
        self.depth_paren = 0
        self.in_single_quote = False
        self.in_double_quote = False
        self.executable = False
        self.ret = False

    def parse(self, code: str, exec_mode: bool = False) -> bool:
        """
        Feeds characters into the state machine. 
        Returns True if the statement is complete.
        """
        for c in code:
            if self.parsing_state == ParsingState.CODE:
                if self._parse_char_code(c):
                    self._finalize(exec_mode)
                    return True
            
            elif self.parsing_state == ParsingState.CONDITION:
                self._accumulate_until_block(c, 'if')

            elif self.parsing_state == ParsingState.LOOP_COUNT:
                self._accumulate_until_block(c, 'loop')

            elif self.parsing_state == ParsingState.SUB_STATEMENTS:
                if self.sub_statements.parse([c]):
                    return True
        return False

    def _parse_char_code(self, c: str) -> bool:
        """Handle character logic for the main CODE state."""
        # Quote toggling
        if c == "'" and not self.in_double_quote:
            self.in_single_quote = not self.in_single_quote
            self.code_buffer += c
            return False
        if c == '"' and not self.in_single_quote:
            self.in_double_quote = not self.in_double_quote
            self.code_buffer += c
            return False

        # Ignore special logic if inside quotes
        if self.in_single_quote or self.in_double_quote:
            self.code_buffer += c
            return False

        # Logic
        if c == '?' and self.depth_paren == 0:
            self.action = 'if'
            self.parsing_state = ParsingState.CONDITION
            return False
        
        if c == ';' and self.depth_paren == 0:
            self.action = self.code_buffer.strip()
            return True
            
        if c == '(':
            self.depth_paren += 1
        elif c == ')':
            self.depth_paren -= 1
            self.code_buffer += c
            # If parens balanced back to 0, statement might be done
            if self.depth_paren == 0:
                self.action = self.code_buffer.strip()
                return True
            return False
            
        if c == '}' and self.depth_paren == 0 and self.code_buffer.strip():
            self.action = self.code_buffer.strip()
            return True
            
        # Check for numeric loop start (e.g. "5 { ... }")
        if c.isdigit() and not self.code_buffer.strip():
            self.action = 'loop'
            self.parsing_state = ParsingState.LOOP_COUNT
            self.code_buffer += c
            return False

        self.code_buffer += c
        return False

    def _accumulate_until_block(self, c: str, mode: str):
        """Helper for Condition and Loop headers ending in '{'."""
        if c == '{':
            content = self.code_buffer.strip()
            if mode == 'if':
                print_debug(f'SP Condition: {content}')
                self.condition = content
            elif mode == 'loop':
                print_debug(f'SP Loop: {content}')
                self.loop_count = int(content)
            
            # Prepare to parse the block body
            self._finalize(exec_mode=True) # Queue the header itself
            self.sub_statements = MiniSpecProgram(self.env)
            self.parsing_state = ParsingState.SUB_STATEMENTS
        else:
            self.code_buffer += c

    def _finalize(self, exec_mode: bool):
        """Marks statement as executable and enqueues it if needed."""
        self.executable = True
        print_debug(f'SP Action Finalized: {self.action}')
        if exec_mode and self.action:
            if Statement.interpreter:
                with Statement.interpreter.program_lock:
                    Statement.interpreter.program_count += 1
            self.execution_queue.put(self)

    # ---------------------------------------------------------------- Runtime
    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Statement eval: {self.action}')
        # Wait until parsing is fully complete for this statement
        while not self.executable:
            time.sleep(0.05)

        if self.action == 'if':
            return self._eval_if()
        if self.action == 'loop':
            return self._eval_loop()

        # Standard action execution
        self.ret = False
        action_ret_val = self.eval_expr(self.action)
        
        # Log to history
        if self.interpreter:
            self.interpreter.execution_history[1].append((self.action, action_ret_val))
            
        return action_ret_val

    def _eval_if(self) -> MiniSpecReturnValue:
        ret_val = self.eval_condition(self.condition)
        if ret_val.replan: return ret_val
        
        if ret_val.value:
            ret_val = self.sub_statements.eval()
            if ret_val.replan or self.sub_statements.ret:
                self.ret = True
            return ret_val
        return MiniSpecReturnValue.default()

    def _eval_loop(self) -> MiniSpecReturnValue:
        for _ in range(self.loop_count):
            ret_val = self.sub_statements.eval()
            if ret_val.replan or self.sub_statements.ret:
                self.ret = True
                return ret_val
        return MiniSpecReturnValue.default()

    # ---------------------------------------------------------------- Functions
    def eval_function(self, func_str: str) -> MiniSpecReturnValue:
        print_debug(f'Eval function: {func_str}')
        name, *rest = func_str.split('(', 1)
        name = name.strip()

        # Parse Arguments
        args = []
        if rest:
            raw_args = rest[0].rstrip(')')
            # Handle single list arg vs multiple args
            if raw_args.strip().startswith('[') and raw_args.strip().endswith(']'):
                args_list = [raw_args.strip()]
            else:
                args_list = split_args(raw_args)
            
            for arg in args_list:
                arg = arg.strip()
                # 1. List Literal
                if arg.startswith('['):
                    try:
                        args.append(ast.literal_eval(arg))
                        continue
                    except:
                        pass # Fallback to string
                
                # 2. Variable Reference
                if arg.startswith('_'):
                    args.append(self.get_env_value(arg))
                    continue
                
                # 3. String with Interpolation
                val = arg.strip('\'"')
                # Regex to replace _vars inside string
                val = re.sub(
                    r'\b(_[a-zA-Z0-9_]*)\b', 
                    lambda m: str(self.env.get(m.group(1), m.group(1))), 
                    val
                )
                args.append(val)

        # Built-ins
        if name in ('int', 'float', 'str'):
            types = {'int': int, 'float': float, 'str': str}
            return MiniSpecReturnValue(types[name](args[0]), False)

        # Skills
        skill = self.low_level_skillset.get_skill(name)
        if skill:
            print_debug(f'Exec LowLevel: {name} {args}')
            return MiniSpecReturnValue.from_tuple(skill.execute(args))

        skill = self.high_level_skillset.get_skill(name)
        if skill:
            print_debug(f'Exec HighLevel: {name} {args}')
            # Recursive execution for high level skills
            interp = MiniSpecProgram(env=self.env)
            interp.parse([skill.execute(args)])
            interp.finished = True
            
            val = interp.eval()
            if val.replan: return MiniSpecReturnValue(val.value, True)
            if interp.ret: return MiniSpecReturnValue(val.value, False)
            return val

        raise Exception(f'Skill {name} is not defined')

    def eval_expr(self, expr: str) -> MiniSpecReturnValue:
        expr = expr.strip()
        print_t(f'Eval expr: {expr}')

        # Return statement '->'
        if expr.startswith('->'):
            self.ret = True
            res = self.eval_expr(expr.lstrip('->'))
            # Treat specific string keywords as replan signals
            is_error_str = isinstance(res.value, str) and any(x in res.value.lower() for x in ['failed', 'error', 'replan'])
            return MiniSpecReturnValue(res.value, res.replan or is_error_str)

        # Assignment '='
        # Simple top-level split (brittle if strings contain '=', simplified for brevity)
        if '=' in expr and not expr.startswith('='):
             # A more robust parser would check for quoted '='
            parts = expr.split('=', 1)
            left, right = parts[0].strip(), parts[1].strip()
            # Simple check to avoid splitting 'a==b'
            if not left.endswith(('>', '<', '!', '=')): 
                val = self.eval_expr(right)
                self.env[left] = val.value
                return val

        # Arithmetic (Basic support)
        for op in ['+', '-', '*', '/']:
            if op in expr and '(' not in expr: # Avoid breaking function calls
                parts = expr.rsplit(op, 1) # Split from right to handle precedence simply
                if len(parts) == 2:
                    v1 = self.eval_expr(parts[0]).value
                    v2 = self.eval_expr(parts[1]).value
                    if op == '+': return MiniSpecReturnValue(v1 + v2, False)
                    if op == '-': return MiniSpecReturnValue(v1 - v2, False)
                    if op == '*': return MiniSpecReturnValue(v1 * v2, False)
                    if op == '/': return MiniSpecReturnValue(v1 / v2, False)

        # Atoms
        if expr.startswith('_'):
            return MiniSpecReturnValue(self.get_env_value(expr), False)
        if expr[0].isalpha():
            return self.eval_function(expr)
        
        return MiniSpecReturnValue(evaluate_value(expr), False)

    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        # Basic parsing for logical AND/OR
        if '&' in condition:
            parts = condition.split('&')
            return MiniSpecReturnValue(all(self.eval_condition(p).value for p in parts), False)
        if '|' in condition:
            parts = condition.split('|')
            return MiniSpecReturnValue(any(self.eval_condition(p).value for p in parts), False)

        # Comparators
        match = re.split(r'(>|<|==|!=)', condition)
        if len(match) == 3:
            op1, comp, op2 = match
            v1 = self.eval_expr(op1).value
            v2 = self.eval_expr(op2).value
            
            if comp == '>': return MiniSpecReturnValue(v1 > v2, False)
            if comp == '<': return MiniSpecReturnValue(v1 < v2, False)
            if comp == '==': return MiniSpecReturnValue(v1 == v2, False)
            if comp == '!=': return MiniSpecReturnValue(v1 != v2, False)

        return MiniSpecReturnValue(False, False)

    def get_env_value(self, var: str) -> Any:
        if var not in self.env:
            raise KeyError(f'Variable {var} is not defined')
        return self.env[var]

    def __repr__(self) -> str:
        if self.action == 'if': return f'if {self.condition} {{...}}'
        if self.action == 'loop': return f'[{self.loop_count}] {{...}}'
        return self.action

# ------------------------------------------------------------------------------
# MiniSpecProgram (Container)
# ------------------------------------------------------------------------------

class MiniSpecProgram:
    """A parsed MiniSpec program consisting of a list of Statements."""
    def __init__(self, env: Optional[dict] = None, mq: queue.Queue = None) -> None:
        self.statements: List[Statement] = []
        self.env = env if env is not None else {}
        self.mq = mq
        self.current_statement = Statement(self.env)
        self.depth = 0
        self.finished = False
        self.ret = False

    def parse(self, stream: Union[Stream, List[str]], exec: bool = False) -> bool:
        """Parses code chunks into statements."""
        for chunk in stream:
            code = chunk if isinstance(chunk, str) else chunk.choices[0].delta.content
            if not code: continue
            
            if self.mq: self.mq.put(code + '\\\\')

            for c in code:
                # Returns True if statement completed
                if self.current_statement.parse(c, exec_mode=exec):
                    if self.current_statement.action:
                        self.statements.append(self.current_statement)
                    self.current_statement = Statement(self.env)

                # Track braces for program termination
                if c == '{': self.depth += 1
                elif c == '}':
                    if self.depth == 0:
                        self.finished = True
                        return True
                    self.depth -= 1
        return False

    def eval(self) -> MiniSpecReturnValue:
        ret = MiniSpecReturnValue.default()
        for stmt in self.statements:
            ret = stmt.eval()
            if ret.replan or stmt.ret:
                self.ret = True
                return ret
        return ret

# ------------------------------------------------------------------------------
# Interpreter (Threaded Executor)
# ------------------------------------------------------------------------------

class MiniSpecInterpreter:
    """Main entry point. Manages the execution thread."""
    def __init__(self, message_queue: queue.Queue):
        self.execution_history = ["", []]
        self.message_queue = message_queue
        self.ret_queue: Queue[MiniSpecReturnValue] = Queue()
        
        # Stats
        self.program_count = 0
        self.program_lock = Lock()
        self.is_executing = False
        self.timestamp_start = None

        # Setup Static Context
        if Statement.low_level_skillset is None:
            raise Exception('Interpreter Error: Skillset not initialized')
            
        Statement.execution_queue = Queue()
        Statement.interpreter = self
        
        # Start Worker
        self.thread = Thread(target=self._executor, daemon=True)
        self.thread.start()

    def execute(self, code: Union[Stream, List[str]]) -> MiniSpecReturnValue:
        print_t('>>> Starting new Program execution')
        self.execution_history = ["", []]
        
        # Reset counters
        with self.program_lock:
            if not self.is_executing:
                self.program_count = 0
            
        program = MiniSpecProgram(mq=self.message_queue)
        program.parse(code, exec=True)
        
        # Block until result is ready
        return self.ret_queue.get()

    def _executor(self):
        """Consumer thread: pulls ready statements and executes them."""
        while True:
            try:
                # Blocking get with timeout to allow checking interrupts/shutdown
                stmt = Statement.execution_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Mark start time
            with self.program_lock:
                if not self.is_executing and self.program_count > 0:
                    self.timestamp_start = time.time()
                    self.is_executing = True
                    print_t('>>> Execution Started')

            # EXECUTE
            print_debug('Executing:', stmt)
            ret_val = stmt.eval()
            print_t('Done:', stmt)

            # Handle Completion / Early Return
            with self.program_lock:
                if stmt.ret:
                    # Clear remaining queue on return
                    with Statement.execution_queue.mutex:
                        Statement.execution_queue.queue.clear()
                    self._finish_execution(ret_val)
                    continue

                self.program_count -= 1
                if self.program_count == 0:
                    self._finish_execution(ret_val)

    def _finish_execution(self, ret_val: MiniSpecReturnValue):
        """Helper to package results and reset state."""
        if self.timestamp_start:
            print_t(f'>>> Execution time: {time.time() - self.timestamp_start:.4f}s')
        self.timestamp_start = None
        self.is_executing = False
        self.ret_queue.put(ret_val)