import json
import os
import re
import time
import queue
from enum import Enum
from threading import Thread, Lock
from typing import List, Tuple, Union, Optional
from queue import Queue

from openai import ChatCompletion, Stream

from .skillset import SkillSet
from .utils.general_utils import split_args, print_t

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

def print_debug(*args):
    """Abilita/disabilita rapidamente la stampa di debug."""
    print(*args)
    # pass  # decommenta per silenziare

MiniSpecValueType = Union[int, float, bool, str, None]

def evaluate_value(value: str) -> MiniSpecValueType:
    value = str(value)
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit():
        return float(value)
    elif value == 'True':
        return True
    elif value == 'False':
        return False
    elif value == 'None' or len(value) == 0:
        return None
    else:
        return value.strip('\'"')

# ------------------------------------------------------------------------------
# Core data structures
# ------------------------------------------------------------------------------

class MiniSpecReturnValue:
    """Wrapper per valore di ritorno + flag di replanning"""
    def __init__(self, value: MiniSpecValueType, replan: bool, wait_user_answer: Optional[bool] = False):
        self.value = value
        self.replan = replan
        self.wait_user_answer = wait_user_answer

    @staticmethod
    def from_tuple(t: Tuple[MiniSpecValueType, bool, bool]):
        if len(t) == 3:
            return MiniSpecReturnValue(t[0], t[1], t[2])
        else:
            return MiniSpecReturnValue(t[0], t[1], False)

    @staticmethod
    def default():
        return MiniSpecReturnValue(None, False, False)

    def __repr__(self) -> str:               # pragma: no cover
        return f'value={self.value}, replan={self.replan}'

class ParsingState(Enum):
    CODE          = 0
    ARGUMENTS     = 1
    CONDITION     = 2
    LOOP_COUNT    = 3
    SUB_STATEMENTS= 4

# ------------------------------------------------------------------------------
# MiniSpecProgram
# ------------------------------------------------------------------------------

class MiniSpecProgram:
    """Rappresenta un intero programma MiniSpec già tokenizzato in Statement."""
    def __init__(self, env: Optional[dict] = None, mq: queue.Queue | None = None) -> None:
        self.statements: List[Statement] = []
        self.depth = 0
        self.finished = False
        self.ret = False
        self.env = {} if env is None else env
        self.current_statement = Statement(self.env)
        self.mq = mq

    # ------------------------------------------------------------------ parsing
    def parse(
        self,
        code_instance: Stream[ChatCompletion.ChatCompletionChunk] | List[str],
        exec: bool = False
    ) -> bool:
        """Parse pixel-stream o lista di stringhe in Statement"""
        for chunk in code_instance:
            code = chunk if isinstance(chunk, str) else chunk.choices[0].delta.content
            if not code:
                continue
            if self.mq:
                self.mq.put(code + '\\\\')

            for c in code:
                if self.current_statement.parse(c, exec):
                    # Statement terminato
                    if self.current_statement.action:
                        print_debug("Adding statement:", self.current_statement, exec)
                        self.statements.append(self.current_statement)
                    self.current_statement = Statement(self.env)

                # aggiorna depth parentesi graffe per capire fine programma
                if c == '{':
                    self.depth += 1
                elif c == '}':
                    if self.depth == 0:
                        self.finished = True
                        return True
                    self.depth -= 1
        return False

    # ------------------------------------------------------------------ runtime
    def eval(self) -> MiniSpecReturnValue:
        print_debug(f'Eval program: {self}, finished: {self.finished}')
        ret_val = MiniSpecReturnValue.default()
        idx = 0
        while not self.finished:
            if idx >= len(self.statements):
                time.sleep(0.1)
                continue
            ret_val = self.statements[idx].eval()
            if ret_val.replan or self.statements[idx].ret:
                self.ret = True
                return ret_val
            idx += 1

        # eventuali statement rimasti dopo finished
        for j in range(idx, len(self.statements)):
            ret_val = self.statements[j].eval()
            if ret_val.replan or self.statements[j].ret:
                self.ret = True
                return ret_val
        return ret_val

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:               # pragma: no cover
        return ' '.join(f'{st};' for st in self.statements)

# ------------------------------------------------------------------------------
# Statement
# ------------------------------------------------------------------------------

class Statement:
    execution_queue: Queue['Statement'] = None   # condivisa
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    # Add a reference to the interpreter to update program count
    interpreter = None

    def __init__(self, env: dict) -> None:
        self.code_buffer = ''
        self.parsing_state = ParsingState.CODE
        self.condition: str | None = None
        self.loop_count: int | None = None
        self.action = ''
        self.allow_digit = False
        self.executable = False
        self.ret = False
        self.sub_statements: MiniSpecProgram | None = None
        self.env = env
        self.read_argument = False
        self.depth_paren = 0  # ( … )

    # ---------------------------------------------------------- parsing helpers
    @staticmethod
    def _is_skill_declaration(text: str) -> bool:
        txt = text.lstrip()
        return txt.startswith(('as(', 'add_skill('))

    # ------------------------------------------------------------- parse stream
    def parse(self, code: str, exec: bool = False) -> bool:
        """Feed un carattere per volta. Restituisce True se lo Statement è chiuso."""
        for c in code:
            match self.parsing_state:
                # -------------------------------------------------- codice base
                case ParsingState.CODE:
                    # if/condition
                    if c == '?' and not self.read_argument:
                        self.action = 'if'
                        self.parsing_state = ParsingState.CONDITION

                    # fine statement con ';'
                    elif c == ';' and self.depth_paren == 0:
                        self.action = self.code_buffer.strip()
                        print_debug(f'SP Action: {self.action}')
                        self.executable = True
                        if exec and self.action:
                            self.execution_queue.put(self)
                            # Update program count when adding to queue
                            if Statement.interpreter:
                                with Statement.interpreter.program_lock:
                                    Statement.interpreter.program_count += 1
                        return True

                    # chiusura parentesi tonda a livello 0
                    elif c == ')':
                        self.depth_paren -= 1
                        self.code_buffer += c
                        if self.depth_paren == 0:
                            self.read_argument = False
                            self.action = self.code_buffer.strip()
                            print_debug(f'SP Action: {self.action}')
                            self.executable = True
                            if exec and self.action:
                                self.execution_queue.put(self)
                                # Update program count when adding to queue
                                if Statement.interpreter:
                                    with Statement.interpreter.program_lock:
                                        Statement.interpreter.program_count += 1
                            return True
                        
                    elif c == '}' and self.depth_paren == 0 and self.code_buffer.strip():
                        # Considera '}' come terminatore se abbiamo accumulato codice
                        self.action = self.code_buffer.strip()
                        self.executable = True
                        if exec and self.action:
                            self.execution_queue.put(self)
                            # Update program count when adding to queue
                            if Statement.interpreter:
                                with Statement.interpreter.program_lock:
                                    Statement.interpreter.program_count += 1
                        # L'attuale '}' verrà gestita dal chiamante (MiniSpecProgram)
                        return True

                    # accumula carattere
                    else:
                        if c == '(':
                            self.read_argument = True
                            self.depth_paren += 1
                        if c.isalpha() or c == '_':
                            self.allow_digit = True
                        self.code_buffer += c

                    # inizio di un loop numerico
                    if c.isdigit() and not self.allow_digit:
                        self.action = 'loop'
                        self.parsing_state = ParsingState.LOOP_COUNT

                # ----------------------------------------------------- condizione
                case ParsingState.CONDITION:
                    if c == '{':
                        print_debug(f'SP Condition: {self.code_buffer}')
                        self.condition = self.code_buffer.strip()
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                            # Update program count when adding to queue
                            if Statement.interpreter:
                                with Statement.interpreter.program_lock:
                                    Statement.interpreter.program_count += 1
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c

                # --------------------------------------------------- loop count
                case ParsingState.LOOP_COUNT:
                    if c == '{':
                        print_debug(f'SP Loop: {self.code_buffer}')
                        self.loop_count = int(self.code_buffer)
                        self.executable = True
                        if exec:
                            self.execution_queue.put(self)
                            # Update program count when adding to queue
                            if Statement.interpreter:
                                with Statement.interpreter.program_lock:
                                    Statement.interpreter.program_count += 1
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c

                # ------------------------------------------------ sub-statements
                case ParsingState.SUB_STATEMENTS:
                    if self.sub_statements.parse([c]):
                        return True
        return False

    # ---------------------------------------------------------------- eval     
    def eval(self) -> 'MiniSpecReturnValue':
        print_debug(f'Statement eval: {self.action}')
        while not self.executable:
            time.sleep(0.05)

        if self.action == 'if':
            ret_val = self.eval_condition(self.condition)
            if ret_val.replan:
                return ret_val
            if ret_val.value:
                ret_val = self.sub_statements.eval()
                if ret_val.replan or self.sub_statements.ret:
                    self.ret = True
                return ret_val
            return MiniSpecReturnValue.default()

        if self.action == 'loop':
            ret_val = MiniSpecReturnValue.default()
            for _ in range(self.loop_count):
                ret_val = self.sub_statements.eval()
                if ret_val.replan or self.sub_statements.ret:
                    self.ret = True
                    return ret_val
            return ret_val

        # normale espressione
        self.ret = False
        return self.eval_expr(self.action)

    # ---------------------------------------------------------------- eval func
    def get_env_value(self, var: str) -> MiniSpecValueType:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def eval_function(self, func: str) -> MiniSpecReturnValue:
        print_debug(f'Eval function: {func}')
        name, *rest = func.split('(', 1)
        name = name.strip()

        if rest:
            args_str = rest[0].rstrip(')')

            # Check if the entire argument string is a single list
            args_str = args_str.strip()
            if args_str.startswith('[') and args_str.endswith(']'):
                # Single list argument - don't split it
                args = [args_str]

            else:
                # Multiple arguments - use split_args
                args = split_args(args_str)
            
            # Process each argument
            processed_args = []
            for a in args:
                a = a.strip()

                # Check if it's a list literal
                if a.startswith('[') and a.endswith(']'):
                    import ast
                    try:
                        # Parse the list literal
                        parsed_list = ast.literal_eval(a)
                        processed_args.append(parsed_list)
                    except:
                        # If parsing fails, treat as string
                        processed_args.append(a.strip('\'"'))
                elif a.startswith('_'):
                    # Variable reference
                    processed_args.append(self.get_env_value(a))
                else:
                    # Regular string argument - check for variable interpolation
                    arg_value = a.strip('\'"')

                    # Look for variable references in the string (e.g., "text _varname more text")
                    import re
                    var_pattern = r'\b(_[a-zA-Z0-9_]*)\b'

                    def replace_var(match):
                        var_name = match.group(1)
                        try:
                            var_value = self.get_env_value(var_name)
                            return str(var_value)
                        except:
                            # If variable doesn't exists, leave it as-is
                            return var_name
                    
                    arg_value = re.sub(var_pattern, replace_var, arg_value)
                    processed_args.append(arg_value)

            args = processed_args
        else:
            args = []

        # built-in casts -------------------------------------------------------
        if name in ('int', 'float', 'str'):
            caster = {'int': int, 'float': float, 'str': str}[name]
            return MiniSpecReturnValue(caster(args[0]), False)

        # low-level skill ------------------------------------------------------
        skill = Statement.low_level_skillset.get_skill(name)
        if skill:
            print_debug(f'Executing low-level skill: {skill.get_name()} {args}')
            ret = MiniSpecReturnValue.from_tuple(skill.execute(args))
            if skill.get_name() == 'add_skill':
                Statement.high_level_skillset.update()
            return ret

        # high-level skill -----------------------------------------------------
        skill = Statement.high_level_skillset.get_skill(name)
        if skill:
            print_debug(f'Executing high-level skill: {skill.get_name()} {args}')
            print(args)
            print(type(args))
            interp = MiniSpecProgram(env=self.env)
            interp.parse([skill.execute(args)])
            interp.finished = True
            val = interp.eval()
            if val.replan:
                return MiniSpecReturnValue(val.value, True)
            
            # If the skill exited early with a return (but not a replan), pass through value
            if interp.ret:
                return MiniSpecReturnValue(val.value, False)
            
            return val

        raise Exception(f'Skill {name} is not defined')

    # --------------------------------------------------- helper '=' top-level
    @staticmethod
    def _top_level_eq_index(s: str) -> int:
        depth = 0
        in_single = in_double = False
        for i, ch in enumerate(s):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif in_single or in_double:
                continue
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '=' and depth == 0:
                return i
        return -1

    # ---------------------------------------------------------------- eval expr
    def eval_expr(self, expr: str) -> MiniSpecReturnValue:
        print_t(f'Eval expr: {expr}')
        expr = expr.strip()

        # return arrow ---------------------------------------------------------
        if expr.startswith('->'):
            self.ret = True
            result = self.eval_expr(expr.lstrip('->'))

            if result.replan:
                return MiniSpecReturnValue(result.value, True)  # replan signal
            elif isinstance(result.value, str) and any(keyword in result.value.lower()
                                                       for keyword in ['failed', 'error', 'replan']):
                return MiniSpecReturnValue(result.value, True)  # Error/replan signal
            else:
                # Normal return value (True, False, etc.) - not a replan
                return MiniSpecReturnValue(result.value, False)

        # assegnazione (=) -----------------------------------------------------
        eq_idx = self._top_level_eq_index(expr)
        if eq_idx != -1:
            left = expr[:eq_idx].strip()
            right = expr[eq_idx + 1:].strip()
            print_t(f'Eval expr var assign: {left} {right}')
            ret_val = self.eval_expr(right)
            self.env[left] = ret_val.value
            return ret_val

        # operatori aritmetici -------------------------------------------------
        for op in '+-*/':
            if op in expr and '(' not in expr:  # semplice split (non robustissimo ma suff.)
                parts = [p.strip() for p in expr.split(op)]
                base = self.eval_expr(parts[0]).value
                for p in parts[1:]:
                    val = self.eval_expr(p).value
                    if op == '+':
                        # somma numerica o concatenazione stringhe
                        base = base + val  # funzionerà per int/float o str
                    elif op == '-':
                        base -= val
                    elif op == '*':
                        base *= val
                    elif op == '/':
                        base /= val
                return MiniSpecReturnValue(base, False)

        # valori atomici -------------------------------------------------------
        if not expr:
            raise Exception('Empty operand')
        if expr.startswith('_'):
            return MiniSpecReturnValue(self.get_env_value(expr), False)
        if expr in ('True', 'False'):
            return MiniSpecReturnValue(evaluate_value(expr), False)
        if expr[0].isalpha():
            return self.eval_function(expr)
        return MiniSpecReturnValue(evaluate_value(expr), False)

    # -------------------------------------------------------------- conditions
    def eval_condition(self, condition: str) -> MiniSpecReturnValue:
        if '&' in condition:
            sub = condition.split('&')
            cond = True
            for c in sub:
                rv = self.eval_condition(c)
                if rv.replan:
                    return rv
                cond = cond and rv.value
            return MiniSpecReturnValue(cond, False)

        if '|' in condition:
            for c in condition.split('|'):
                rv = self.eval_condition(c)
                if rv.replan:
                    return rv
                if rv.value:
                    return MiniSpecReturnValue(True, False)
            return MiniSpecReturnValue(False, False)

        op1, comp, op2 = re.split(r'(>|<|==|!=)', condition)
        v1 = self.eval_expr(op1)
        if v1.replan:
            return v1
        v2 = self.eval_expr(op2)
        if v2.replan:
            return v2

        # coercion int<->float
        if isinstance(v1.value, (int, float)) and isinstance(v2.value, (int, float)):
            v1_val, v2_val = float(v1.value), float(v2.value)
        else:
            v1_val, v2_val = v1.value, v2.value

        # valuta SOLO il confronto richiesto, evitando di eseguire operazioni non necessarie
        if comp == '>':
            cmp_res = v1_val > v2_val
        elif comp == '<':
            cmp_res = v1_val < v2_val
        elif comp == '==':
            cmp_res = v1_val == v2_val
        elif comp == '!=':
            cmp_res = v1_val != v2_val
        else:
            raise Exception(f'Invalid comparator {comp}')

        return MiniSpecReturnValue(cmp_res, False)

    # ---------------------------------------------------------------- misc
    def __repr__(self) -> str:               # pragma: no cover
        if self.action == 'if':
            return f'if {self.condition} {{...}}'
        if self.action == 'loop':
            return f'[{self.loop_count}] {{...}}'
        return self.action

# ------------------------------------------------------------------------------
# MiniSpecInterpreter
# ------------------------------------------------------------------------------

class MiniSpecInterpreter:
    """Thread che esegue gli Statement in una coda FIFO condivisa."""
    def __init__(self, message_queue: queue.Queue):
        self.env: dict = {}
        self.execution_history: list[Statement] = []
        if Statement.low_level_skillset is None or Statement.high_level_skillset is None:
            raise Exception('Statement: Skillset is not initialized')

        Statement.execution_queue = Queue()
        Statement.interpreter = self  # Set reference to this interpreter
        self.execution_thread = Thread(target=self._executor, daemon=True)
        self.execution_thread.start()

        self.message_queue = message_queue
        self.ret_queue: Queue[MiniSpecReturnValue] = Queue()
        self.timestamp_get_plan = None
        self.timestamp_start_execution = None
        self.program_count = 0
        self.program_lock = Lock()  # Lock for thread-safe program_count updates
        self.is_executing = False  # Track if we're currently executing a program

    # ------------------------------------------------------------- public API
    def execute(self, code: Stream[ChatCompletion.ChatCompletionChunk] | List[str]) -> MiniSpecReturnValue:
        print_t('>>> Get a stream')
        self.timestamp_get_plan = time.time()

        self.execution_history.clear()
        
        # Don't reset program_count here - let the parse function update it
        with self.program_lock:
            if not self.is_executing:
                # Only reset if we're starting a fresh execution
                self.program_count = 0
            
        program = MiniSpecProgram(mq=self.message_queue)
        program.parse(code, exec=True)
        
        print_t('>>> Program:', program, 'Time:', time.time() - self.timestamp_get_plan)
        # attende risultato thread
        return self.ret_queue.get()

    # --------------------------------------------------------- executor thread
    def _executor(self):
        while True:
            if Statement.execution_queue.empty():
                time.sleep(0.005)
                continue

            with self.program_lock:
                if self.timestamp_start_execution is None and self.program_count > 0:
                    self.timestamp_start_execution = time.time()
                    self.is_executing = True
                    print_t('>>> Start execution')

            stmt = Statement.execution_queue.get()
            print_debug('Queue get statement:', stmt)
            ret_val = stmt.eval()
            print_t('Queue statement done:', stmt)

            self.execution_history.append(stmt) 
            
            with self.program_lock:
                if stmt.ret:                           # early return
                    with Statement.execution_queue.mutex:
                        Statement.execution_queue.queue.clear()
                    self.ret_queue.put(ret_val)
                    self._reset_timing()
                    continue                

                self.program_count -= 1
                if self.program_count == 0:            # programma terminato
                    print_t('>>> Execution time:',
                            time.time() - self.timestamp_start_execution)
                    self.ret_queue.put(ret_val)
                    self._reset_timing()

    def _reset_timing(self):
        with self.program_lock:
            self.timestamp_start_execution = None
            self.is_executing = False
            # Don't reset program_count to 0 here - let it be managed by the parser