from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}:{self.arg_type.__name__}"

class SkillItem(ABC):
    abbr_dict = {}

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_skill_description(self) -> str:
        pass

    @abstractmethod
    def get_argument(self) -> List[SkillArg]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def execute(self, arg_list: List[Union[int, float, str, List]]) -> Tuple[Union[int, float, bool, str], bool, bool]:
        pass


    def generate_abbreviation(self, word: str) -> str:
        parts = word.split('_')
        abbr  = ''.join(p[0] for p in parts)[:2]

        if abbr not in self.abbr_dict:
            self.abbr_dict[abbr] = word
            return abbr

        pool  = ''.join(parts)[1:]        # letters to try next
        count = 0
        while abbr in self.abbr_dict:
            if count >= len(pool):        # out of letters → use numbers
                num = 1
                while f"{abbr[0]}{num}" in self.abbr_dict:
                    num += 1
                abbr = f"{abbr[0]}{num}"
                break
            abbr = abbr[0] + pool[count]
            count += 1

        self.abbr_dict[abbr] = word
        return abbr


    def parse_args(self, args_str_list: List[Union[int, float, str, Optional[any]]], allow_positional_args: bool = False):
        """Parses the string of arguments and converts them to the expected types."""
        # Count required vs optional arguments
        required_args = 0
        for arg in self.args:
            arg_type = arg.arg_type
            required_args += 1
        
        # Check if we have at least the required arguments
        if len(args_str_list) < required_args:
            raise ValueError(f"Expected at least {required_args} arguments, but got {len(args_str_list)}.")
        
        # Check if we have too many arguments
        if len(args_str_list) > len(self.args):
            raise ValueError(f"Expected at most {len(self.args)} arguments, but got {len(args_str_list)}.")
        
        parsed_args = []
        for i in range(len(self.args)):
            # If we've run out of provided arguments, use None for remaining optional args
            if i >= len(args_str_list):
                arg_type = self.args[i].arg_type
                # Verify this is actually an optional argument
                if hasattr(arg_type, '__origin__') and arg_type.__origin__ is Union and type(None) in arg_type.__args__:
                    parsed_args.append(None)
                    continue
                else:
                    raise ValueError(f"Missing required argument at position {i + 1}")
            arg = args_str_list[i]

            # if arg is not a string, skip parsing
            if not isinstance(arg, str):
                parsed_args.append(arg)
                continue
            # Allow positional arguments
            if arg.startswith('$') and allow_positional_args:
                parsed_args.append(arg)
                continue
            try:
                arg_type = self.args[i].arg_type

                # Check if it's an Optional type
                if hasattr(arg_type, '__origin__') and arg_type.__origin__ is Union:
                    # Get the actual type from Optional[T] (which is Union[T, None])
                    actual_types = [t for t in arg_type.__args__ if t is not type(None)]

                    # Handle None/empty values
                    if arg.strip().lower() in ['none', ''] or arg.strip() == '':
                        parsed_args.append(None)
                        continue
                    # Try to parse with the actual type
                    if actual_types:
                        actual_type = actual_types[0]
                        if actual_type == bool:
                            parsed_args.append(arg.strip().lower() == 'true')
                        elif actual_type == list:
                            # Handle list types
                            import ast
                            try:
                                parsed_list = ast.literal_eval(arg.strip())
                                if not isinstance(parsed_list, list):
                                    parsed_list = [arg.strip()]
                                parsed_args.append([arg.strip()])
                            except:
                                parsed_args.append([arg.strip()])
                        else:
                            parsed_args.append(actual_type(arg.strip()))
                    else:
                        # Fallback if we can't determine the type
                        parsed_args.append(arg.strip())

                # Handle regular List type
                elif hasattr(arg_type, '__origin__') and arg_type.__origin__ is list:
                    import ast
                    try:
                        parsed_list = ast.literal_eval(arg.strip())
                        if not isinstance(parsed_list, list):
                            parsed_list = [arg.strip()]
                        parsed_args.append(parsed_list)
                    except:
                        parsed_args.append([arg.strip()])
                # Handle bool type
                elif arg_type == bool:
                    parsed_args.append(arg.strip().lower() == 'true')
                    
                # Handle regular types
                else:
                    parsed_args.append(arg_type(arg.strip()))
                    
            except ValueError as e:
                raise ValueError(f"Error parsing argument {i + 1}. Expected type {self.args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}")
                
        return parsed_args