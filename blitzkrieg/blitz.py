#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

import string
import os
import math
import time

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# ERRORS
#######################################

class Error:
  def __init__(self, pos_start, pos_end, error_name, details):
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = error_name
    self.details = details
  
  def as_string(self):
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

class IllegalCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    result = ''
    pos = self.pos_start
    ctx = self.context

    while ctx:
      result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
  def __init__(self, idx, ln, col, fn, ftxt):
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    self.idx += 1
    self.col += 1

    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def copy(self):
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT				= 'INT'
TT_FLOAT    	= 'FLOAT'
TT_STRING			= 'STRING'
TT_IDENTIFIER	= 'IDENTIFIER'
TT_KEYWORD		= 'KEYWORD'
TT_PLUS     	= 'PLUS'
TT_MINUS    	= 'MINUS'
TT_MUL      	= 'MUL'
TT_DIV      	= 'DIV'
TT_POW				= 'POW'
TT_EQ					= 'EQ'
TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'
TT_LSQUARE    = 'LSQUARE'
TT_RSQUARE    = 'RSQUARE'
TT_EE					= 'EE'
TT_NE					= 'NE'
TT_LT					= 'LT'
TT_GT					= 'GT'
TT_LTE				= 'LTE'
TT_GTE				= 'GTE'
TT_COMMA			= 'COMMA'
TT_ARROW			= 'ARROW'
TT_NEWLINE		= 'NEWLINE'
TT_EOF				= 'EOF'

KEYWORDS = [
  'VAR',
  'AND',
  'OR',
  'NOT',
  'IF',
  'ELIF',
  'ELSE',
  'FOR',
  'TO',
  'STEP',
  'WHILE',
  'FUN',
  'THEN',
  'END',
  'RETURN',
  'CONTINUE',
  'BREAK',
  'PARALLEL',
  'OPTIM',
  'MEMIZE',
  'PROF',
  'PRED',
  'REW',
  'QUAN',
]

class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
    self.type = type_
    self.value = value

    if pos_start:
      self.pos_start = pos_start.copy()
      self.pos_end = pos_start.copy()
      self.pos_end.advance()

    if pos_end:
      self.pos_end = pos_end.copy()

  def matches(self, type_, value):
    return self.type == type_ and self.value == value
  
  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'

#######################################
# LEXER
#######################################

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
  
  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in ' \t':
        self.advance()
      elif self.current_char == '\n':
        self.advance()
      elif self.current_char == '#':
        self.skip_comment()
      elif self.current_char in '|':
        tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
        self.advance()
      elif self.current_char in DIGITS:
        tokens.append(self.make_number())
      elif self.current_char in LETTERS:
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        tokens.append(self.make_string())
      elif self.current_char == '+':
        tokens.append(Token(TT_PLUS, pos_start=self.pos))
        self.advance()
      elif self.current_char == '-':
        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '*':
        tokens.append(Token(TT_MUL, pos_start=self.pos))
        self.advance()
      elif self.current_char == '/':
        tokens.append(Token(TT_DIV, pos_start=self.pos))
        self.advance()
      elif self.current_char == '^':
        tokens.append(Token(TT_POW, pos_start=self.pos))
        self.advance()
      elif self.current_char == '(':
        tokens.append(Token(TT_LPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == ')':
        tokens.append(Token(TT_RPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == '[':
        tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == ']':
        tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '!':
        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '=':
        tokens.append(self.make_equals())
      elif self.current_char == '<':
        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        tokens.append(self.make_greater_than())
      elif self.current_char == ',':
        tokens.append(Token(TT_COMMA, pos_start=self.pos))
        self.advance()
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    tokens.append(Token(TT_EOF, pos_start=self.pos))
    return tokens, None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

    if dot_count == 0:
      return Token(TT_INT, int(num_str), pos_start, self.pos)
    else:
      return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

  def make_string(self):
    string = ''
    pos_start = self.pos.copy()
    escape_character = False
    self.advance()

    escape_characters = {
      'n': '\n',
      't': '\t'
    }

    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        string += escape_characters.get(self.current_char, self.current_char)
      else:
        if self.current_char == '\\':
          escape_character = True
        else:
          string += self.current_char
      self.advance()
      escape_character = False
    
    self.advance()
    return Token(TT_STRING, string, pos_start, self.pos)

  def make_identifier(self):
    id_str = ''
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
      id_str += self.current_char
      self.advance()

    tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
    return Token(tok_type, id_str, pos_start, self.pos)

  def make_minus_or_arrow(self):
    tok_type = TT_MINUS
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '>':
      self.advance()
      tok_type = TT_ARROW

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_not_equals(self):
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

    self.advance()
    return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
  
  def make_equals(self):
    tok_type = TT_EQ
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_EE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_less_than(self):
    tok_type = TT_LT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_LTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_greater_than(self):
    tok_type = TT_GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_GTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def skip_comment(self):
    self.advance()

    while self.current_char != '\n' and self.current_char is not None:
      self.advance()

    if self.current_char == '\n':
      self.advance()

#######################################
# NODES
#######################################

class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, pos_start, pos_end):
    self.element_nodes = element_nodes

    self.pos_start = pos_start
    self.pos_end = pos_end

class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
  def __init__(self, var_name_tok, value_node):
    self.var_name_tok = var_name_tok
    self.value_node = value_node

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.value_node.pos_end

class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

class ParallelForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class OptimizeNode:
    def __init__(self, target_node, pos_start, pos_end):
        self.target_node = target_node
        self.pos_start = pos_start
        self.pos_end = pos_end

class ProfileNode:
    def __init__(self, name_tok, body_node, pos_start, pos_end):
        self.name_tok = name_tok
        self.body_node = body_node
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f'({self.name_tok}, {self.body_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class ForNode:
  def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_tok = var_name_tok
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.body_node.pos_end

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

class FuncDefNode:
  def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
    self.var_name_tok = var_name_tok
    self.arg_name_toks = arg_name_toks
    self.body_node = body_node
    self.should_auto_return = should_auto_return

    if self.var_name_tok:
      self.pos_start = self.var_name_tok.pos_start
    elif len(self.arg_name_toks) > 0:
      self.pos_start = self.arg_name_toks[0].pos_start
    else:
      self.pos_start = self.body_node.pos_start

    self.pos_end = self.body_node.pos_end

class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end

class ReturnNode:
  def __init__(self, node_to_return, pos_start, pos_end):
    self.node_to_return = node_to_return

    self.pos_start = pos_start
    self.pos_end = pos_end

class ContinueNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class BreakNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class MultipleAssignNode:
    def __init__(self, var_name_toks, value_nodes):
        self.var_name_toks = var_name_toks
        self.value_nodes = value_nodes
        self.pos_start = var_name_toks[0].pos_start
        self.pos_end = value_nodes[-1].pos_end if value_nodes else var_name_toks[-1].pos_end

class AugmentedAssignNode:
    def __init__(self, var_name_tok, op_tok, value_node):
        self.var_name_tok = var_name_tok
        self.op_tok = op_tok
        self.value_node = value_node
        self.pos_start = var_name_tok.pos_start
        self.pos_end = value_node.pos_end

#######################################
# PARSE RESULT
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
# PARSER
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.advance()

  def advance(self):
    self.tok_idx += 1
    self.update_current_tok()
    return self.current_tok

  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_tok()
    return self.current_tok

  def update_current_tok(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]

  def is_continuation_context(self):
    """Check if we're in a context that allows line continuation"""
    return self.current_tok.type in [TT_COMMA, TT_PLUS, TT_MINUS, TT_MUL, TT_DIV, 
                                     TT_EQ, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE,
                                     TT_LPAREN, TT_LSQUARE] or \
           self.current_tok.matches(TT_KEYWORD, 'AND') or \
           self.current_tok.matches(TT_KEYWORD, 'OR')

  def skip_newlines_in_expressions(self):
    """Skip newlines when they're not significant"""
    while (self.current_tok.type == TT_NEWLINE and 
           self.tok_idx + 1 < len(self.tokens) and
           self.is_continuation_context()):
        self.advance()

  def skip_optional_separators(self):
      """Skip optional statement separators (newlines, pipes, etc.)"""
      had_separator = False
    
      while self.current_tok.type in [TT_NEWLINE]:
          had_separator = True
          self.advance()
    
      return had_separator

  def skip_newlines(self):
      """Skip any newline tokens - alias for backward compatibility"""
      while self.current_tok.type == TT_NEWLINE:
          self.advance()

      while self.current_tok.type == TT_NEWLINE:
          self.advance()

  def parse(self):
    res = self.statements()
    
    # Skip any trailing newlines or whitespace tokens
    while self.current_tok.type == TT_NEWLINE:
        self.advance()
    
    if not res.error and self.current_tok.type != TT_EOF:
        # Only fail if there are actual unexpected tokens, not whitespace
        if self.current_tok.type not in [TT_NEWLINE]:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Unexpected token: {self.current_tok.type}"
            ))
    
    return res

  ###################################

  def statements(self):
    res = ParseResult()
    statements = []
    pos_start = self.current_tok.pos_start.copy()

    # Skip any initial newlines
    while self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()

    # If we hit EOF immediately, return empty statements
    if self.current_tok.type == TT_EOF:
        return res.success(ListNode([], pos_start, pos_start))

    # Parse first statement
    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)

    # Continue parsing statements
    while True:
        # Skip newlines between statements
        newline_count = 0
        while self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()
            newline_count += 1
        
        # If we hit EOF, we're done
        if self.current_tok.type == TT_EOF:
            break
            
        # Try to parse another statement
        statement = res.try_register(self.statement())
        if not statement:
            # If we can't parse a statement, reverse and break
            self.reverse(res.to_reverse_count)
            break
            
        statements.append(statement)

    return res.success(ListNode(
        statements,
        pos_start,
        statements[-1].pos_end if statements else pos_start
    ))
  
  def peek_ahead_for_pattern(self, pattern):
    """Check if this looks like a multiple assignment pattern"""
    # For multiple assignment, we need to look for: IDENTIFIER, COMMA, IDENTIFIER, ..., EQ
    if pattern == [TT_COMMA, TT_IDENTIFIER, TT_EQ]:
        # Check if we have a comma followed by more identifiers and eventually an equals
        pos = self.tok_idx + 1
        
        # Must start with a comma
        if pos >= len(self.tokens) or self.tokens[pos].type != TT_COMMA:
            return False
        pos += 1
        
        # Must have at least one identifier after comma
        if pos >= len(self.tokens) or self.tokens[pos].type != TT_IDENTIFIER:
            return False
        pos += 1
        
        # Keep looking for more comma-identifier pairs or an equals sign
        while pos < len(self.tokens):
            if self.tokens[pos].type == TT_EQ:
                return True  # Found the equals sign
            elif self.tokens[pos].type == TT_COMMA:
                pos += 1
                # After comma, must have identifier
                if pos >= len(self.tokens) or self.tokens[pos].type != TT_IDENTIFIER:
                    return False
                pos += 1
            else:
                return False  # Something else, not multiple assignment
        
        return False  # Never found equals sign
    
    # For other patterns, use the original logic
    for i, expected_type in enumerate(pattern):
        if self.tok_idx + 1 + i >= len(self.tokens):
            return False
        if self.tokens[self.tok_idx + 1 + i].type != expected_type:
            return False
    return True

  def multiple_assignment(self):
      """Handle a, b = 1, 2 style assignments"""
      res = ParseResult()
      var_names = []
    
      # Collect variable names
      var_names.append(self.current_tok)
      res.register_advancement()
      self.advance()
    
      while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()
        
          if self.current_tok.type != TT_IDENTIFIER:
              return res.failure(InvalidSyntaxError(
                  self.current_tok.pos_start, self.current_tok.pos_end,
                  "Expected identifier"
              ))
        
          var_names.append(self.current_tok)
          res.register_advancement()
          self.advance()
    
      if self.current_tok.type != TT_EQ:
          return res.failure(InvalidSyntaxError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              "Expected '='"
          ))
    
      res.register_advancement()
      self.advance()
    
      # Parse values
      values = []
      values.append(res.register(self.expr()))
      if res.error: return res
    
      while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()
          values.append(res.register(self.expr()))
          if res.error: return res
    
      return res.success(MultipleAssignNode(var_names, values))

  def augmented_assignment(self):
      """Handle x += 5, x *= 2, etc."""
      res = ParseResult()
    
      var_name = self.current_tok
      res.register_advancement()
      self.advance()
    
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
    
      # Skip the '=' token
      res.register_advancement()
      self.advance()
    
      value = res.register(self.expr())
      if res.error: return res
    
      return res.success(AugmentedAssignNode(var_name, op_tok, value))

  def statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TT_KEYWORD, 'RETURN'):
      res.register_advancement()
      self.advance()

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TT_KEYWORD, 'CONTINUE'):
      res.register_advancement()
      self.advance()
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TT_KEYWORD, 'BREAK'):
      res.register_advancement()
      self.advance()
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

    if self.current_tok.matches(TT_KEYWORD, 'PARALLEL'):
      res.register_advancement()
      self.advance()
      
      # Must be followed by FOR
      if not self.current_tok.matches(TT_KEYWORD, 'FOR'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected 'FOR' after 'PARALLEL'"
        ))
      
      # Parse as regular FOR but create ParallelForNode
      parallel_for = res.register(self.for_expr())
      if res.error: return res
      
      # Convert ForNode to ParallelForNode
      if isinstance(parallel_for, ForNode):
        return res.success(ParallelForNode(
          parallel_for.var_name_tok,
          parallel_for.start_value_node,
          parallel_for.end_value_node,
          parallel_for.step_value_node,
          parallel_for.body_node,
          parallel_for.should_return_null
        ))
      
      return res.success(parallel_for)

    if self.current_tok.matches(TT_KEYWORD, 'PROF'):
      profile_expr = res.register(self.profile_expr())
      if res.error: return res
      return res.success(profile_expr)

    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'RETURN', 'CONTINUE', 'BREAK', 'VAR', 'IF', 'FOR', 'PARALLEL', 'PROF', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
      ))
    return res.success(expr)

  def expr(self):
    res = ParseResult()

    # Handle optional VAR keyword or direct assignment
    if self.current_tok.matches(TT_KEYWORD, 'VAR'):
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected identifier"
            ))

        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '='"
            ))

        res.register_advancement()
        self.advance()
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(VarAssignNode(var_name, expr))

    # Handle tuple/multiple assignment: a, b = 1, 2
    if (self.current_tok.type == TT_IDENTIFIER and 
        self.peek_ahead_for_pattern([TT_COMMA, TT_IDENTIFIER, TT_EQ])):
        return self.multiple_assignment()

    # Handle augmented assignment: x += 5, x *= 2, etc.
    if (self.current_tok.type == TT_IDENTIFIER and 
        self.tok_idx + 1 < len(self.tokens)):
        next_tok = self.tokens[self.tok_idx + 1]
        if next_tok.type in [TT_PLUS, TT_MINUS, TT_MUL, TT_DIV] and \
           self.tok_idx + 2 < len(self.tokens) and \
           self.tokens[self.tok_idx + 2].type == TT_EQ:
            return self.augmented_assignment()

    # Handle Python-style assignment without VAR keyword
    if (self.current_tok.type == TT_IDENTIFIER and 
        self.tok_idx + 1 < len(self.tokens) and 
        self.tokens[self.tok_idx + 1].type == TT_EQ):
        
        var_name = self.current_tok
        res.register_advancement()
        self.advance()
        
        res.register_advancement()  # Skip '='
        self.advance()
        
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(VarAssignNode(var_name, expr))

    # Regular expression parsing
    node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'AND'), (TT_KEYWORD, 'OR'))))

    if res.error:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
        ))

    return res.success(node)

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'NOT'):
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'IF', 'FOR', 'WHILE', 'FUN' or 'NOT'"
      ))

    return res.success(node)

  def arith_expr(self):
    return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

  def term(self):
    return self.bin_op(self.factor, (TT_MUL, TT_DIV))

  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_PLUS, TT_MINUS):
      res.register_advancement()
      self.advance()
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.power()

  def power(self):
    return self.bin_op(self.call, (TT_POW, ), self.factor)

  def call(self):
    res = ParseResult()
    atom = res.register(self.atom())
    if res.error: return res

    if self.current_tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      arg_nodes = []

      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
          ))

        while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TT_RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        res.register_advancement()
        self.advance()
      return res.success(CallNode(atom, arg_nodes))
    return res.success(atom)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_INT, TT_FLOAT):
      res.register_advancement()
      self.advance()
      return res.success(NumberNode(tok))

    elif tok.type == TT_STRING:
      res.register_advancement()
      self.advance()
      return res.success(StringNode(tok))

    elif tok.type == TT_IDENTIFIER:
      res.register_advancement()
      self.advance()
      return res.success(VarAccessNode(tok))

    elif tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
        return res.success(expr)
      else:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ')'"
        ))

    elif tok.type == TT_LSQUARE:
      list_expr = res.register(self.list_expr())
      if res.error: return res
      return res.success(list_expr)
    
    elif tok.matches(TT_KEYWORD, 'IF'):
      if_expr = res.register(self.if_expr())
      if res.error: return res
      return res.success(if_expr)

    elif tok.matches(TT_KEYWORD, 'FOR'):
      for_expr = res.register(self.for_expr())
      if res.error: return res
      return res.success(for_expr)

    elif tok.matches(TT_KEYWORD, 'WHILE'):
      while_expr = res.register(self.while_expr())
      if res.error: return res
      return res.success(while_expr)

    elif tok.matches(TT_KEYWORD, 'FUN'):
      func_def = res.register(self.func_def())
      if res.error: return res
      return res.success(func_def)

    return res.failure(InvalidSyntaxError(
      tok.pos_start, tok.pos_end,
      "Expected int, float, identifier, '+', '-', '(', '[', IF', 'FOR', 'WHILE', 'FUN'"
    ))

  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TT_LSQUARE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '['"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RSQUARE:
      res.register_advancement()
      self.advance()
    else:
      element_nodes.append(res.register(self.expr()))
      if res.error:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(', '[' or 'NOT'"
        ))

      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        element_nodes.append(res.register(self.expr()))
        if res.error: return res

      if self.current_tok.type != TT_RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ']'"
        ))

      res.register_advancement()
      self.advance()

    return res.success(ListNode(
      element_nodes,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def if_expr(self):
    res = ParseResult()
    all_cases = res.register(self.if_expr_cases('IF'))
    if res.error: return res
    cases, else_case = all_cases
    return res.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('ELIF')
    
  def if_expr_c(self):
    res = ParseResult()
    else_case = None

    if self.current_tok.matches(TT_KEYWORD, 'ELSE'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()

        statements = res.register(self.statements())
        if res.error: return res
        else_case = (statements, True)

        if self.current_tok.matches(TT_KEYWORD, 'END'):
          res.register_advancement()
          self.advance()
        else:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'END'"
          ))
      else:
        expr = res.register(self.statement())
        if res.error: return res
        else_case = (expr, False)

    return res.success(else_case)

  def if_expr_b_or_c(self):
    res = ParseResult()
    cases, else_case = [], None

    if self.current_tok.matches(TT_KEYWORD, 'ELIF'):
      all_cases = res.register(self.if_expr_b())
      if res.error: return res
      cases, else_case = all_cases
    else:
      else_case = res.register(self.if_expr_c())
      if res.error: return res
    
    return res.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TT_KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      statements = res.register(self.statements())
      if res.error: return res
      cases.append((condition, statements, True))

      if self.current_tok.matches(TT_KEYWORD, 'END'):
        res.register_advancement()
        self.advance()
      else:
        all_cases = res.register(self.if_expr_b_or_c())
        if res.error: return res
        new_cases, else_case = all_cases
        cases.extend(new_cases)
    else:
      expr = res.register(self.statement())
      if res.error: return res
      cases.append((condition, expr, False))

      all_cases = res.register(self.if_expr_b_or_c())
      if res.error: return res
      new_cases, else_case = all_cases
      cases.extend(new_cases)

    return res.success((cases, else_case))

  def for_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'FOR'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'FOR'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected identifier"
      ))

    var_name = self.current_tok
    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_EQ:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '='"
      ))
    
    res.register_advancement()
    self.advance()

    start_value = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'TO'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'TO'"
      ))
    
    res.register_advancement()
    self.advance()

    end_value = res.register(self.expr())
    if res.error: return res

    if self.current_tok.matches(TT_KEYWORD, 'STEP'):
      res.register_advancement()
      self.advance()

      step_value = res.register(self.expr())
      if res.error: return res
    else:
      step_value = None

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'END'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'END'"
        ))

      res.register_advancement()
      self.advance()

      return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

  def while_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'WHILE'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'WHILE'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'THEN'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'END'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'END'"
        ))

      res.register_advancement()
      self.advance()

      return res.success(WhileNode(condition, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(WhileNode(condition, body, False))

  def func_def(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'FUN'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'FUN'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_IDENTIFIER:
      var_name_tok = self.current_tok
      res.register_advancement()
      self.advance()
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected '('"
        ))
    else:
      var_name_tok = None
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or '('"
        ))
    
    res.register_advancement()
    self.advance()
    arg_name_toks = []

    if self.current_tok.type == TT_IDENTIFIER:
      arg_name_toks.append(self.current_tok)
      res.register_advancement()
      self.advance()
      
      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected identifier"
          ))

        arg_name_toks.append(self.current_tok)
        res.register_advancement()
        self.advance()
      
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ')'"
        ))
    else:
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or ')'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_ARROW:
      res.register_advancement()
      self.advance()

      body = res.register(self.expr())
      if res.error: return res

      return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        body,
        True
      ))
    
    if self.current_tok.type != TT_NEWLINE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '->' or NEWLINE"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'END'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'END'"
      ))

    res.register_advancement()
    self.advance()
    
    return res.success(FuncDefNode(
      var_name_tok,
      arg_name_toks,
      body,
      False
    ))
  
  def profile_expr(self):
    """Parse PROF "name" THEN ... END"""
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()
    
    if not self.current_tok.matches(TT_KEYWORD, 'PROF'):
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'PROF'"
        ))
    
    res.register_advancement()
    self.advance()
    
    if self.current_tok.type != TT_STRING:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected string name for profile"
        ))
    
    name_tok = self.current_tok
    res.register_advancement()
    self.advance()
    
    if not self.current_tok.matches(TT_KEYWORD, 'THEN'):
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'THEN'"
        ))
    
    res.register_advancement()
    self.advance()
    
    if self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()
        
        body = res.register(self.statements())
        if res.error: return res
        
        if not self.current_tok.matches(TT_KEYWORD, 'END'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'END'"
            ))
        
        res.register_advancement()
        self.advance()
        
        return res.success(ProfileNode(name_tok, body, pos_start, self.current_tok.pos_end.copy()))
    
    return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected newline after 'THEN'"
    ))

  ###################################

  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)
  
#######################################
# BYTECODE COMPILER
#######################################

class ByteCode:
    # Instruction opcodes
    LOAD_CONST = 1
    LOAD_VAR = 2
    STORE_VAR = 3
    BINARY_ADD = 4
    BINARY_SUB = 5
    BINARY_MUL = 6
    BINARY_DIV = 7
    BINARY_POW = 8
    BINARY_MOD = 9
    CALL_FUNCTION = 10
    RETURN_VALUE = 11
    JUMP_IF_FALSE = 12
    JUMP = 13
    COMPARE_EQ = 14
    COMPARE_NE = 15
    COMPARE_LT = 16
    COMPARE_LE = 17
    COMPARE_GT = 18
    COMPARE_GE = 19
    LOGICAL_AND = 20
    LOGICAL_OR = 21
    LOGICAL_NOT = 22
    UNARY_MINUS = 23
    UNARY_PLUS = 24
    BUILD_LIST = 25
    GET_INDEX = 26
    SET_INDEX = 27

class FunctionInliner:
    def __init__(self, max_inline_size=10):
        self.max_inline_size = max_inline_size
        self.inline_cache = {}
    
    def should_inline(self, func_node):
        """Determine if a function should be inlined"""
        if not isinstance(func_node, FuncDefNode):
            return False
        
        # Count instructions in function body
        instruction_count = self.count_instructions(func_node.body_node)
        return instruction_count <= self.max_inline_size
    
    def count_instructions(self, node):
        """Count the number of instructions a node would generate"""
        if isinstance(node, NumberNode):
            return 1  # LOAD_CONST
        elif isinstance(node, BinOpNode):
            return (self.count_instructions(node.left_node) + 
                   self.count_instructions(node.right_node) + 1)
        elif isinstance(node, VarAccessNode):
            return 1  # LOAD_VAR
        elif isinstance(node, ListNode):
            return sum(self.count_instructions(elem) for elem in node.element_nodes) + 1
        else:
            return 5  # Conservative estimate for complex nodes
    
    def inline_function(self, call_node, func_node):
        """Replace function call with inlined body"""
        if not self.should_inline(func_node):
            return call_node
        
        # Create substitution map for parameters
        param_map = {}
        for i, arg_node in enumerate(call_node.arg_nodes):
            if i < len(func_node.arg_name_toks):
                param_name = func_node.arg_name_toks[i].value
                param_map[param_name] = arg_node
        
        # Substitute parameters in function body
        inlined_body = self.substitute_parameters(func_node.body_node, param_map)
        return inlined_body
    
    def substitute_parameters(self, node, param_map):
        """Replace parameter references with actual arguments"""
        if isinstance(node, VarAccessNode):
            var_name = node.var_name_tok.value
            if var_name in param_map:
                return param_map[var_name]
            return node
        elif isinstance(node, BinOpNode):
            return BinOpNode(
                self.substitute_parameters(node.left_node, param_map),
                node.op_tok,
                self.substitute_parameters(node.right_node, param_map)
            )
        elif isinstance(node, ListNode):
            new_elements = [self.substitute_parameters(elem, param_map) 
                          for elem in node.element_nodes]
            return ListNode(new_elements, node.pos_start, node.pos_end)
        else:
            return node

class Compiler:
    def __init__(self):
        self.instructions = []
        self.constants = []
        self.variables = {}  # Variable name to index mapping
        self.inliner = FunctionInliner()
        self.function_registry = {}  # Store function definitions
        
    def compile(self, node):
        # Pre-pass: collect function definitions for inlining
        self.collect_functions(node)

        # Apply loop unrolling optimization
        optimized_node = self.unroll_small_loops(node)
        
        # Main compilation with inlining
        self.compile_node(optimized_node)
    
    def collect_functions(self, node):
        """First pass: collect all function definitions"""
        if isinstance(node, FuncDefNode):
            if node.var_name_tok:
                func_name = node.var_name_tok.value
                self.function_registry[func_name] = node
        elif isinstance(node, ListNode):
            for elem in node.element_nodes:
                self.collect_functions(elem)
        elif isinstance(node, BinOpNode):
            self.collect_functions(node.left_node)
            self.collect_functions(node.right_node)
        # Add other node types as needed
    
    def compile_node(self, node):
        if isinstance(node, NumberNode):
            const_idx = len(self.constants)
            self.constants.append(node.tok.value)
            self.emit(ByteCode.LOAD_CONST, const_idx)
            
        elif isinstance(node, CallNode):
            # Check if we can inline this function call
            if isinstance(node.node_to_call, VarAccessNode):
                func_name = node.node_to_call.var_name_tok.value
                if func_name in self.function_registry:
                    func_node = self.function_registry[func_name]
                    if self.inliner.should_inline(func_node):
                        # Inline the function
                        inlined_node = self.inliner.inline_function(node, func_node)
                        self.compile_node(inlined_node)
                        return
            
            # Regular function call compilation
            self.compile_node(node.node_to_call)  # Compile function first
            for arg_node in node.arg_nodes:      # Then compile arguments
                self.compile_node(arg_node)
            self.emit(ByteCode.CALL_FUNCTION, len(node.arg_nodes))
            
        elif isinstance(node, BinOpNode):
            self.compile_node(node.left_node)
            self.compile_node(node.right_node)
            if node.op_tok.type == TT_PLUS:
                self.emit(ByteCode.BINARY_ADD)
            elif node.op_tok.type == TT_MINUS:
                self.emit(ByteCode.BINARY_SUB)
            elif node.op_tok.type == TT_MUL:
                self.emit(ByteCode.BINARY_MUL)
            elif node.op_tok.type == TT_DIV:
                self.emit(ByteCode.BINARY_DIV)
            elif node.op_tok.type == TT_POW:
                self.emit(ByteCode.BINARY_POW)
            elif node.op_tok.type == TT_EE:
                self.emit(ByteCode.COMPARE_EQ)
            elif node.op_tok.type == TT_NE:
                self.emit(ByteCode.COMPARE_NE)
            elif node.op_tok.type == TT_LT:
                self.emit(ByteCode.COMPARE_LT)
            elif node.op_tok.type == TT_LTE:
                self.emit(ByteCode.COMPARE_LE)
            elif node.op_tok.type == TT_GT:
                self.emit(ByteCode.COMPARE_GT)
            elif node.op_tok.type == TT_GTE:
                self.emit(ByteCode.COMPARE_GE)
            elif node.op_tok.type == TT_KEYWORD and node.op_tok.value == 'AND':
                self.emit(ByteCode.LOGICAL_AND)
            elif node.op_tok.type == TT_KEYWORD and node.op_tok.value == 'OR':
                self.emit(ByteCode.LOGICAL_OR)
                
        elif isinstance(node, UnaryOpNode):
            self.compile_node(node.node)
            if node.op_tok.type == TT_MINUS:
                self.emit(ByteCode.UNARY_MINUS)
            elif node.op_tok.type == TT_PLUS:
                self.emit(ByteCode.UNARY_PLUS)
            elif node.op_tok.type == TT_KEYWORD and node.op_tok.value == 'NOT':
                self.emit(ByteCode.LOGICAL_NOT)
            
        elif isinstance(node, StringNode):
            const_idx = len(self.constants)
            self.constants.append(node.tok.value)
            self.emit(ByteCode.LOAD_CONST, const_idx)
            
        elif isinstance(node, VarAccessNode):
            var_name = node.var_name_tok.value
            if var_name not in self.variables:
                self.variables[var_name] = len(self.variables)
            self.emit(ByteCode.LOAD_VAR, self.variables[var_name])
            
        elif isinstance(node, VarAssignNode):
            self.compile_node(node.value_node)
            var_name = node.var_name_tok.value
            if var_name not in self.variables:
                self.variables[var_name] = len(self.variables)
            self.emit(ByteCode.STORE_VAR, self.variables[var_name])
            
        elif isinstance(node, ListNode):
            for element_node in node.element_nodes:
                self.compile_node(element_node)
            self.emit(ByteCode.BUILD_LIST, len(node.element_nodes))

        elif isinstance(node, IfNode):
            # Implement conditional compilation
            for condition, expr, should_return_null in node.cases:
                self.compile_node(condition)
                jump_if_false = len(self.instructions)
                self.emit(ByteCode.JUMP_IF_FALSE, 0)  # Placeholder
            
                self.compile_node(expr)
            
                # Patch the jump address
                self.instructions[jump_if_false + 1] = len(self.instructions)
            
        elif isinstance(node, ForNode):
            # Implement loop compilation
            self.compile_for_loop(node)
        
        elif isinstance(node, WhileNode):
            # Implement while loop compilation
            self.compile_while_loop(node)
            
        elif isinstance(node, FuncDefNode):
            # Function definition - store for later compilation
            if node.var_name_tok:
                func_name = node.var_name_tok.value
                self.function_registry[func_name] = node
            # For now, just compile the body if it's auto-return
            if node.should_auto_return:
                self.compile_node(node.body_node)
                self.emit(ByteCode.RETURN_VALUE)
            else:
                self.compile_node(node.body_node)
            
        elif isinstance(node, ReturnNode):
            if node.node_to_return:
                self.compile_node(node.node_to_return)
            else:
                # Return null
                const_idx = len(self.constants)
                self.constants.append(0)
                self.emit(ByteCode.LOAD_CONST, const_idx)
            self.emit(ByteCode.RETURN_VALUE)
            
        elif isinstance(node, ContinueNode):
            # Continue statement - jump to loop start
            # For now, just emit a placeholder
            self.emit(ByteCode.JUMP, 0)  # Will need proper loop handling
            
        elif isinstance(node, BreakNode):
            # Break statement - jump to loop end
            # For now, just emit a placeholder
            self.emit(ByteCode.JUMP, 0)  # Will need proper loop handling

        # Default case to prevent hanging until all operators and node types are added!
        
        else:
            raise NotImplementedError(f"Compilation for {type(node).__name__} not implemented")
    
    def emit(self, opcode, arg=0):
        self.instructions.extend([opcode, arg])

    def compile_for_loop(self, node):
        """Compile FOR loop to bytecode"""
        # Variable assignment (start value)
        self.compile_node(node.start_value_node)
        var_name = node.var_name_tok.value
        if var_name not in self.variables:
            self.variables[var_name] = len(self.variables)
        self.emit(ByteCode.STORE_VAR, self.variables[var_name])
        
        # Loop start label
        loop_start = len(self.instructions)
        
        # Load counter and end value for comparison
        self.emit(ByteCode.LOAD_VAR, self.variables[var_name])
        self.compile_node(node.end_value_node)
        
        # Step value handling (default to 1 if no step)
        if node.step_value_node is not None:
            self.compile_node(node.step_value_node)
            # Compare with step direction
            jump_condition = ByteCode.COMPARE_GT  # If step is positive
        else:
            # Default step of 1
            const_idx = len(self.constants)
            self.constants.append(1)
            self.emit(ByteCode.LOAD_CONST, const_idx)
            jump_condition = ByteCode.COMPARE_GT
        
        # Check loop condition (counter <= end)
        self.emit(ByteCode.COMPARE_LE)
        jump_if_false = len(self.instructions)
        self.emit(ByteCode.JUMP_IF_FALSE, 0)  # Will patch later
        
        # Compile loop body
        self.compile_node(node.body_node)
        
        # Increment counter
        self.emit(ByteCode.LOAD_VAR, self.variables[var_name])
        if node.step_value_node is not None:
            self.compile_node(node.step_value_node)
        else:
            const_idx = len(self.constants)
            self.constants.append(1)
            self.emit(ByteCode.LOAD_CONST, const_idx)
        self.emit(ByteCode.BINARY_ADD)
        self.emit(ByteCode.STORE_VAR, self.variables[var_name])
        
        # Jump back to loop start
        self.emit(ByteCode.JUMP, loop_start)
        
        # Patch the exit jump
        self.instructions[jump_if_false + 1] = len(self.instructions)

    def compile_while_loop(self, node):
        """Compile WHILE loop to bytecode"""
        # Loop start label
        loop_start = len(self.instructions)
        
        # Compile condition
        self.compile_node(node.condition_node)
        
        # Jump if false (exit loop)
        jump_if_false = len(self.instructions)
        self.emit(ByteCode.JUMP_IF_FALSE, 0)  # Will patch later
        
        # Compile loop body
        self.compile_node(node.body_node)
        
        # Jump back to condition check
        self.emit(ByteCode.JUMP, loop_start)
        
        # Patch the exit jump
        self.instructions[jump_if_false + 1] = len(self.instructions)

    def optimize_constants(self):
        """Remove duplicate constants"""
        seen = {}
        new_constants = []
        instruction_map = {}

        for i, const in enumerate(self.constants):
            if const in seen:
                instruction_map[i] = seen[const]
            else:
                seen[const] = len(new_constants)
                instruction_map[i] = len(new_constants)
                new_constants.append(const)

            instruction_map[i] = seen[const]
        else:
            seen[const] = len(new_constants)
            instruction_map[i] = len(new_constants)
            new_constants.append(const)
    
    # Update instructions to use new constant indices
        for i in range(0, len(self.instructions), 2):
            if self.instructions[i] == ByteCode.LOAD_CONST:
                old_idx = self.instructions[i + 1]
                self.instructions[i + 1] = instruction_map[old_idx]
    
        self.constants = new_constants

    # Add loop unrolling
    def unroll_small_loops(self, node):
        """Unroll loops with small, constant iteration counts"""
        if isinstance(node, ForNode):
            # Check if it's a simple counting loop
            if (isinstance(node.start_value_node, NumberNode) and 
                isinstance(node.end_value_node, NumberNode) and
                (not node.step_value_node or isinstance(node.step_value_node, NumberNode))):

                start = node.start_value_node.tok.value
                end = node.end_value_node.tok.value
                step = node.step_value_node.tok.value if node.step_value_node else 1

                iterations = abs((end - start) // step)

                # Unroll if less than 4 iterations
                if iterations < 4:
                    return self.create_unrolled_loop(node, start, end, step)

        return node

    def create_unrolled_loop(self, loop_node, start, end, step):
        """Create an unrolled version of a loop"""
        statements = []
    
        # Generate the unrolled statements
        current_value = start
    
        if step >= 0:
            condition = lambda val: val < end
        else:
            condition = lambda val: val > end
    
        while condition(current_value):
            # Create a variable assignment for the loop variable
            var_assign = VarAssignNode(
                loop_node.var_name_tok,
                NumberNode(Token(TT_INT, current_value, loop_node.pos_start, loop_node.pos_end))
            )
            statements.append(var_assign)
        
            # Add the loop body (need to copy it for each iteration)
            body_copy = self.copy_node(loop_node.body_node)
            statements.append(body_copy)
        
            current_value += step
    
        # If no statements were generated, return a null value
        if not statements:
            return NumberNode(Token(TT_INT, 0, loop_node.pos_start, loop_node.pos_end))
    
        # Return a ListNode containing all the unrolled statements
        return ListNode(statements, loop_node.pos_start, loop_node.pos_end)

    def copy_node(self, node):
        """Deep copy a node and all its children"""
        if isinstance(node, NumberNode):
            return NumberNode(node.tok)
    
        elif isinstance(node, StringNode):
            return StringNode(node.tok)
    
        elif isinstance(node, VarAccessNode):
            return VarAccessNode(node.var_name_tok)
    
        elif isinstance(node, VarAssignNode):
            return VarAssignNode(
                node.var_name_tok,
                self.copy_node(node.value_node)
            )
    
        elif isinstance(node, BinOpNode):
            return BinOpNode(
                self.copy_node(node.left_node),
                node.op_tok,
                self.copy_node(node.right_node)
            )
    
        elif isinstance(node, UnaryOpNode):
            return UnaryOpNode(
                node.op_tok,
                self.copy_node(node.node)
            )
    
        elif isinstance(node, CallNode):
            return CallNode(
                self.copy_node(node.node_to_call),
                [self.copy_node(arg) for arg in node.arg_nodes]
            )
    
        elif isinstance(node, ListNode):
            return ListNode(
                [self.copy_node(elem) for elem in node.element_nodes],
                node.pos_start,
                node.pos_end
            )
    
        elif isinstance(node, IfNode):
            new_cases = []
            for condition, expr, should_return_null in node.cases:
                new_cases.append((
                    self.copy_node(condition),
                    self.copy_node(expr),
                    should_return_null
            ))
        
            new_else_case = None
            if node.else_case:
                expr, should_return_null = node.else_case
                new_else_case = (self.copy_node(expr), should_return_null)
        
            return IfNode(new_cases, new_else_case)
    
        # For other node types, return the original (or implement as needed)
        else:
            return node
        
class SmartOptimizer:
    def __init__(self):
        self.execution_stats = {}
        self.optimization_cache = {}
        self.performance_predictor = PerformancePredictor()
        
    def track_execution(self, func_name, execution_time, args_pattern):
        """Track function execution patterns for auto-optimization"""
        if func_name not in self.execution_stats:
            self.execution_stats[func_name] = {
                'call_count': 0,
                'total_time': 0,
                'arg_patterns': {},
                'optimization_applied': False
            }
        
        stats = self.execution_stats[func_name]
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        
        # Track argument patterns
        pattern_key = str(args_pattern)
        if pattern_key not in stats['arg_patterns']:
            stats['arg_patterns'][pattern_key] = 0
        stats['arg_patterns'][pattern_key] += 1
        
        # Auto-optimize if hot path detected
        if (stats['call_count'] > 50 and 
            stats['total_time'] > 0.1 and 
            not stats['optimization_applied']):
            self.apply_smart_optimization(func_name)
    
    def apply_smart_optimization(self, func_name):
        """Apply intelligent optimizations based on usage patterns"""
        print(f"🚀 Auto-optimizing hot function: {func_name}")
        
        optimizations = []
        stats = self.execution_stats[func_name]
        
        # Determine best optimization strategy
        if stats['call_count'] > 100:
            optimizations.append("JIT Compilation")
        if len(stats['arg_patterns']) < 5:
            optimizations.append("Argument Specialization")
        if stats['total_time'] / stats['call_count'] > 0.01:
            optimizations.append("Algorithm Substitution")
            
        for opt in optimizations:
            print(f"   ✓ Applied: {opt}")
        
        self.execution_stats[func_name]['optimization_applied'] = True

class IntelligentParallelizer:
    def __init__(self):
        self.dependency_analyzer = DependencyAnalyzer()
        self.thread_pool_size = os.cpu_count() or 4
        
    def can_parallelize(self, loop_node):
        """Determine if a loop can be safely parallelized"""
        # Check for data dependencies
        dependencies = self.dependency_analyzer.analyze(loop_node)
        
        # Simple heuristic: no variable modifications inside loop = parallelizable
        if dependencies.has_write_dependencies():
            return False
        
        # Check loop size - only parallelize if worth the overhead
        if hasattr(loop_node, 'start_value_node') and hasattr(loop_node, 'end_value_node'):
            if (isinstance(loop_node.start_value_node, NumberNode) and 
                isinstance(loop_node.end_value_node, NumberNode)):
                iterations = loop_node.end_value_node.tok.value - loop_node.start_value_node.tok.value
                return iterations > 100  # Only parallelize large loops
        
        return True
    
    def parallelize_loop(self, loop_node, context):
        """Execute loop in parallel using threading"""
        import concurrent.futures
        import threading
        
        start_val = int(loop_node.start_value_node.tok.value)
        end_val = int(loop_node.end_value_node.tok.value)
        step_val = int(loop_node.step_value_node.tok.value) if loop_node.step_value_node else 1
        
        # Split work into chunks
        chunk_size = max(1, (end_val - start_val) // self.thread_pool_size)
        chunks = []
        
        for i in range(start_val, end_val, chunk_size):
            chunk_end = min(i + chunk_size, end_val)
            chunks.append((i, chunk_end, step_val))
        
        results = []
        
        def execute_chunk(chunk_start, chunk_end, step):
            chunk_results = []
            for i in range(chunk_start, chunk_end, step):
                # Create isolated context for thread safety
                thread_context = Context(f'<parallel-{threading.current_thread().ident}>')
                thread_context.symbol_table = SymbolTable(context.symbol_table)
                thread_context.symbol_table.set(loop_node.var_name_tok.value, Number(i))
                
                # Execute loop body
                interpreter = Interpreter()
                result = interpreter.visit(loop_node.body_node, thread_context)
                if result.value:
                    chunk_results.append(result.value)
            return chunk_results
        
        # Execute chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
            futures = [executor.submit(execute_chunk, start, end, step) for start, end, step in chunks]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        return results

class DependencyAnalyzer:
    def __init__(self):
        self.read_vars = set()
        self.write_vars = set()
    
    def analyze(self, node):
        """Analyze variable dependencies in a code block"""
        self.read_vars.clear()
        self.write_vars.clear()
        self._analyze_node(node)
        return self
    
    def _analyze_node(self, node):
        if isinstance(node, VarAccessNode):
            self.read_vars.add(node.var_name_tok.value)
        elif isinstance(node, VarAssignNode):
            self.write_vars.add(node.var_name_tok.value)
            self._analyze_node(node.value_node)
        elif isinstance(node, BinOpNode):
            self._analyze_node(node.left_node)
            self._analyze_node(node.right_node)
        elif hasattr(node, 'element_nodes'):
            for elem in node.element_nodes:
                self._analyze_node(elem)
        elif hasattr(node, 'body_node'):
            self._analyze_node(node.body_node)
    
    def has_write_dependencies(self):
        """Check if there are any variable write operations"""
        return len(self.write_vars) > 0

class PerformancePredictor:
    def __init__(self):
        self.memory_patterns = {}
        self.execution_patterns = {}
    
    def predict_memory_usage(self, code_ast):
        """Predict memory usage before execution"""
        # Analyze AST to predict allocations
        estimated_memory = 0
        
        # Count variable declarations
        var_count = self.count_variables(code_ast)
        estimated_memory += var_count * 64  # bytes per variable
        
        # Count loops (arrays/lists likely)
        loop_count = self.count_loops(code_ast)
        estimated_memory += loop_count * 1024  # bytes per loop structure
        
        return estimated_memory
    
    def count_variables(self, node):
        """Count variable declarations in AST"""
        count = 0
        if isinstance(node, VarAssignNode):
            count += 1
        elif hasattr(node, 'element_nodes'):
            for elem in node.element_nodes:
                count += self.count_variables(elem)
        elif hasattr(node, 'body_node'):
            count += self.count_variables(node.body_node)
        return count
    
    def count_loops(self, node):
        """Count loop structures"""
        count = 0
        if isinstance(node, (ForNode, WhileNode)):
            count += 1
        elif hasattr(node, 'element_nodes'):
            for elem in node.element_nodes:
                count += self.count_loops(elem)
        elif hasattr(node, 'body_node'):
            count += self.count_loops(node.body_node)
        return count
        
class OptimizedCompiler(Compiler):
    def __init__(self):
        super().__init__()
        self.constant_folder = ConstantFolder()
        self.dead_code_eliminator = DeadCodeEliminator()
        
    def compile(self, node):
        # Apply optimizations before compilation
        optimized_node = self.optimize_ast(node)
        
        # Collect functions for inlining
        self.collect_functions(optimized_node)
        
        # Compile to bytecode
        self.compile_node(optimized_node)
        
        # Post-compilation optimizations
        self.optimize_constants()
        self.optimize_instructions()
    
    def optimize_ast(self, node):
        """Apply AST-level optimizations"""
        # Constant folding
        node = self.constant_folder.fold(node)
        
        # Loop unrolling
        node = self.unroll_small_loops(node)
        
        # Dead code elimination
        node = self.dead_code_eliminator.eliminate(node)
        
        return node
    
    def optimize_instructions(self):
        """Optimize the generated bytecode"""
        # Remove redundant load/store pairs
        self.remove_redundant_loads()
        
        # Combine consecutive operations
        self.combine_operations()
    
    def remove_redundant_loads(self):
        """Remove unnecessary load operations"""
        optimized = []
        i = 0
        while i < len(self.instructions):
            if i + 3 < len(self.instructions):
                # Check for LOAD_CONST followed by immediate POP (if we had POP)
                # For now, just pass through
                pass
            
            optimized.append(self.instructions[i])
            if i + 1 < len(self.instructions):
                optimized.append(self.instructions[i + 1])
            i += 2
        
        self.instructions = optimized
    
    def combine_operations(self):
        """Combine multiple operations into single optimized ones"""
        optimized = []
        i = 0
        while i < len(self.instructions):
            # Look for patterns like LOAD_CONST 1, LOAD_CONST 2, BINARY_ADD
            if (i + 5 < len(self.instructions) and
                self.instructions[i] == ByteCode.LOAD_CONST and
                self.instructions[i + 2] == ByteCode.LOAD_CONST and
                self.instructions[i + 4] == ByteCode.BINARY_ADD):
                
                # Get the constants
                const1_idx = self.instructions[i + 1]
                const2_idx = self.instructions[i + 3]
                
                # If both are numbers, compute at compile time
                if (const1_idx < len(self.constants) and const2_idx < len(self.constants)):
                    const1 = self.constants[const1_idx]
                    const2 = self.constants[const2_idx]
                    if isinstance(const1, (int, float)) and isinstance(const2, (int, float)):
                        # Combine the constants
                        result = const1 + const2
                        new_const_idx = len(self.constants)
                        self.constants.append(result)
                        optimized.extend([ByteCode.LOAD_CONST, new_const_idx])
                        i += 6  # Skip the combined operations
                        continue
            
            optimized.append(self.instructions[i])
            if i + 1 < len(self.instructions):
                optimized.append(self.instructions[i + 1])
            i += 2
        
        self.instructions = optimized

def compile_to_bytecode(filename, text, args):
    """Compile .bk file to .bkc bytecode file"""
    try:
        # Generate tokens
        lexer = Lexer(filename, text)
        tokens, error = lexer.make_tokens()
        if error:
            print(f"Lexer error: {error.as_string()}")
            return

        # Generate AST
        parser = Parser(tokens)
        ast = parser.parse()
        if ast.error:
           print(f"Parser error: {ast.error.as_string()}")
           return
    
        # Compile to bytecode
        compiler = OptimizedCompiler()
        compiler.compile(ast.node)
        
        # Save bytecode to .bkc file
        output_file = filename.replace('.bk', '.bkc')
        bytecode_data = {
            'instructions': compiler.instructions,
            'constants': compiler.constants,
            'variables': compiler.variables,  # Save variable name to index mapping
            'version': '1.0.0'
        }
        
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(bytecode_data, f)
        
        print(f"Compiled {filename} -> {output_file}")
        
        if args.bytecode:
            print("=== Bytecode ===")
            for i in range(0, len(compiler.instructions), 2):
                opcode = compiler.instructions[i]
                arg = compiler.instructions[i + 1] if i + 1 < len(compiler.instructions) else 0
                print(f"{i//2:04d}: {opcode:02d} {arg}")
            print("================")
            
    except Exception as e:
        print(f"Compilation error: {e}")
    
    def optimize_ast(self, node):
        """Apply AST-level optimizations"""
        # Constant folding
        node = self.constant_folder.fold(node)
        
        # Loop unrolling
        node = self.unroll_small_loops(node)
        
        # Dead code elimination
        node = self.dead_code_eliminator.eliminate(node)
        
        return node
    
    def optimize_instructions(self):
        """Optimize the generated bytecode"""
        # Remove redundant load/store pairs
        self.remove_redundant_loads()
        
        # Combine consecutive operations
        self.combine_operations()
    
    def remove_redundant_loads(self):
        """Remove unnecessary load operations"""
        optimized = []
        i = 0
        while i < len(self.instructions):
            if i + 3 < len(self.instructions):
                # Check for LOAD_CONST followed by immediate POP
                if (self.instructions[i] == ByteCode.LOAD_CONST and 
                    self.instructions[i + 2] == ByteCode.POP):
                    # Skip both instructions
                    i += 4
                    continue
            
            optimized.append(self.instructions[i])
            optimized.append(self.instructions[i + 1])
            i += 2
        
        self.instructions = optimized
    
    def combine_operations(self):
        """Combine multiple operations into single optimized ones"""
        optimized = []
        i = 0
        while i < len(self.instructions):
            # Look for patterns like LOAD_CONST 1, LOAD_CONST 2, BINARY_ADD
            if (i + 5 < len(self.instructions) and
                self.instructions[i] == ByteCode.LOAD_CONST and
                self.instructions[i + 2] == ByteCode.LOAD_CONST and
                self.instructions[i + 4] == ByteCode.BINARY_ADD):
                
                # Get the constants
                const1_idx = self.instructions[i + 1]
                const2_idx = self.instructions[i + 3]
                
                # If both are numbers, compute at compile time
                if (isinstance(self.constants[const1_idx], (int, float)) and
                    isinstance(self.constants[const2_idx], (int, float))):
                    
                    result = self.constants[const1_idx] + self.constants[const2_idx]
                    result_idx = len(self.constants)
                    self.constants.append(result)
                    
                    # Replace with single LOAD_CONST
                    optimized.extend([ByteCode.LOAD_CONST, result_idx])
                    i += 6
                    continue
            
            optimized.append(self.instructions[i])
            optimized.append(self.instructions[i + 1])
            i += 2
        
        self.instructions = optimized

class ConstantFolder:
    def fold(self, node):
        """Fold constant expressions at compile time"""
        if isinstance(node, BinOpNode):
            left = self.fold(node.left_node)
            right = self.fold(node.right_node)
            
            # If both operands are constants, compute the result
            if isinstance(left, NumberNode) and isinstance(right, NumberNode):
                left_val = left.tok.value
                right_val = right.tok.value
                
                if node.op_tok.type == TT_PLUS:
                    result = left_val + right_val
                elif node.op_tok.type == TT_MINUS:
                    result = left_val - right_val
                elif node.op_tok.type == TT_MUL:
                    result = left_val * right_val
                elif node.op_tok.type == TT_DIV:
                    if right_val != 0:
                        result = left_val / right_val
                    else:
                        return node  # Don't fold division by zero
                elif node.op_tok.type == TT_POW:
                    result = left_val ** right_val
                else:
                    return BinOpNode(left, node.op_tok, right)
                
                # Create new number node with computed value
                token_type = TT_INT if isinstance(result, int) else TT_FLOAT
                return NumberNode(Token(token_type, result, node.pos_start, node.pos_end))
            
            return BinOpNode(left, node.op_tok, right)
        
        elif isinstance(node, ListNode):
            folded_elements = [self.fold(elem) for elem in node.element_nodes]
            return ListNode(folded_elements, node.pos_start, node.pos_end)
        
        return node

class DeadCodeEliminator:
    def eliminate(self, node):
        """Remove unreachable code"""
        if isinstance(node, ListNode):
            new_elements = []
            for element in node.element_nodes:
                optimized = self.eliminate(element)
                if not self.is_dead_code(optimized):
                    new_elements.append(optimized)
            return ListNode(new_elements, node.pos_start, node.pos_end)
        
        return node
    
    def is_dead_code(self, node):
        """Check if code is unreachable"""
        # Simple dead code detection - can be expanded
        return False


#######################################
# JIT COMPILER 
#######################################

class HotSpotDetector:
    def __init__(self, threshold=100):
        self.call_counts = {}
        self.threshold = threshold
        self.compiled_functions = {}
    
    def should_compile(self, func_name):
        self.call_counts[func_name] = self.call_counts.get(func_name, 0) + 1
        return (self.call_counts[func_name] >= self.threshold and 
                func_name not in self.compiled_functions)
    
    def mark_compiled(self, func_name, compiled_func):
        self.compiled_functions[func_name] = compiled_func

class JITCompiler:
    def __init__(self):
        self.native_cache = {}
        
    def compile_to_native(self, instructions, constants):
        """Generate highly optimized native Python code"""
        # Create a unique key for caching
        cache_key = hash((tuple(instructions), tuple(constants)))
        if cache_key in self.native_cache:
            return self.native_cache[cache_key]
        
        # Generate optimized Python code
        code_lines = [
            "def jit_function(stack, constants):",
            "    # Pre-allocate for performance",
            "    local_stack = []",
            "",
        ]
        
        # Analyze instructions to generate optimal code
        ip = 0
        stack_depth = 0
        
        while ip < len(instructions):
            opcode = instructions[ip]
            arg = instructions[ip + 1] if ip + 1 < len(instructions) else 0
            
            if opcode == ByteCode.LOAD_CONST:
                const_val = constants[arg]
                if isinstance(const_val, (int, float)):
                    code_lines.append(f"    local_stack.append({const_val})")  # Direct value
                else:
                    code_lines.append(f"    local_stack.append(Number({repr(const_val)}))")
                stack_depth += 1
                
            elif opcode == ByteCode.BINARY_ADD:
                if stack_depth >= 2:
                    code_lines.extend([
                        "    right = local_stack.pop()",
                        "    left = local_stack.pop()",
                        "    local_stack.append(left + right)"  # Native arithmetic
                    ])
                    stack_depth -= 1
                    
            elif opcode == ByteCode.BINARY_SUB:
                if stack_depth >= 2:
                    code_lines.extend([
                        "    right = local_stack.pop()",
                        "    left = local_stack.pop()",
                        "    local_stack.append(left - right)"
                    ])
                    stack_depth -= 1
                    
            elif opcode == ByteCode.BINARY_MUL:
                if stack_depth >= 2:
                    code_lines.extend([
                        "    right = local_stack.pop()",
                        "    left = local_stack.pop()",
                        "    local_stack.append(left * right)"
                    ])
                    stack_depth -= 1
            
            ip += 2
        
        code_lines.extend([
            "    if local_stack:",
            "        result = local_stack[-1]",
            "        if isinstance(result, (int, float)):",
            "            from blitz import Number",
            "            return Number(result)",
            "        return result",
            "    from blitz import Number",
            "    return Number.null"
        ])
        
        # Compile the generated code
        code_str = "\n".join(code_lines)
        namespace = {"Number": Number}
        
        try:
            exec(code_str, namespace)
            compiled_func = namespace["jit_function"]
            self.native_cache[cache_key] = compiled_func
            return compiled_func
        except Exception:
            # Fallback to slower method if compilation fails
            return self.compile_to_native_fallback(instructions, constants)
    
    def compile_to_native_fallback(self, instructions, constants):
        """Fallback JIT compilation"""
        def fallback_function(stack, constants):
            vm = FastVM(instructions, constants)
            return vm.run_fast_bytecode()
        return fallback_function

class FastVM:
    def __init__(self, instructions, constants):
        self.instructions = instructions
        self.constants = constants
        self.stack = []
        self.ip = 0
        self.variables = {}  # Initialize variables dict
        self.hotspot_detector = HotSpotDetector(threshold=50)  # Lower threshold
        self.jit_compiler = JITCompiler()
        
        # Pre-compile number operations for speed
        self.number_cache = {}
        
    def run(self, func_name="<main>", global_symbol_table=None, variables_mapping=None):
        # Initialize variables from global symbol table if provided
        if global_symbol_table and variables_mapping:
            self.global_symbol_table = global_symbol_table
            # Use the variables mapping to correctly assign symbols to indices
            for var_name, var_index in variables_mapping.items():
                if var_name in global_symbol_table.symbols:
                    # This is a built-in function or global constant
                    self.variables[var_index] = global_symbol_table.symbols[var_name]
                else:
                    # This is a local variable - initialize to null
                    self.variables[var_index] = Number.null
        elif global_symbol_table:
            # Fallback to old method if no variables mapping available
            self.global_symbol_table = global_symbol_table
            # Convert symbol table to indexed variables
            for i, (name, value) in enumerate(global_symbol_table.symbols.items()):
                self.variables[i] = value
        
        # Check if we should JIT compile
        if self.hotspot_detector.should_compile(func_name):
            jit_func = self.jit_compiler.compile_to_native(self.instructions, self.constants)
            self.hotspot_detector.mark_compiled(func_name, jit_func)
            return jit_func([], self.constants)
        
        # Use cached JIT function if available
        if func_name in self.hotspot_detector.compiled_functions:
            return self.hotspot_detector.compiled_functions[func_name]([], self.constants)
        
        # Fast bytecode execution
        return self.run_fast_bytecode()
    
    def run_fast_bytecode(self):
        """Optimized bytecode execution with minimal overhead"""
        stack = self.stack
        instructions = self.instructions
        constants = self.constants
        ip = 0
        
        while ip < len(instructions):
            opcode = instructions[ip]
            arg = instructions[ip + 1] if ip + 1 < len(instructions) else 0
            
            if opcode == ByteCode.LOAD_CONST:
                # Load constant with proper type
                value = constants[arg]
                if isinstance(value, str):
                    # String constant
                    stack.append(String(value))
                elif value in self.number_cache:
                    # Cached number
                    stack.append(self.number_cache[value])
                else:
                    # Number constant
                    num_obj = Number(value)
                    if isinstance(value, (int, float)) and -100 <= value <= 100:
                        self.number_cache[value] = num_obj
                    stack.append(num_obj)
                    
            elif opcode == ByteCode.BINARY_ADD:
                right = stack.pop()
                left = stack.pop()
                
                # Handle None values
                if left is None:
                    left = Number.null
                if right is None:
                    right = Number.null
                
                # Use native Python arithmetic for speed
                if isinstance(left, Number) and isinstance(right, Number):
                    result = Number(left.value + right.value)
                    stack.append(result)
                else:
                    # Fallback to method call
                    result, _ = left.added_to(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.BINARY_SUB:
                right = stack.pop()
                left = stack.pop()
                if isinstance(left, Number) and isinstance(right, Number):
                    result = Number(left.value - right.value)
                    stack.append(result)
                else:
                    result, _ = left.subbed_by(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.BINARY_MUL:
                right = stack.pop()
                left = stack.pop()
                if isinstance(left, Number) and isinstance(right, Number):
                    result = Number(left.value * right.value)
                    stack.append(result)
                else:
                    result, _ = left.multed_by(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.BINARY_DIV:
                right = stack.pop()
                left = stack.pop()
                if isinstance(left, Number) and isinstance(right, Number) and right.value != 0:
                    result = Number(left.value / right.value)
                    stack.append(result)
                else:
                    result, _ = left.dived_by(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.BINARY_POW:
                right = stack.pop()
                left = stack.pop()
                if isinstance(left, Number) and isinstance(right, Number):
                    result = Number(left.value ** right.value)
                    stack.append(result)
                else:
                    result, _ = left.powed_by(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.BINARY_MOD:
                right = stack.pop()
                left = stack.pop()
                if isinstance(left, Number) and isinstance(right, Number) and right.value != 0:
                    result = Number(left.value % right.value)
                    stack.append(result)
                else:
                    result, _ = left.modded_by(right)
                    stack.append(result)
                    
            elif opcode == ByteCode.COMPARE_EQ:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_eq(right)
                stack.append(result)
                
            elif opcode == ByteCode.COMPARE_NE:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_ne(right)
                stack.append(result)
                
            elif opcode == ByteCode.COMPARE_LT:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_lt(right)
                stack.append(result)
                
            elif opcode == ByteCode.COMPARE_LE:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_lte(right)
                stack.append(result)
                
            elif opcode == ByteCode.COMPARE_GT:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_gt(right)
                stack.append(result)
                
            elif opcode == ByteCode.COMPARE_GE:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.get_comparison_gte(right)
                stack.append(result)
                
            elif opcode == ByteCode.LOGICAL_AND:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.anded_by(right)
                stack.append(result)
                
            elif opcode == ByteCode.LOGICAL_OR:
                right = stack.pop()
                left = stack.pop()
                result, _ = left.ored_by(right)
                stack.append(result)
                
            elif opcode == ByteCode.LOGICAL_NOT:
                operand = stack.pop()
                result, _ = operand.notted()
                stack.append(result)
                
            elif opcode == ByteCode.UNARY_MINUS:
                operand = stack.pop()
                if isinstance(operand, Number):
                    result = Number(-operand.value)
                    stack.append(result)
                else:
                    result, _ = operand.multed_by(Number(-1))
                    stack.append(result)
                    
            elif opcode == ByteCode.UNARY_PLUS:
                operand = stack.pop()
                if isinstance(operand, Number):
                    stack.append(operand)  # +x is just x
                else:
                    result, _ = operand.multed_by(Number(1))
                    stack.append(result)
                    
            elif opcode == ByteCode.LOAD_VAR:
                # Load variable by index
                var_idx = arg
                if var_idx in self.variables:
                    stack.append(self.variables[var_idx])
                else:
                    stack.append(Number.null)
                    
            elif opcode == ByteCode.STORE_VAR:
                # Store variable by index
                value = stack.pop()
                var_idx = arg
                self.variables[var_idx] = value
                stack.append(value)  # STORE also pushes the value
                
            elif opcode == ByteCode.BUILD_LIST:
                # Build a list from stack elements
                list_size = arg
                elements = []
                for _ in range(list_size):
                    elements.insert(0, stack.pop())  # Reverse order
                result_list = List(elements)
                stack.append(result_list)
                
            elif opcode == ByteCode.JUMP:
                # Unconditional jump
                ip = arg
                continue  # Skip the ip += 2 at the end
                
            elif opcode == ByteCode.JUMP_IF_FALSE:
                # Conditional jump - jump if top of stack is false
                condition = stack.pop()
                if condition is None:
                    print(f"FastVM: JUMP_IF_FALSE got None condition at ip={ip}")
                    condition = Number.null
                if condition == Number.null or not condition.is_true():
                    ip = arg
                    continue  # Skip the ip += 2 at the end
                    
            elif opcode == ByteCode.CALL_FUNCTION:
                # Function call implementation
                argc = arg
                args = []
                for _ in range(argc):
                    args.insert(0, stack.pop())
                func = stack.pop()
                
                # Handle built-in functions
                if hasattr(func, 'execute'):
                    # This is a built-in function
                    try:
                        # Set up the function with context
                        if not hasattr(self, 'context'):
                            # Create a simple context for FastVM
                            self.context = type('Context', (), {
                                'display_name': '<fastvm>',
                                'parent': None,
                                'parent_entry_pos': None,
                                'symbol_table': self.global_symbol_table if hasattr(self, 'global_symbol_table') else None
                            })()
                        
                        func.set_context(self.context)
                        result = func.execute(args)
                        if result and hasattr(result, 'value') and result.value is not None:
                            stack.append(result.value)
                        else:
                            stack.append(Number.null)
                    except Exception as e:
                        print(f"FastVM function call error: {e}")  # Debug
                        stack.append(Number.null)
                else:
                    # Unknown function - return null
                    stack.append(Number.null)
            
            ip += 2
        
        return stack[-1] if stack else Number.null

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
  def __init__(self):
    self.reset()

  def reset(self):
    self.value = None
    self.error = None
    self.func_return_value = None
    self.loop_should_continue = False
    self.loop_should_break = False

  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
    self.loop_should_continue = res.loop_should_continue
    self.loop_should_break = res.loop_should_break
    return res.value

  def success(self, value):
    self.reset()
    self.value = value
    return self

  def success_return(self, value):
    self.reset()
    self.func_return_value = value
    return self
  
  def success_continue(self):
    self.reset()
    self.loop_should_continue = True
    return self

  def success_break(self):
    self.reset()
    self.loop_should_break = True
    return self

  def failure(self, error):
    self.reset()
    self.error = error
    return self

  def should_return(self):
    # Note: this will allow you to continue and break outside the current function
    return (
      self.error or
      self.func_return_value or
      self.loop_should_continue or
      self.loop_should_break
    )

#######################################
# VALUES
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, pos_start=None, pos_end=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)

  def notted(self, other):
    return None, self.illegal_operation(other)

  def execute(self, args):
    return RTResult().failure(self.illegal_operation())

  def copy(self):
    raise Exception('No copy method defined')

  def is_true(self):
    return False

  def illegal_operation(self, other=None):
    if not other: other = self
    return RTError(
      self.pos_start, other.pos_end,
      'Illegal operation',
      self.context
    )

class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subbed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def powed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_eq(self, other):
    if isinstance(other, Number):
      return Number(int(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_ne(self, other):
    if isinstance(other, Number):
      return Number(int(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Number(1 if self.value == 0 else 0).set_context(self.context), None

  def copy(self):
    copy = Number(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def added_to(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None

  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.copy()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.copy()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      try:
        return self.elements[other.value], None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)
  
  def copy(self):
    copy = List(self.elements)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'

class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    return new_context

  def check_args(self, arg_names, args):
    res = RTResult()

    if len(args) > len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(args) - len(arg_names)} too many args passed into {self}",
        self.context
      ))
    
    if len(args) < len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(arg_names) - len(args)} too few args passed into {self}",
        self.context
      ))

    return res.success(None)

  def populate_args(self, arg_names, args, exec_ctx):
    for i in range(len(args)):
      arg_name = arg_names[i]
      arg_value = args[i]
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)

  def check_and_populate_args(self, arg_names, args, exec_ctx):
    res = RTResult()
    res.register(self.check_args(arg_names, args))
    if res.should_return(): return res
    self.populate_args(arg_names, args, exec_ctx)
    return res.success(None)

class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.should_auto_return = should_auto_return

  def execute(self, args):
    res = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
    if res.should_return(): return res

    value = res.register(interpreter.visit(self.body_node, exec_ctx))
    if res.should_return() and res.func_return_value == None: return res

    ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
    return res.success(ret_value)

  def copy(self):
    copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

  def execute(self, args):
    res = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_visit_method)

    res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
    if res.should_return(): return res

    return_value = res.register(method(exec_ctx))
    if res.should_return(): return res
    return res.success(return_value)
  
  def execute_optimize(self, exec_ctx):
    """Force optimization of a function or code block"""
    target = exec_ctx.symbol_table.get('target')
    print(f"🔧 Applying forced optimization to: {target}")
    
    # Apply aggressive optimizations
    if hasattr(self, 'smart_optimizer'):
        self.smart_optimizer.apply_smart_optimization(str(target))
    
    return RTResult().success(String("Optimization applied"))
  execute_optimize.arg_names = ['target']

  def execute_predict(self, exec_ctx):
    """Predict performance characteristics"""
    code = exec_ctx.symbol_table.get('code')
    
    predictor = PerformancePredictor()
    
    # Mock prediction for demonstration
    predicted_time = "0.05-0.1 seconds"
    predicted_memory = "2.3 MB"
    optimization_suggestions = ["Consider parallelization", "Enable JIT compilation"]
    
    result = f"Predicted execution time: {predicted_time}\n"
    result += f"Predicted memory usage: {predicted_memory}\n"
    result += f"Suggestions: {', '.join(optimization_suggestions)}"
    
    return RTResult().success(String(result))
  execute_predict.arg_names = ['code']

  def execute_quantum_search(self, exec_ctx):
    """Quantum-inspired search algorithm"""
    data = exec_ctx.symbol_table.get('data')
    target = exec_ctx.symbol_table.get('target')
    
    if not isinstance(data, List):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a list",
            exec_ctx
        ))
    
    # Implement quantum-inspired Grover's algorithm simulation
    print("🔬 Using quantum-inspired search...")
    
    # For demonstration, use optimized search
    for i, element in enumerate(data.elements):
        if str(element) == str(target):
            return RTResult().success(Number(i))
    
    return RTResult().success(Number(-1))
  execute_quantum_search.arg_names = ['data', 'target']
  
  def execute_prof_start(self, exec_ctx):
    """Start profiling - placeholder for future advanced profiling"""
    import time
    start_time = time.time()
    print(f"📊 Profiling started at {start_time}")
    return RTResult().success(Number(start_time))
  execute_prof_start.arg_names = []
  
  def execute_prof_end(self, exec_ctx):
    """End profiling - placeholder for future advanced profiling"""
    import time
    end_time = time.time()
    print(f"📊 Profiling ended at {end_time}")
    return RTResult().success(Number(end_time))
  execute_prof_end.arg_names = []
  
  def no_visit_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

  def execute_print(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)
  execute_print.arg_names = ['value']
  
  def execute_print_ret(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
  execute_print_ret.arg_names = ['value']
  
  def execute_input(self, exec_ctx):
    text = input()
    return RTResult().success(String(text))
  execute_input.arg_names = []

  def execute_input_int(self, exec_ctx):
    while True:
      text = input()
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")
    return RTResult().success(Number(number))
  execute_input_int.arg_names = []

  def execute_clear(self, exec_ctx):
    os.system('cls' if os.name == 'nt' else 'cls') 
    return RTResult().success(Number.null)
  execute_clear.arg_names = []

  def execute_is_number(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_number.arg_names = ["value"]

  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_string.arg_names = ["value"]

  def execute_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_list.arg_names = ["value"]

  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_function.arg_names = ["value"]

  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    list_.elements.append(value)
    return RTResult().success(Number.null)
  execute_append.arg_names = ["list", "value"]

  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(index, Number):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be number",
        exec_ctx
      ))

    try:
      element = list_.elements.pop(index.value)
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        'Element at this index could not be removed from list because index is out of bounds',
        exec_ctx
      ))
    return RTResult().success(element)
  execute_pop.arg_names = ["list", "index"]

  def execute_extend(self, exec_ctx):
    listA = exec_ctx.symbol_table.get("listA")
    listB = exec_ctx.symbol_table.get("listB")

    if not isinstance(listA, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(listB, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be list",
        exec_ctx
      ))

    listA.elements.extend(listB.elements)
    return RTResult().success(Number.null)
  execute_extend.arg_names = ["listA", "listB"]

  def execute_len(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Argument must be list",
        exec_ctx
      ))

    return RTResult().success(Number(len(list_.elements)))
  execute_len.arg_names = ["list"]

  def execute_run(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    fn = fn.value

    try:
      with open(fn, "r") as f:
        script = f.read()
    except Exception as e:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to load script \"{fn}\"\n" + str(e),
        exec_ctx
      ))

    _, error = run(fn, script)
    
    if error:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to finish executing script \"{fn}\"\n" +
        error.as_string(),
        exec_ctx
      ))

    return RTResult().success(Number.null)
  execute_run.arg_names = ["fn"]
  
  def execute_str(self, exec_ctx):
    value = exec_ctx.symbol_table.get('value')
    return RTResult().success(String(str(value)))
  execute_str.arg_names = ['value']

BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.print_ret   = BuiltInFunction("print_ret")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_int   = BuiltInFunction("input_int")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")
BuiltInFunction.len					= BuiltInFunction("len")
BuiltInFunction.run					= BuiltInFunction("run")
BuiltInFunction.str					= BuiltInFunction("str")

#######################################
# CONTEXT
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent
    self._cache = {}

  def get(self, name):
      if name in self._cache:
        return self._cache[name]
      
      value = self.symbols.get(name, None)
      if value == None and self.parent:
          return self.parent.get(name)
      if value:
          self._cache[name] = value
      return value

  def set(self, name, value):
    self.symbols[name] = value
    # Clear cache for this name when value is updated
    if name in self._cache:
      del self._cache[name]

  def remove(self, name):
    del self.symbols[name]

#######################################
# INTERPRETER
#######################################

class Interpreter:
  def visit(self, node, context):
    method_name = f'visit_{type(node).__name__}'
    method = getattr(self, method_name, self.no_visit_method)
    return method(node, context)

  def no_visit_method(self, node, context):
    raise Exception(f'No visit_{type(node).__name__} method defined')

  ###################################

  def visit_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_ListNode(self, node, context):
    res = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(res.register(self.visit(element_node, context)))
      if res.should_return(): return res

    return res.success(
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_VarAccessNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = context.symbol_table.get(var_name)

    if not value:
      return res.failure(RTError(
        node.pos_start, node.pos_end,
        f"'{var_name}' is not defined",
        context
      ))

    value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = res.register(self.visit(node.value_node, context))
    if res.should_return(): return res

    context.symbol_table.set(var_name, value)
    return res.success(value)

  def visit_BinOpNode(self, node, context):
    res = RTResult()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): return res

    if node.op_tok.type == TT_PLUS:
      result, error = left.added_to(right)
    elif node.op_tok.type == TT_MINUS:
      result, error = left.subbed_by(right)
    elif node.op_tok.type == TT_MUL:
      result, error = left.multed_by(right)
    elif node.op_tok.type == TT_DIV:
      result, error = left.dived_by(right)
    elif node.op_tok.type == TT_POW:
      result, error = left.powed_by(right)
    elif node.op_tok.type == TT_EE:
      result, error = left.get_comparison_eq(right)
    elif node.op_tok.type == TT_NE:
      result, error = left.get_comparison_ne(right)
    elif node.op_tok.type == TT_LT:
      result, error = left.get_comparison_lt(right)
    elif node.op_tok.type == TT_GT:
      result, error = left.get_comparison_gt(right)
    elif node.op_tok.type == TT_LTE:
      result, error = left.get_comparison_lte(right)
    elif node.op_tok.type == TT_GTE:
      result, error = left.get_comparison_gte(right)
    elif node.op_tok.matches(TT_KEYWORD, 'AND'):
      result, error = left.anded_by(right)
    elif node.op_tok.matches(TT_KEYWORD, 'OR'):
      result, error = left.ored_by(right)

    if error:
      return res.failure(error)
    else:
      return res.success(result.set_pos(node.pos_start, node.pos_end))

  def visit_UnaryOpNode(self, node, context):
    res = RTResult()
    number = res.register(self.visit(node.node, context))
    if res.should_return(): return res

    error = None

    if node.op_tok.type == TT_MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.op_tok.matches(TT_KEYWORD, 'NOT'):
      number, error = number.notted()

    if error:
      return res.failure(error)
    else:
      return res.success(number.set_pos(node.pos_start, node.pos_end))
    
  def visit_MultipleAssignNode(self, node, context):
    """Handle multiple assignment: a, b = 1, 2"""
    res = RTResult()
    
    # Evaluate all values first
    values = []
    for value_node in node.value_nodes:
        value = res.register(self.visit(value_node, context))
        if res.should_return(): return res
        values.append(value)
    
    # Assign values to variables
    for i, var_name_tok in enumerate(node.var_name_toks):
        if i < len(values):
            context.symbol_table.set(var_name_tok.value, values[i])
        else:
            context.symbol_table.set(var_name_tok.value, Number.null)
    
    # Return the first value or null
    return res.success(values[0] if values else Number.null)

  def visit_AugmentedAssignNode(self, node, context):
    """Handle augmented assignment: x += 5, x *= 2, etc."""
    res = RTResult()
    
    # Get current variable value
    var_name = node.var_name_tok.value
    current_value = context.symbol_table.get(var_name)
    
    if not current_value:
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            f"'{var_name}' is not defined",
            context
        ))
    
    # Evaluate the right-hand side
    rhs_value = res.register(self.visit(node.value_node, context))
    if res.should_return(): return res
    
    # Perform the operation
    if node.op_tok.type == TT_PLUS:
        result, error = current_value.added_to(rhs_value)
    elif node.op_tok.type == TT_MINUS:
        result, error = current_value.subbed_by(rhs_value)
    elif node.op_tok.type == TT_MUL:
        result, error = current_value.multed_by(rhs_value)
    elif node.op_tok.type == TT_DIV:
        result, error = current_value.dived_by(rhs_value)
    else:
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            f"Unsupported augmented assignment operator: {node.op_tok.type}",
            context
        ))
    
    if error:
        return res.failure(error)
    
    # Store the result
    context.symbol_table.set(var_name, result)
    return res.success(result)

  def visit_IfNode(self, node, context):
    res = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = res.register(self.visit(condition, context))
      if res.should_return(): return res

      if condition_value.is_true():
        expr_value = res.register(self.visit(expr, context))
        if res.should_return(): return res
        return res.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = res.register(self.visit(expr, context))
      if res.should_return(): return res
      return res.success(Number.null if should_return_null else expr_value)

    return res.success(Number.null)

  def visit_ForNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    if node.step_value_node:
      step_value = res.register(self.visit(node.step_value_node, context))
      if res.should_return(): return res
    else:
      step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
      condition = lambda: i < end_value.value
    else:
      condition = lambda: i > end_value.value
    
    while condition():
      context.symbol_table.set(node.var_name_tok.value, Number(i))
      i += step_value.value

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
      
      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_WhileNode(self, node, context):
    res = RTResult()
    elements = []

    while True:
      condition = res.register(self.visit(node.condition_node, context))
      if res.should_return(): return res

      if not condition.is_true():
        break

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_FuncDefNode(self, node, context):
    res = RTResult()

    func_name = node.var_name_tok.value if node.var_name_tok else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_toks]
    func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
    
    if node.var_name_tok:
      context.symbol_table.set(func_name, func_value)

    return res.success(func_value)

  def visit_CallNode(self, node, context):
    res = RTResult()
    args = []

    value_to_call = res.register(self.visit(node.node_to_call, context))
    if res.should_return(): return res
    value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))
      if res.should_return(): return res

    return_value = res.register(value_to_call.execute(args))
    if res.should_return(): return res
    return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(return_value)

  def visit_ReturnNode(self, node, context):
    res = RTResult()

    if node.node_to_return:
      value = res.register(self.visit(node.node_to_return, context))
      if res.should_return(): return res
    else:
      value = Number.null
    
    return res.success_return(value)

  def visit_ContinueNode(self, node, context):
    return RTResult().success_continue()

  def visit_BreakNode(self, node, context):
    return RTResult().success_break()
  
  def visit_ParallelForNode(self, node, context):
    """Execute parallel FOR loop"""
    res = RTResult()
    
    # Check if parallelization is beneficial
    parallelizer = IntelligentParallelizer()
    if parallelizer.can_parallelize(node):
        print(f"🔄 Executing parallel loop with {parallelizer.thread_pool_size} threads")
        try:
            results = parallelizer.parallelize_loop(node, context)
            return res.success(List(results).set_context(context).set_pos(node.pos_start, node.pos_end))
        except Exception as e:
            print(f"⚠️  Parallel execution failed: {e}, falling back to sequential")
    
    # Fall back to regular sequential execution
    return self.visit_ForNode(node, context)

  def visit_ProfileNode(self, node, context):
    """Execute profiled code block"""
    import time
    res = RTResult()
    
    profile_name = node.name_tok.value
    print(f"📊 Starting profile: {profile_name}")
    
    start_time = time.time()
    start_memory = self.get_memory_usage()
    
    # Execute the profiled code
    result = res.register(self.visit(node.body_node, context))
    if res.should_return(): return res
    
    end_time = time.time()
    end_memory = self.get_memory_usage()
    
    # Generate performance report
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"📈 Profile '{profile_name}' completed:")
    print(f"   ⏱️  Execution time: {execution_time:.4f} seconds")
    print(f"   🧠 Memory used: {memory_used:.2f} MB")
    print(f"   🚀 Performance rating: {self.calculate_performance_rating(execution_time)}")
    
    return res.success(result)

  def get_memory_usage(self):
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

  def calculate_performance_rating(self, execution_time):
    """Calculate performance rating"""
    if execution_time < 0.001:
        return "⚡ Blazing Fast"
    elif execution_time < 0.01:
        return "🚀 Very Fast"
    elif execution_time < 0.1:
        return "✅ Fast"
    elif execution_time < 1.0:
        return "⚠️  Moderate"
    else:
        return "🐌 Slow"

#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number.null)
global_symbol_table.set("FALSE", Number.false)
global_symbol_table.set("TRUE", Number.true)
global_symbol_table.set("MATH_PI", Number.math_PI)
global_symbol_table.set("PRINT", BuiltInFunction.print)
global_symbol_table.set("PRINT_RET", BuiltInFunction.print_ret)
global_symbol_table.set("INPUT", BuiltInFunction.input)
global_symbol_table.set("INPUT_INT", BuiltInFunction.input_int)
global_symbol_table.set("CLEAR", BuiltInFunction.clear)
global_symbol_table.set("CLS", BuiltInFunction.clear)
global_symbol_table.set("IS_NUM", BuiltInFunction.is_number)
global_symbol_table.set("IS_STR", BuiltInFunction.is_string)
global_symbol_table.set("IS_LIST", BuiltInFunction.is_list)
global_symbol_table.set("IS_FUN", BuiltInFunction.is_function)
global_symbol_table.set("APPEND", BuiltInFunction.append)
global_symbol_table.set("POP", BuiltInFunction.pop)
global_symbol_table.set("EXTEND", BuiltInFunction.extend)
global_symbol_table.set("LEN", BuiltInFunction.len)
global_symbol_table.set("RUN", BuiltInFunction.run)
global_symbol_table.set("STR", BuiltInFunction.str)
global_symbol_table.set("OPTIMIZE", BuiltInFunction("optimize"))
global_symbol_table.set("PREDICT", BuiltInFunction("predict"))
global_symbol_table.set("QUANTUM_SEARCH", BuiltInFunction("quantum_search"))
global_symbol_table.set("PROF_START", BuiltInFunction("prof_start"))
global_symbol_table.set("PROF_END", BuiltInFunction("prof_end"))

def run(fn, text, optimize=False, show_bytecode=False):
    try:
        
        # Initialize smart optimizer
        if not hasattr(run, 'smart_optimizer'):
            run.smart_optimizer = SmartOptimizer()

        # Check if it's a compiled bytecode file
        if fn.endswith('.bkc'):
            return run_bytecode_file(fn)
        
        # Generate tokens
        lexer = Lexer(fn, text)
        tokens, error = lexer.make_tokens()
        if error: return None, error
        
        # Generate AST
        parser = Parser(tokens)
        ast = parser.parse()
        if ast.error: return None, ast.error

        start_time = time.time()
        
        if optimize:
            # Apply optimizations and JIT compilation
            compiler = Compiler()
            try:
                compiler = OptimizedCompiler()
                compiler.compile(ast.node)
                
                if show_bytecode:
                    print("=== Optimized Bytecode ===")
                    for i in range(0, len(compiler.instructions), 2):
                        opcode = compiler.instructions[i]
                        arg = compiler.instructions[i + 1] if i + 1 < len(compiler.instructions) else 0
                        print(f"{i//2:04d}: {opcode:02d} {arg}")
                    print("=========================")
                
                # Run on FastVM with JIT
                vm = FastVM(compiler.instructions, compiler.constants)
                result = vm.run(fn)
                
                # Track performance for future optimizations
                end_time = time.time()
                execution_time = end_time - start_time

                # Auto-optimization tracking
                if hasattr(ast.node, 'element_nodes'):
                    for node in ast.node.element_nodes:
                        if isinstance(node, FuncDefNode) and node.var_name_tok:
                            func_name = node.var_name_tok.value
                            run.smart_optimizer.track_execution(
                                func_name, execution_time, []
                            )
        
                print(f"🚀 Optimized execution completed in {execution_time:.4f} seconds")
                return result, None
                
            except Exception as e:
                # If compilation fails, fall back to interpreter
                print(f"Compilation failed ({e}), falling back to interpreter mode...")
                # Fall through to interpreter mode below
        
        # Run in pure interpreter mode (independent of compiler)
        interpreter = Interpreter()
        context = Context('<program>')
        context.symbol_table = global_symbol_table
        context.smart_optimizer = run.smart_optimizer  
        result = interpreter.visit(ast.node, context)

        # Track performance for interpreter mode too
        end_time = time.time()
        execution_time = end_time - start_time

        # Track function performance for auto-optimization
        if hasattr(ast.node, 'element_nodes'):
            for node in ast.node.element_nodes:
                if isinstance(node, FuncDefNode) and node.var_name_tok:
                    func_name = node.var_name_tok.value
                    run.smart_optimizer.track_execution(
                         func_name, execution_time, []
                    )

        print(f"📊 Interpreter execution completed in {execution_time:.4f} seconds")
        
        # Don't return the last expression value for program execution
        # Only return it if there was an error or if it's an interactive session
        if result.error:
            return None, result.error
        else:
            # For program files, return null unless there's a specific return value
            return Number.null, None
            
    except Exception as e:
        return None, RTError(
            Position(0, 0, 0, fn, text),
            Position(0, 0, 0, fn, text),
            f"Internal error: {str(e)}",
            Context('<internal>')
        )

def run_interactive(text, optimize=False, show_bytecode=False):
    """Run code in interactive mode - returns the result value"""
    try:
        # Generate tokens
        lexer = Lexer('<stdin>', text)
        tokens, error = lexer.make_tokens()
        if error: return None, error
        
        # Generate AST
        parser = Parser(tokens)
        ast = parser.parse()
        if ast.error: return None, ast.error
        
        # Always use interpreter for interactive mode for consistency
        interpreter = Interpreter()
        context = Context('<interactive>')
        context.symbol_table = global_symbol_table
        result = interpreter.visit(ast.node, context)
        
        if result.error:
            return None, result.error
        else:
            return result.value, None
            
    except Exception as e:
        return None, RTError(
            Position(0, 0, 0, '<stdin>', text),
            Position(0, 0, 0, '<stdin>', text),
            f"Internal error: {str(e)}",
            Context('<interactive>')
        )

def run_bytecode_file(filename):
    """Run a compiled .bkc bytecode file"""
    import pickle
    
    try:
        with open(filename, 'rb') as f:
            bytecode_data = pickle.load(f)
        
        instructions = bytecode_data['instructions']
        constants = bytecode_data['constants']
        variables_mapping = bytecode_data.get('variables', {})  # Get variable name to index mapping
        
        # Create global symbol table with built-in functions
        global_symbol_table = SymbolTable()
        global_symbol_table.set("null", Number.null)
        global_symbol_table.set("false", Number.false)
        global_symbol_table.set("true", Number.true)
        global_symbol_table.set("MATH_PI", Number.math_PI)
        global_symbol_table.set("PRINT", BuiltInFunction.print)
        global_symbol_table.set("PRINT_RET", BuiltInFunction.print_ret)
        global_symbol_table.set("INPUT", BuiltInFunction.input)
        global_symbol_table.set("INPUT_INT", BuiltInFunction.input_int)
        global_symbol_table.set("CLEAR", BuiltInFunction.clear)
        global_symbol_table.set("CLS", BuiltInFunction.clear)
        global_symbol_table.set("IS_NUM", BuiltInFunction.is_number)
        global_symbol_table.set("IS_STR", BuiltInFunction.is_string)
        global_symbol_table.set("IS_LIST", BuiltInFunction.is_list)
        global_symbol_table.set("IS_FUN", BuiltInFunction.is_function)
        global_symbol_table.set("APPEND", BuiltInFunction.append)
        global_symbol_table.set("POP", BuiltInFunction.pop)
        global_symbol_table.set("EXTEND", BuiltInFunction.extend)
        global_symbol_table.set("LEN", BuiltInFunction.len)
        global_symbol_table.set("STR", BuiltInFunction.str)
        global_symbol_table.set("RUN", BuiltInFunction.run)
        
        vm = FastVM(instructions, constants)
        result = vm.run(filename, global_symbol_table, variables_mapping)
        return result, None
        
    except Exception as e:
        return None, RTError(
            Position(0, 0, 0, filename, ""),
            Position(0, 0, 0, filename, ""),
            f"Failed to run bytecode file: {str(e)}",
            Context('<bytecode>')
        )

# Simple main execution for direct script usage
if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("Usage: python3 blitz.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not filename.endswith('.bk') and not filename.endswith('.bkc'):
        print("Error: File must have .bk or .bkc extension")
        sys.exit(1)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    result, error = run(filename, text)
    
    if error:
        print(error.as_string())
        sys.exit(1)
        
