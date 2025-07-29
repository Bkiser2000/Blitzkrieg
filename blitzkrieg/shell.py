#!/usr/bin/env python3
# filepath: /mnt/Basefiles/blitzkrieg/shell.py
import sys
import os
import argparse
from pathlib import Path
from blitz import run, run_interactive, Number

def main():
    parser = argparse.ArgumentParser(description='Blitzkrieg Programming Language')
    parser.add_argument('file', nargs='?', help='Blitz file to execute (.bk)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-v', '--version', action='version', version='Blitzkrieg 1.0.0')
    parser.add_argument('--no-optimize', action='store_true', help='Disable optimizations')
    parser.add_argument('--interpreter', action='store_true', help='Force pure interpreter mode (no bytecode/JIT)')
    parser.add_argument('--bytecode', action='store_true', help='Show bytecode')
    parser.add_argument('-c', '--compile', action='store_true', help='Compile to bytecode file')
    
    args = parser.parse_args()
    
    if args.file:
        run_file(args.file, args)
    elif args.interactive:
        interactive_shell()
    else:
        interactive_shell()

def run_file(filename, args):
    try:
        # Check file extension
        if filename.endswith('.bkc'):
            # Execute compiled bytecode file
            run_bytecode_file(filename, args)
            return
        elif not filename.endswith('.bk'):
            print(f"Error: Expected .bk or .bkc file, got {filename}")
            sys.exit(1)
            
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            sys.exit(1)
            
        # Read and execute file
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Compile to bytecode file if requested
        if args.compile:
            compile_to_bytecode(filename, text, args)
            return
        
        # Determine execution mode
        if args.interpreter:
            # Force pure interpreter mode (disable optimizations)
            optimize = False
        else:
            # Use optimization flags
            optimize = not args.no_optimize  # Default to optimized
            
        result, error = run(filename, text, optimize=optimize, show_bytecode=args.bytecode)
        
        if error:
            print(error.as_string())
            sys.exit(1)
        # Don't print result for file execution - only for interactive mode
        # Results are printed by PRINT statements within the program
            
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Internal error: {e}")
        sys.exit(1)

def compile_to_bytecode(filename, text, args):
    """Compile .bk file to .bkc bytecode file"""
    from blitz import Lexer, Parser, OptimizedCompiler
    import pickle
    
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

def run_bytecode_file(filename, args):
    """Run a compiled .bkc bytecode file"""
    from blitz import run_bytecode_file as run_bytecode
    import time
    
    try:
        print(f"Running bytecode file: {filename}")
        start_time = time.time()
        
        result, error = run_bytecode(filename)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
        
        if error:
            print(error.as_string())
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Internal error: {e}")
        sys.exit(1)

def interactive_shell():
    print("Blitz 1.0.0 Interactive Shell")
    print("Type 'exit' to quit")
    
    while True:
        try:
            text = input('blitz > ')
            
            if text.strip() == 'exit':
                break
            elif text.strip() == '':
                continue
                
            result, error = run_interactive(text)
            
            if error:
                print(error.as_string())
            elif result and result != Number.null:
                print(result)
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            break

if __name__ == '__main__':
    main()
