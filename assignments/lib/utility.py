def inspect_variable(var, var_name="variable"):
    print(f"=== {var_name} ===")
    print(f"Class: {var.__class__}")
    print(f"Class Name: {var.__class__.__name__}")
    print(f"Type: {type(var)}")
    print(f"Type name: {type(var).__name__}")
    print(f"String representation: {str(var)[:100]}...")
    if hasattr(var, 'shape'):
        print(f"Shape: {var.shape}")
    print()


