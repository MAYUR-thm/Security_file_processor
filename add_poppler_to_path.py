import winreg
import os
import sys

def add_to_path(new_path):
    # Open the registry key for environment variables
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
        try:
            # Get the current PATH value
            path_value, _ = winreg.QueryValueEx(key, "Path")
            
            # Check if the path is already in PATH
            if new_path not in path_value:
                # Add the new path
                new_path_value = f"{path_value};{new_path}"
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path_value)
                print(f"Successfully added {new_path} to PATH")
                # Notify the system about the change
                os.system('rundll32.exe user32.dll,UpdatePerUserSystemParameters')
            else:
                print(f"{new_path} is already in PATH")
                
        except WindowsError:
            print("Error: Could not modify the PATH environment variable")
            sys.exit(1)

if __name__ == "__main__":
    poppler_path = r"C:\poppler\Library\bin"
    add_to_path(poppler_path)