import subprocess
import sys

def run_apt_command(command_args, description):
    print(f"\n--- Running: {description} ---")
    try:
        process = subprocess.run(
            command_args,
            check=True,
            text=True,
            capture_output=True,
            shell=False
        )
        print(process.stdout)
        if process.stderr:
            print(f"--- {description} (stderr) ---")
            print(process.stderr)
        print(f"--- {description} completed successfully. ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: {description} failed with exit code {e.returncode} ---")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"--- ERROR: Command '{command_args[0]}' not found. Is apt installed and in PATH? ---")
        return False
    except Exception as e:
        print(f"--- An unexpected error occurred during {description}: {e} ---")
        return False

def main():
    print("Starting system update and upgrade process...")

    if not run_apt_command(['sudo', 'apt', 'update'], "sudo apt update"):
        print("apt update failed. Aborting upgrade.")
        sys.exit(1)

    if not run_apt_command(['sudo', 'apt', 'upgrade', '-y'], "sudo apt upgrade -y"):
        print("apt upgrade failed.")
        sys.exit(1)

    print("\nSystem update and upgrade process completed.")

if __name__ == "__main__":
    main()
