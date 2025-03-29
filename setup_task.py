
"""
Setup Script for Windows Task Scheduler
This script creates a scheduled task that runs the auto-login script when the workstation is unlocked.
It also creates and stores an encrypted PIN file for secure storage.
Author: Vedant Korade
"""

import os
import subprocess
import sys
import getpass
from cryptography.fernet import Fernet
import ctypes

def is_admin():
    """Check if the script is running with administrative privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def create_directories():
    """Create necessary directories for scripts and logs"""
    # Create directory for scripts and logs
    script_dir = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'Documents', 'FaceUnlock')
    os.makedirs(script_dir, exist_ok=True)
    return script_dir

def generate_encryption_key(script_dir):
    """Generate and save an encryption key for the PIN"""
    key_file = os.path.join(script_dir, '.key')
    
    # Only generate a new key if one doesn't exist
    if not os.path.exists(key_file):
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions on the key file
        os.chmod(key_file, 0o600)  # Only owner can read/write
    
    # Read the key from file
    with open(key_file, 'rb') as f:
        return f.read()

def encrypt_pin(pin, key, script_dir):
    """Encrypt the PIN and save it to a file"""
    fernet = Fernet(key)
    encrypted_pin = fernet.encrypt(pin.encode())
    
    pin_file = os.path.join(script_dir, '.pin.encrypted')
    with open(pin_file, 'wb') as f:
        f.write(encrypted_pin)
    
    # Set restrictive permissions on the PIN file
    os.chmod(pin_file, 0o600)  # Only owner can read/write

def create_scheduled_task(script_path, task_name="FaceUnlockLogin"):
    """Create a Windows scheduled task that runs on workstation unlock"""
    # Build the XML for the task
    xml_content = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2023-01-01T00:00:00</Date>
    <Author>{os.environ.get('USERNAME', 'User')}</Author>
    <Description>Automatically enters PIN when workstation is unlocked and face is detected</Description>
  </RegistrationInfo>
  <Triggers>
    <SessionStateChangeTrigger>
      <StartBoundary>2023-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <StateChange>SessionUnlock</StateChange>
    </SessionStateChangeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{os.environ.get('USERDOMAIN', '')}\{os.environ.get('USERNAME', '')}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>pythonw</Command>
      <Arguments>"{script_path}"</Arguments>
    </Exec>
  </Actions>
</Task>
"""
    
    # Save the XML to a temporary file
    xml_path = os.path.join(os.environ.get('TEMP', ''), 'face_unlock_task.xml')
    with open(xml_path, 'w', encoding='utf-16') as f:
        f.write(xml_content)
    
    # Create the task using schtasks
    try:
        subprocess.run(
            ['schtasks', '/create', '/tn', task_name, '/xml', xml_path, '/f'], 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Task '{task_name}' created successfully")
        
        # Clean up the temporary XML file
        os.remove(xml_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating task: {e.stderr}")
        return False

def set_environment_variable(var_name, value):
    """Set a user environment variable (alternative to file storage)"""
    try:
        # Use setx to set a user environment variable
        subprocess.run(
            ['setx', var_name, value],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Environment variable {var_name} set successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting environment variable: {e.stderr}")
        return False

def main():
    """Main setup function"""
    if not is_admin():
        print("This script needs administrative privileges to create a scheduled task.")
        print("Please run this script as administrator.")
        return
    
    # Create necessary directories
    script_dir = create_directories()
    
    # Copy the auto-login script to the script directory
    # Get path of current script
    current_script = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script)
    
    # Copy the auto-login script from the current directory
    auto_login_source = os.path.join(current_dir, "auto_login.py")
    auto_login_dest = os.path.join(script_dir, "auto_login.py")
    
    if os.path.exists(auto_login_source):
        with open(auto_login_source, 'r') as src, open(auto_login_dest, 'w') as dest:
            dest.write(src.read())
        print(f"Auto-login script copied to {auto_login_dest}")
    else:
        print("Please create the auto_login.py script in the same directory as this setup script")
        return
    
    # Ask for PIN
    pin = getpass.getpass("Enter your Windows PIN: ")
    if not pin:
        print("PIN cannot be empty")
        return
    
    # Store PIN securely
    print("\nChoose how to store your PIN:")
    print("1. As an encrypted file (recommended)")
    print("2. As an environment variable")
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        # Generate encryption key and encrypt PIN
        key = generate_encryption_key(script_dir)
        encrypt_pin(pin, key, script_dir)
        print("PIN encrypted and saved successfully")
    elif choice == "2":
        # Set environment variable
        success = set_environment_variable("FACE_UNLOCK_PIN", pin)
        if not success:
            print("Failed to set environment variable. Falling back to encrypted file storage.")
            key = generate_encryption_key(script_dir)
            encrypt_pin(pin, key, script_dir)
    else:
        print("Invalid choice. Exiting...")
        return
    
    # Create scheduled task
    create_scheduled_task(auto_login_dest)
    
    print("\nSetup complete! The system will now automatically enter your PIN when you unlock your workstation.")
    print(f"Logs will be stored in {script_dir}\\auto_login.log")

if __name__ == "__main__":
    main()