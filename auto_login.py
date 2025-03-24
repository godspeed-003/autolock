"""
Windows Auto-Login Script
This script is designed to be triggered by Windows Task Scheduler on workstation unlock.
It securely retrieves your PIN from environment variables and enters it automatically.
"""

import pyautogui
import time
import os
import sys
import logging
from datetime import datetime
import ctypes
from cryptography.fernet import Fernet

# Set up logging to help troubleshoot if needed
log_dir = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), 'Documents', 'FaceUnlock')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'auto_login.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_pin_from_env():
    """Retrieve PIN from environment variable (safest method)"""
    pin = os.environ.get("FACE_UNLOCK_PIN")
    if pin:
        return pin
    return None

def get_pin_from_encrypted_file():
    """Alternative method: retrieve PIN from encrypted file"""
    try:
        # The key file should be stored in a secure location with restricted permissions
        key_file = os.path.join(log_dir, '.key')
        pin_file = os.path.join(log_dir, '.pin.encrypted')
        
        if not os.path.exists(key_file) or not os.path.exists(pin_file):
            logging.error("Key or PIN file not found")
            return None
            
        # Read the encryption key
        with open(key_file, 'rb') as f:
            key = f.read()
            
        # Read the encrypted PIN
        with open(pin_file, 'rb') as f:
            encrypted_pin = f.read()
            
        # Decrypt the PIN
        fernet = Fernet(key)
        pin = fernet.decrypt(encrypted_pin).decode()
        return pin
        
    except Exception as e:
        logging.error(f"Failed to read PIN from encrypted file: {e}")
        return None

def is_lock_screen_visible():
    """Check if we're at the Windows lock screen/login screen"""
    # This is a simple heuristic - in a real implementation,
    # you might want to use more sophisticated methods
    
    # For Windows 10/11, get the foreground window title
    # We expect it to be either empty or contain "Windows" and "Lock"
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
    title = buff.value.lower()
    
    # Check screen resolution - lock screen often has a specific resolution
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    
    logging.info(f"Current window: '{title}', Screen: {screen_width}x{screen_height}")
    
    # Simple heuristic: If we're at the login screen, pixel at coordinate (10,10)
    # is likely to be a specific color from the login background
    # This is just an example - you'd need to adjust for your specific system
    
    return True  # For initial testing, assume we're always at the login screen

def type_pin_and_enter(pin):
    """Type the PIN and press Enter"""
    try:
        logging.info("Beginning PIN entry")
        
        # Wait for the login screen to be fully ready
        time.sleep(2)
        
        # Move the mouse slightly to ensure the screen is active
        pyautogui.moveTo(500, 500, duration=0.5)
        pyautogui.click()
        
        # Wait for the click to register
        time.sleep(0.5)
        
        # Type the PIN (slowly to ensure each digit registers)
        pyautogui.typewrite(pin, interval=0.2)
        
        # Press Enter
        time.sleep(0.2)
        pyautogui.press('enter')
        
        logging.info("PIN entry completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error typing PIN: {e}")
        return False

def main():
    """Main function to handle the auto-login process"""
    logging.info("Auto-login script started")
    
    # Wait a bit to ensure the login screen is fully loaded
    time.sleep(1)
    
    # Check if we're actually at the lock screen
    if not is_lock_screen_visible():
        logging.warning("Not at login screen, exiting")
        return
    
    # Try to get the PIN (first from env var, then from encrypted file)
    pin = get_pin_from_env() or get_pin_from_encrypted_file()
    
    if not pin:
        logging.error("Failed to retrieve PIN, exiting")
        return
    
    # Type the PIN and press Enter
    success = type_pin_and_enter(pin)
    
    if success:
        logging.info("Auto-login completed successfully")
    else:
        logging.error("Auto-login failed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")