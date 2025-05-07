import subprocess
import time
import sys
import os
from threading import Thread

def run_fastapi():
    print("Starting FastAPI service on port 8000...")
    # Use shell=True for Windows compatibility
    subprocess.Popen([sys.executable, "fast_api.py"], shell=True)

def run_django():
    print("Starting Django service on port 8080...")
    os.chdir("ipl_backend")
    # Use different port than FastAPI and shell=True for Windows
    subprocess.run([sys.executable, "manage.py", "runserver", "8080"], shell=True)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    api_thread = Thread(target=run_fastapi)
    api_thread.daemon = True
    api_thread.start()
    
    # Give FastAPI a moment to start
    print("Waiting for FastAPI to initialize...")
    time.sleep(3)
    
    # Start Django in the main thread
    run_django()