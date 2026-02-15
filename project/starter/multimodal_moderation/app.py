#!/usr/bin/env python3
import subprocess
import signal
import sys
import phoenix as px


def main():
    session = px.launch_app(port=6006)

    if not session:
        raise RuntimeError("Failed to launch Phoenix session.")
    
    print(f"🔍 Phoenix UI: {session.url}")

    def signal_handler(sig, frame):
        print("\nShutting down...")
        api_process.terminate()
        chat_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    api_process = subprocess.Popen(["multimodal-moderation-api"])
    chat_process = subprocess.Popen(["multimodal-moderation-chat"])

    try:
        api_process.wait()
        chat_process.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
