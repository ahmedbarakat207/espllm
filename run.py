import argparse
import os
import platform
import shutil
import subprocess
import sys


def get_pio_path():
    pio = shutil.which("pio") or shutil.which("platformio")
    if pio:
        return pio
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        candidates = [
            os.path.join(home, ".platformio", "penv", "Scripts", "pio.exe"),
            os.path.join(home, ".platformio", "penv", "Scripts", "platformio.exe"),
        ]
    else:
        candidates = [
            os.path.join(home, ".platformio", "penv", "bin", "pio"),
            os.path.join(home, ".platformio", "penv", "bin", "platformio"),
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "pio"


def auto_detect_port():
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if ports:
            return ports[0].device
    except ImportError:
        pass
    return "COM4" if platform.system() == "Windows" else "/dev/ttyUSB0"


def main():
    parser = argparse.ArgumentParser(description="ESP-LLM Serial Monitor")
    parser.add_argument("-p", "--port", default=None, help="Serial port (default: auto-detected / COM4)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = parser.parse_args()

    pio = get_pio_path()
    port = args.port or auto_detect_port()

    cmd = [pio, "device", "monitor", "-b", str(args.baud), "-p", port]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
