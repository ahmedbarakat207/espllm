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
    return None


def main():
    parser = argparse.ArgumentParser(description="Build, flash, and monitor ESP-LLM.")
    parser.add_argument("target", nargs="?", default=None, choices=["esp32", "esp8266", "esp32dev", "monitor"], help="Target architecture or 'monitor'")
    parser.add_argument("-p", "--port", default=None, help="Serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--build-only", action="store_true", help="Compile without uploading")
    parser.add_argument("--monitor-only", action="store_true", help="Open serial monitor without flashing")
    parser.add_argument("-m", "--monitor", action="store_true", help="Open serial monitor after upload")
    parser.add_argument("--skip-convert", action="store_true", help="Skip running convert_model_to_c.py")
    args = parser.parse_args()

    pio = get_pio_path()
    port = args.port or auto_detect_port()

    if args.target is None or args.target == "monitor" or args.monitor_only:
        mon_cmd = [pio, "device", "monitor", "-b", str(args.baud)]
        if port:
            mon_cmd.extend(["-p", port])
        print(f">> Starting monitor on {port or 'auto'} ({args.baud} baud)...")
        try:
            subprocess.run(mon_cmd)
        except KeyboardInterrupt:
            pass
        return

    env_name = "esp8266" if args.target == "esp8266" else "esp32dev"

    if not args.skip_convert:
        convert_cmd = [sys.executable, "convert_model_to_c.py", args.target]
        print(f">> Exporting weights: {' '.join(convert_cmd)}")
        res = subprocess.run(convert_cmd)
        if res.returncode != 0:
            sys.exit(res.returncode)

    build_cmd = [pio, "run", "-e", env_name]
    if not args.build_only:
        build_cmd.extend(["--target", "upload"])
        if port:
            build_cmd.extend(["--upload-port", port])

    print(f">> Running: {' '.join(build_cmd)}")
    res = subprocess.run(build_cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)

    if args.monitor:
        mon_cmd = [pio, "device", "monitor", "-b", str(args.baud)]
        if port:
            mon_cmd.extend(["-p", port])
        print(f">> Starting monitor ({args.baud} baud)...")
        try:
            subprocess.run(mon_cmd)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
