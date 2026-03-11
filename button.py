#!/usr/bin/env python3
import lgpio
import time
import subprocess

# --- GPIO Pin Numbers ---
BUTTON1_PIN = 17  # Physical Pin 11
BUTTON2_PIN = 27  # Physical Pin 13

# --- Your Python Scripts ---
BUTTON1_SCRIPT = "/home/abel/Visionmate_v2/main.py"  # ← Change this
BUTTON2_SCRIPT = "/home/abel/Visionmate_v2/ocr.py"  # ← Change this


# --- Open GPIO chip (Pi 5 uses chip 4) ---
h = lgpio.gpiochip_open(4)

# --- Set pins as input with internal pull-up ---
lgpio.gpio_claim_input(h, BUTTON1_PIN, lgpio.SET_PULL_UP)
lgpio.gpio_claim_input(h, BUTTON2_PIN, lgpio.SET_PULL_UP)

# --- Track previous button states ---
prev = {BUTTON1_PIN: 1, BUTTON2_PIN: 1}

def run_script(path):
    print(f"▶ Running: {path}")
    subprocess.Popen(["python3", path])

print("✅ Listening for button presses... (Ctrl+C to stop)")

try:
    while True:
        for pin, script in [(BUTTON1_PIN, BUTTON1_SCRIPT),
                            (BUTTON2_PIN, BUTTON2_SCRIPT)]:
            state = lgpio.gpio_read(h, pin)
            if prev[pin] == 1 and state == 0:  # Button pressed
                run_script(script)
            prev[pin] = state
        time.sleep(0.05)  # 50ms debounce

except KeyboardInterrupt:
    print("\n Exiting...")

finally:
    lgpio.gpiochip_close(h)