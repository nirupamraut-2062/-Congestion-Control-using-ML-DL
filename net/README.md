# Real-network test

The project's controllers send real UDP traffic over Wi-Fi from one laptop to another. The
packets cross a real wireless link and two real operating systems, so RTTs, jitter and losses
are measured, not simulated.

| Laptop | Runs | Needs |
|---|---|---|
| Receiver (any OS, here a Mac) | `receiver.py` only | Python 3.8 or newer, nothing else |
| Sender (here Windows) | the controllers, through `net/run_network_test.py` | this project with its virtual environment |

To compare with the simulator on equal terms, the receiver puts the simulator's standard
bottleneck in front of itself: a 10 Mbps drop-tail queue of 50 packets, plus enough extra delay
that the base RTT is about 40 ms. The sender sets this before every run.

## Step 1: receiver (the Mac)

1. Connect both laptops to the same Wi-Fi network. If the college Wi-Fi does not let laptops
   reach each other, turn on a phone's hotspot and connect both laptops to it.
2. Copy `receiver.py` to the Mac, open Terminal in that folder and run:

   ```
   python3 receiver.py
   ```

   If macOS offers to install the command line developer tools, accept: that installs `python3`.
   If it asks whether Python may accept incoming network connections, click Allow.
3. It prints `This laptop's address: 192.168.x.y`. Send that address to the sender.
4. Leave the window open and the Mac awake (lid open, on power) until the test ends.
   Stop it with Ctrl+C.

## Step 2: sender (the Windows laptop)

Open PowerShell in the project folder:

```
.venv\Scripts\activate
python -m net.run_network_test --receiver 192.168.x.y --only path
```

This takes about 5 seconds and prints the Wi-Fi RTT. If it works, run the full test:

```
python -m net.run_network_test --receiver 192.168.x.y
```

| Part | What it does | Time |
|---|---|---|
| path | RTT of the bare Wi-Fi path and through the emulated bottleneck | 5 s |
| single | each controller alone for 30 s | 4 min |
| loss | 1% random loss added at the receiver | 3 min |
| fairness | two flows sharing the bottleneck for 60 s | 5 min |

`--quick` halves every run (about 7 minutes in total). `--only single fairness` runs only those parts.

Results go to `results/network_*.csv` and `results/network_*.png`, next to the simulator's
result for the same link. Every run's 100 ms history is in `results/network_runs/`.

## For a clean measurement

- Keep both laptops close to the router or phone, and plugged in.
- Close downloads, video calls and streaming on both laptops, and turn off any VPN.
- Do not use the laptops while the test runs.

## If it does not work

| Message | What to check |
|---|---|
| `No reply from the receiver` | `receiver.py` is running; the address is typed correctly; both laptops are on the same network; the Mac's firewall allows Python (System Settings, Network, Firewall) |
| `Address already in use` on the Mac | Another program uses UDP port 9000: run `python3 receiver.py --port 9001` and add `--port 9001` to the sender's command |
| Very high or unstable RTT | Move closer to the router, or use a phone hotspot with no other devices on it |
