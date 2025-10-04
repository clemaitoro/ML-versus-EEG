# read_waves.py
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def parse_args():
    p = argparse.ArgumentParser(description="Live EEG plot from OpenBCI/BrainFlow")
    p.add_argument("--board", default="cyton",
                   choices=["cyton", "cyton_daisy", "ganglion", "synthetic"],
                   help="Which board to use")
    p.add_argument("--port", default=None,
                   help="Serial port for Cyton/Ganglion (e.g., COM5, /dev/ttyUSB0, /dev/tty.usbserial-*)")
    p.add_argument("--window", type=float, default=8.0,
                   help="Seconds to display in the scrolling window")
    p.add_argument("--refresh", type=float, default=0.05,
                   help="Plot refresh period in seconds (e.g., 0.05 ≈ 20 FPS)")
    return p.parse_args()

def board_id_from_name(name: str) -> int:
    name = name.lower()
    if name == "cyton":
        return BoardIds.CYTON_BOARD.value
    if name == "cyton_daisy":
        return BoardIds.CYTON_DAISY_BOARD.value
    if name == "ganglion":
        return BoardIds.GANGLION_BOARD.value
    if name == "synthetic":
        return BoardIds.SYNTHETIC_BOARD.value
    raise ValueError(f"Unsupported board: {name}")



def main():
    args = parse_args()

    # Basic BrainFlow setup
    BoardShim.enable_dev_board_logger()  # helpful logs to stdout
    params = BrainFlowInputParams()

    bid = board_id_from_name(args.board)

    # For serial-based boards, a port is required
    if bid != BoardIds.SYNTHETIC_BOARD.value:
        if not args.port:
            raise SystemExit("Error: --port is required for real hardware (e.g., --port COM5 or /dev/ttyUSB0).")
        params.serial_port = args.port

    board = BoardShim(bid, params)
    sr = BoardShim.get_sampling_rate(bid)
    eeg_chs = BoardShim.get_eeg_channels(bid)

    print(f"Board: {args.board} | Sampling rate: {sr} Hz | EEG channels: {eeg_chs}")

    # Prepare session and start stream
    board.prepare_session()
    # internal ring buffer size in samples; 45000 is common and sufficient
    board.start_stream(45000)
    print("Streaming… (Ctrl+C to stop)")

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Live EEG — {args.board}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (raw units)")
    window_len = int(args.window * sr)

    # one line per EEG channel, vertically offset for readability
    offsets = np.arange(len(eeg_chs)) * 200.0  # adjust offset spacing if needed
    lines = []
    t = np.linspace(-args.window, 0, window_len)

    for i, ch in enumerate(eeg_chs):
        y0 = np.zeros(window_len) + offsets[i]
        line, = ax.plot(t, y0, lw=0.8)
        lines.append(line)

    ax.set_xlim(-args.window, 0.0)
    # Auto-scale y based on offsets
    ax.set_ylim(-200.0, offsets[-1] + 200.0 if len(offsets) else 200.0)
    ax.grid(True, alpha=0.3)

    try:
        # give the buffer a moment to fill
        time.sleep(1.0)

        while True:
            # get the most recent window_len samples without clearing the internal buffer
            data = board.get_current_board_data(window_len)
            if data.shape[1] > 0:
                for i, ch in enumerate(eeg_chs):
                    y = data[ch]
                    # pad if we have fewer than window_len samples at start
                    if y.size < window_len:
                        y = np.pad(y, (window_len - y.size, 0), mode="edge")
                    # apply offset for visual separation
                    y_disp = y + offsets[i]
                    lines[i].set_ydata(y_disp)

                # Update time axis in case window length changed
                fig.canvas.draw()
                fig.canvas.flush_events()

            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
