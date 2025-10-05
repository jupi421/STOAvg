import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit as cf
from scipy.optimize import root
from tqdm.auto import tqdm
from dataclasses import dataclass

@dataclass(slots=True)
class Frame:
    center: float
    fit_params: tuple
    x_data: np.ndarray
    OP: np.ndarray
    Pol: np.ndarray

    def apply_window(self, bounds):
        low, high = bounds
        cond = (self.x_data >= low) & (self.x_data < high)
        ids = np.where(cond)[0]
        return Frame( center=self.center, fit_params=self.fit_params, x_data=self.x_data[ids].copy(), OP=self.OP[ids].copy(), Pol=self.Pol[ids].copy())
    
@dataclass(slots=True)
class BlockAvg: 
    x_data: np.ndarray
    OP: np.ndarray
    Pol: np.ndarray

    def __add__(self, other):
        if not isinstance(other, (Frame, BlockAvg)):
            raise NotImplemented

        return BlockAvg(self.x_data + other.x_data, self.OP + other.OP, self.Pol + other.Pol)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplemented

        return BlockAvg(self.x_data / other, self.OP / other, self.Pol / other)

    def save_to_txt(self, save_dir, filename):
        data = np.empty((self.x_data.shape[0], 7))
        data[:,0] = self.x_data
        data[:,1:4] = self.OP
        data[:,4:] = self.Pol
        np.savetxt(save_dir + filename, data)
        return

def tanh(x, a, b, c, xi):
    return a * np.tanh((x + b) / xi) + c

def fit_frames(
    OP_frames_path,
    Pol_frames_path,
    frame_indices=None,          # if given, ignores start/stop/step
    start=200,                   # inclusive
    stop=None,                   # exclusive (required if frame_indices is None)
    step=1,
    coeff_guess=(3*np.pi/180, 120, 0, 25),
    bounds=([-np.inf, -np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf]),
    maxfev=20000,
    plot=True,
    save_dir=".log",
    plot_initial_guess=True
):
    OP_frames_path = Path(OP_frames_path)
    Pol_frames_path = Path(Pol_frames_path)

    # derive frame list
    if frame_indices is None:
        if stop is None:
            raise ValueError("When frame_indices is None, you must provide 'stop'.")
        frame_indices = list(range(start, stop, step))

    # set up output dir and CSV
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "fit_params.csv"
        write_header = not csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["frame", "a", "b", "c", "xi", "rmse", "success", "message"])
    else:
        writer = None
        csv_file = None

    results = []
    frames = []

    for i in tqdm(frame_indices, desc="Fitting"):
        fpath_op = OP_frames_path / f"op{i}.out"
        fpath_pol = Pol_frames_path / f"pol{i}.out"
        if not fpath_op.exists() or not fpath_pol.exists():
            msg = f"directories could not be located"
            row = [i, None, None, None, None, None, False, msg]
            if writer: writer.writerow(row)
            results.append({
                "frame": i, "params": None, "cov": None,
                "success": False, "message": msg
            })
            continue

        data_op = np.loadtxt(fpath_op)
        data_pol = np.loadtxt(fpath_pol)
        x = data_op[:, 0]
        y = data_op[:, 3]

        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        opt, pcov = cf(tanh, x, y, p0=coeff_guess, bounds=bounds, maxfev=maxfev)

        yfit_on_x = tanh(x, *opt)
        rmse = float(np.sqrt(np.mean((yfit_on_x - y)**2)))

        # build frames
        res = root(lambda x: tanh(x, *opt), 150)
        frame = Frame(res.x[0], opt, data_op[:, 0], data_op[:, 1:4], data_pol[:, 1:4])
        frames.append(frame)

        if plot and save_dir is not None:
            xf = np.linspace(x.min(), x.max(), 800)
            plt.figure()
            plt.scatter(x, y, s=12, label="data")
            if plot_initial_guess and coeff_guess is not None:
                plt.plot(xf, tanh(xf, *coeff_guess), "--", label="initial guess")
            plt.plot(
                xf, tanh(xf, *opt),
                label=f"fit: a={opt[0]:.3g}, b={opt[1]:.3g}, c={opt[2]:.3g}, xi={opt[3]:.3g}"
            )
            plt.xlabel("x"); plt.ylabel("data[:,3]")
            plt.title(f"op{i}.out"); plt.legend(); plt.tight_layout()
            # Save with consistent name
            plt.savefig(save_dir / f"op{i}.out.fit.png", dpi=150)
            plt.close()

        row = [i, opt[0], opt[1], opt[2], opt[3], rmse, True, ""]
        if writer: writer.writerow(row)

        results.append({
            "frame": i, "params": opt, "cov": pcov,
            "rmse": rmse, "success": True, "message": ""
        })

    if csv_file is not None:
        csv_file.close()

    return frames, results

def _get_window_bounds(window_size, DW_center):
    c = float(DW_center)
    half = window_size / 2.0
    return (c - half, c + half)

def _get_windowed_data(frame, window_size):
    bounds = _get_window_bounds(window_size, frame.center)
    return frame.apply_window(bounds)

def block_avg(frames, num_blocks : int, window_size : float, save_dir="./res/blocks/"):
    if (len(frames) % num_blocks) != 0:
        raise RuntimeError("Block size does not match frame size")
    
    block_size = len(frames) // num_blocks

    blocks = []
    for block_start in tqdm(range(0, len(frames), block_size), desc="Processing Block:"):
        cur_frame = _get_windowed_data(frames[block_start], window_size)
        block = BlockAvg(cur_frame.x_data, cur_frame.OP, cur_frame.Pol) 

        for block_frame in range(block_start+1, block_start+block_size):
            cur_frame = _get_windowed_data(frames[block_frame], window_size)
            block += cur_frame

        block /= block_size
        blocks.append(block)
        block.save_to_txt(save_dir, filename=f"block{len(blocks)}.dat")

    block_avg = blocks[0]
    for i in range(1, len(blocks)):
        block_avg += blocks[i]

    block_avg /= len(blocks)
    block_avg.save_to_txt(save_dir, "block_avg.dat")
    return block_avg

# def auto_corr_func():

def main():
    frames, _ = fit_frames(
        OP_frames_path="../STOTools/example/OP/OP_T20_p80/",
        Pol_frames_path="../STOTools/example/POL/POL_T20_p80/",
        start=8801, stop=9101,
        plot=True
    )

    #block = block_avg(frames, 20, 110.5)

if __name__ == "__main__":
    main()
