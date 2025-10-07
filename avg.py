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
    x: np.ndarray
    OP: np.ndarray
    Pol: np.ndarray

    def center_data(self):
        def rotate_array(array : np.ndarray, center_id):
            offset = (self.x.shape[0] // 2) - center_id

            if offset < 0: # take top, append bottom
                splits = np.vsplit(array, [np.abs(offset)])
                return np.vstack((splits[1],splits[0]))

            elif offset > 0: # take bottom, append top
                splits = np.vsplit(array, [self.x.shape[0] - offset])
                return np.vstack((splits[1], splits[0]))

            return array

        center_id = (np.abs(self.x - self.center)).argmin()

        self.OP = rotate_array(self.OP, center_id)
        self.Pol = rotate_array(self.Pol, center_id)
        self.x -= self.center
        return

    def save_to_txt(self, save_dir, filename):
        data = np.empty((self.x.shape[0], 7))
        data[:,0] = self.x
        data[:,1:4] = self.OP
        data[:,4:] = self.Pol
        np.savetxt(save_dir + filename, data, header=f"a={self.fit_params[0]}, b={self.fit_params[1]}, c={self.fit_params[2]}, xi={self.fit_params[3]}")
        return
    
@dataclass(slots=True)
class BlockAvg: 
    x: np.ndarray
    OP: np.ndarray
    Pol: np.ndarray

    def __add__(self, other):
        if not isinstance(other, (Frame, BlockAvg)):
            raise NotImplemented

        return BlockAvg(self.x + other.x, self.OP + other.OP, self.Pol + other.Pol)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplemented

        return BlockAvg(self.x / other, self.OP / other, self.Pol / other)

    def save_to_txt(self, save_dir, filename):
        data = np.empty((self.x.shape[0], 7))
        data[:,0] = self.x
        data[:,1:4] = self.OP
        data[:,4:] = self.Pol
        np.savetxt(save_dir + filename, data)
        return

def tanh(x, a, b, c, xi):
    return a * np.tanh((x + b) / xi) + c

def fit_frames(
    OP_frames_path,
    Pol_frames_path,
    window,
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
        data_op = data_op[window:-window, :]
        data_pol = data_pol[window:-window, :]
        x = data_op[:, 0]
        y = data_op[:, 3]

        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        opt, pcov = cf(tanh, x, y, p0=coeff_guess, bounds=bounds, maxfev=maxfev)

        yfit_on_x = tanh(x, *opt)
        rmse = float(np.sqrt(np.mean((yfit_on_x - y)**2)))

        # build frames
        #res = root(lambda x: tanh(x, *opt), 150)
        frame = Frame(-1*opt[1], opt, data_op[:, 0], data_op[:, 1:4], data_pol[:, 1:4])
        frame.save_to_txt("res/frames/", f"frame{i}.dat")
        frames.append(frame)

        # plot
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

def block_avg(frames, num_blocks : int, save_dir="./res/"):
    if (len(frames) % num_blocks) != 0:
        raise RuntimeError("Block size does not match frame size")
    
    block_size = len(frames) // num_blocks

    blocks = []
    centers = []
    for block_start_id in tqdm(range(0, len(frames), block_size), desc="Processing Block:"):
        block = BlockAvg(np.zeros(frames[0].x.shape), np.zeros(frames[0].OP.shape), np.zeros(frames[0].Pol.shape))

        for block_frame_id in range(block_start_id, block_start_id+block_size):
            frames[block_frame_id].center_data()
            block += frames[block_frame_id]
            centers.append(frames[block_frame_id].center)

        block /= block_size
        blocks.append(block)
        block.save_to_txt(save_dir+"blocks/", filename=f"block{len(blocks)}.dat")

    np.savetxt(save_dir + "centers.dat", np.array(centers))

    block_avg = blocks[0]
    for i in range(1, len(blocks)):
        block_avg += blocks[i]

    block_avg /= len(blocks)
    block_avg.save_to_txt(save_dir, "block_avg_40.dat")
    return block_avg

# def auto_corr_func():

#def load_frames(save_dir = "res/frames/"):
#    frames = []
#    data : np.ndarray
#    for i in range(start, stop):
#        data = np.loadtxt(save_dir+f"frame{i}.dat")
#        data[:,1:4] = self.OP
#        data[:,4:7] = self.Pol
#        data[:,7:] = self.fit_params
#        frames.append(Frame(data[:,0],))
#    return

def main():
    start = 3000
    stop = 5600

    frames, _ = fit_frames(
        OP_frames_path="../STOTools/example/OP/OP_T40_p80/",
        Pol_frames_path="../STOTools/example/POL/POL_T40_p80/",
        window = 25,
        start=start, stop=stop,
        plot=True
    )

    block = block_avg(frames, 100)

if __name__ == "__main__":
    main()
