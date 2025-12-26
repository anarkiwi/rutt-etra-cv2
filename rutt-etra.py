#!/usr/bin/env python3


import argparse
import cv2
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def nj_calc_displace(input_ds, line_spacing, displacement):
    y_new = np.repeat(
        np.arange(input_ds.shape[0], dtype=np.int16) * line_spacing, input_ds.shape[1]
    ).reshape(input_ds.shape[0], input_ds.shape[1])
    y_new += displacement
    return y_new


@njit(parallel=True, fastmath=True)
def nj_displace3(input_ds, displacement, shape, line_spacing):
    y_new = nj_calc_displace(input_ds, line_spacing, displacement)
    output_img = np.zeros(shape=shape, dtype=np.uint8)

    for i in prange(input_ds.shape[0]):
        y_new_row = y_new[i, :]
        ds_row = input_ds[i, :]
        for x in prange(input_ds.shape[1]):
            output_img[y_new_row[x], x, :] = ds_row[x, :]

    return output_img


@njit(parallel=True, fastmath=True)
def nj_displace(input_ds, displacement, shape, line_spacing):
    y_new = nj_calc_displace(input_ds, line_spacing, displacement)
    output_img = np.zeros(shape=shape, dtype=np.uint8)

    for i in prange(input_ds.shape[0]):
        y_new_row = y_new[i, :]
        ds_row = input_ds[i, :]
        for x in prange(input_ds.shape[1]):
            output_img[y_new_row[x], x] = ds_row[x]

    return output_img


def rutt_etra_color(input_img, num_lines, displacement_scale):
    h, w = input_img.shape[:2]
    line_spacing = h // num_lines
    input_ds = input_img[::line_spacing, :]
    mean = np.mean([input_ds[:, :, x] for x in range(input_ds.shape[-1])], axis=0)
    displacement = ((mean / 255.0) * (h * displacement_scale)).astype(np.int16)
    output_h = int(h * (1 + displacement_scale))
    shape = (output_h, w, 3)
    return nj_displace3(input_ds, displacement, shape, line_spacing)


def rutt_etra_bw(input_img, num_lines, displacement_scale):
    h, w = input_img.shape[:2]
    line_spacing = h // num_lines
    input_ds = cv2.cvtColor(input_img[::line_spacing, :], cv2.COLOR_BGR2GRAY)
    displacement = ((input_ds / 255.0) * (h * displacement_scale)).astype(np.int16)
    output_h = int(h * (1 + displacement_scale))
    shape = (output_h, w)
    frame = nj_displace(input_ds, displacement, shape, line_spacing)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def get_cap(infile):
    try:
        infile = int(infile)
        return cv2.VideoCapture(infile)
    except (TypeError, ValueError):
        pass
    return cv2.VideoCapture(infile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", default=None, type=str)
    parser.add_argument("--outfile", default="output.avi", type=str)
    parser.add_argument("--lines", default=60, type=int)
    parser.add_argument("--scale", default=0.1, type=float)
    parser.add_argument("--color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--monitor", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    rutt_etra = rutt_etra_color
    if args.color:
        print("doing color")
    else:
        print("not doing color")
        rutt_etra = rutt_etra_bw

    cap = get_cap(args.infile)
    print(f"opened {args.infile} for reading")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None
    frames = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            scanline_frame = rutt_etra(
                frame, num_lines=args.lines, displacement_scale=args.scale
            )
            if args.monitor:
                cv2.imshow("rutt etra", scanline_frame)
                cv2.waitKey(1)
            if out is None:
                frame_height, frame_width = scanline_frame.shape[:2]
                out = cv2.VideoWriter(
                    args.outfile,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS),
                    (frame_width, frame_height),
                )
                print(f"opened {args.outfile} for writing")
            out.write(scanline_frame)
            frames += 1
        except KeyboardInterrupt:
            break

    print(f"wrote {frames} frames")
    cap.release()
    if out:
        out.release()


main()
