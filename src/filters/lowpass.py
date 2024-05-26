import polars as pl
from scipy.signal import butter, sosfiltfilt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    return sos


sos = butter_lowpass(0.25, 1 / 0.1, order=5)


def butter_lowpass_filter(
    data,
):
    try:
        x = sosfiltfilt(
            sos,
            data,
        )

        if len(x) < len(data):
            return data
        return x
    except Exception:
        return data


def butter_lowpass_filter_plot(
    df: pl.DataFrame, col: str, vehicle_col: str
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col)
        .map_elements(
            lambda x: pl.Series(
                values=butter_lowpass_filter(
                    x.to_numpy(),
                ),
                dtype=float,
            ),
        )
        .over(vehicle_col)
        .alias(f"{col}_lowpass")
    )
