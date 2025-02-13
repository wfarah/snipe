import numpy as np
from scipy.stats import median_abs_deviation as MAD
from PIL import Image, ImageDraw, ImageFont


def normalize(ts):
    """
    Normalize timeseries by subtracting median and dividing by
    std derived from MAD
    """
    ts -= np.median(ts)
    mad = MAD(ts)
    std = mad * 1.4826
    ts /= std
    return ts

def findNearest(n, arr, arg=False):
    """
    Find the nearest point (or its position) to n in arr
    """
    argdiff = np.argmin(np.abs(n - arr))
    if arg:
        return argdiff
    return arr[argdiff]

def calculate_snr(signal, noise):
    """
    Text book top-hat SNR calculation:
    - Sum data points (already normalized aka baseline-subtracted)
    - Calculdate rms from noise (if normalized properly should be close to 0)
    - Divide s by sqrt(width) and by rms
    """
    rms = np.std(noise)
    s = signal.sum()
    return s / np.sqrt(len(signal)) / rms

def text_to_matrix(text, width, height, font_size=40):
    """
    Just a fun way to display a text in a matrix, thanks ChatGPT
    """
    # Create a blank image
    img = Image.new("L", (width, height), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()  # Fallback to default font
    font.font_size = 40

    # Get text bounding box and calculate centered position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width)  // 2
    text_y = (height - text_height) // 2

    # Draw text onto image
    draw.text((text_x, text_y), text, fill=255, font=font)

    # Convert image to a binary matrix (1 where text is, 0 elsewhere)
    matrix = np.array(img) > 128  # Threshold to binary (True = text)
    return np.flipud(matrix)
