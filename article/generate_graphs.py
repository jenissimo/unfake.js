import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from collections import Counter
import os
from sklearn.cluster import KMeans

# --- Translations Dictionary ---
TRANSLATIONS = {
    'en': {
        'profile_x_title': 'Horizontal Gradient Profile with Peaks',
        'profile_y_title': 'Vertical Gradient Profile with Peaks',
        'profile_xlabel_x': 'Pixel Column',
        'profile_xlabel_y': 'Pixel Row',
        'profile_ylabel': 'Sum of Gradient Magnitude',
        'profile_legend_profile': 'Gradient Profile',
        'profile_legend_peaks': 'Detected Peaks',
        'hist_x_title': 'Distribution of Horizontal Peak Spacings',
        'hist_y_title': 'Distribution of Vertical Peak Spacings',
        'hist_xlabel': 'Spacing (pixels)',
        'hist_ylabel': 'Frequency',
        'hist_mode_text': 'Most common spacing (Mode):\n{mode_val} pixels'
    },
    'ru': {
        'profile_x_title': 'Горизонтальный Профиль Градиента с Пиками',
        'profile_y_title': 'Вертикальный Профиль Градиента с Пиками',
        'profile_xlabel_x': 'Колонка пикселей',
        'profile_xlabel_y': 'Ряд пикселей',
        'profile_ylabel': 'Сумма магнитуды градиента',
        'profile_legend_profile': 'Профиль градиента',
        'profile_legend_peaks': 'Найденные пики',
        'hist_x_title': 'Распределение Расстояний между Пиками (Горизонтально)',
        'hist_y_title': 'Распределение Расстояний между Пиками (Вертикально)',
        'hist_xlabel': 'Расстояние (пиксели)',
        'hist_ylabel': 'Частота',
        'hist_mode_text': 'Наиболее частое расстояние (Мода):\n{mode_val} пикселей'
    }
}


def detect_scale_js_logic(signal):
    """
    A Python port of the detectScale function from the unfake.js utils.js.
    Uses a more robust statistical approach to find the pixel grid scale.
    """
    if len(signal) < 3:
        return 1, np.array([]), np.array([])

    signal_np = np.array(signal)
    mean_val = np.mean(signal_np)
    std_dev = np.std(signal_np)
    # A minimum threshold to avoid issues with flat signals
    threshold = max(mean_val + 1.5 * std_dev, np.min(signal_np) + 1e-6)

    # distance=3 means next peak must be at least 3 indices away (JS logic is > 2).
    peaks, _ = find_peaks(signal_np, height=threshold, distance=3)

    if len(peaks) <= 2:
        print('detectScale: Not enough peaks found, returning 1.')
        return 1, np.array([]), np.array([])

    spacings = np.diff(peaks)
    if spacings.size == 0:
        return 1, peaks, spacings
        
    print(f'detectScale: Found {len(peaks)} peaks at positions: {peaks}')
    print(f'detectScale: Spacings between peaks: {spacings}')
    
    median_spacing = np.median(spacings)
    close_spacings = [s for s in spacings if abs(s - median_spacing) <= 2]
    consistency_ratio = len(close_spacings) / len(spacings)
    
    print(f'detectScale: Median spacing: {median_spacing:.1f}, consistency: {consistency_ratio:.1%}')
    
    if consistency_ratio > 0.7:
        result = round(median_spacing)
        print(f'detectScale: Using median spacing: {result} (high consistency)')
        return result, peaks, spacings

    mode_spacing = Counter(spacings).most_common(1)[0][0]
    print(f'detectScale: Using fallback mode spacing: {mode_spacing} (low consistency)')
    final_scale = mode_spacing if mode_spacing > 1 else 1
    return final_scale, peaks, spacings


def edge_aware_detect_tiled_logic(gray_image):
    """
    Python port of the main `edgeAwareDetect` function from pixel.js,
    using the tiled approach.
    """
    print("Running tiled edge-aware scale detection...")
    
    h, w = gray_image.shape
    
    # Fallback to single-region analysis for small images
    if w < 150 or h < 150:
        print("Image is small, falling back to single region analysis.")
        profile_x = np.sum(np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)), axis=0)
        scale_x, _, _ = detect_scale_js_logic(profile_x)
        profile_y = np.sum(np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
        scale_y, _, _ = detect_scale_js_logic(profile_y)
        if scale_x > 1 and scale_y > 1 and abs(scale_x - scale_y) <= 2:
            return round((scale_x + scale_y) / 2)
        return max(scale_x, scale_y, 1)

    all_scales = []
    tile_count = 3
    overlap = 0.25

    tile_w = w // tile_count
    tile_h = h // tile_count
    overlap_w = int(tile_w * overlap)
    overlap_h = int(tile_h * overlap)

    for y in range(tile_count):
        for x in range(tile_count):
            roi_x = max(0, x * tile_w - overlap_w)
            roi_y = max(0, y * tile_h - overlap_h)
            roi_w = min(w - roi_x, tile_w + 2 * overlap_w)
            roi_h = min(h - roi_y, tile_h + 2 * overlap_h)

            if roi_w < 30 or roi_h < 30:
                continue
            
            tile_mat = gray_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if np.std(tile_mat) < 5.0:
                print(f"Skipping tile ({x},{y}) due to low variance.")
                continue

            profile_x = np.sum(np.absolute(cv2.Sobel(tile_mat, cv2.CV_64F, 1, 0, ksize=3)), axis=0)
            h_scale, _, _ = detect_scale_js_logic(profile_x)

            profile_y = np.sum(np.absolute(cv2.Sobel(tile_mat, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
            v_scale, _, _ = detect_scale_js_logic(profile_y)

            if h_scale > 1: all_scales.append(h_scale)
            if v_scale > 1: all_scales.append(v_scale)
            print(f"Tile ({x},{y}) scales: h={h_scale}, v={v_scale}")
            
    if not all_scales:
        print("Tiled detection yielded no scales, returning 1.")
        return 1
        
    best_scale = Counter(all_scales).most_common(1)[0][0]
    print(f"Edge-aware detection complete. Found scales: {all_scales}. Best guess: {best_scale}")
    return best_scale


def morphological_cleanup(image_np):
    """
    Python port of morphologicalCleanup from utils.js.
    """
    print("Applying morphological cleanup...")
    # The JS implementation uses a 2x2 kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # OPEN operation (erosion followed by dilation) to remove noise
    opened = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel, iterations=1)
    # CLOSE operation (dilation followed by erosion) to fill small gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def finalize_pixels(image_np):
    """
    Python port of finalizePixels from utils.js.
    Ensures binary alpha and black transparent pixels.
    """
    print("Finalizing pixels...")
    if image_np.shape[2] != 4:
        return image_np
    
    finalized = image_np.copy()
    alpha_channel = finalized[:, :, 3]
    
    # Where alpha is < 128, set RGBA to 0
    transparent_mask = alpha_channel < 128
    finalized[transparent_mask] = [0, 0, 0, 0]
    
    # Where alpha is >= 128, set it to 255
    opaque_mask = ~transparent_mask
    finalized[opaque_mask, 3] = 255
    
    return finalized

def jaggy_cleaner(image_np):
    """
    Python port of jaggyCleaner from utils.js.
    """
    print("Applying jaggy cleaner...")
    if image_np.shape[2] != 4:
        return image_np # Requires alpha channel

    h, w, _ = image_np.shape
    cleaned_img = image_np.copy()
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Check if current pixel is opaque
            if cleaned_img[y, x, 3] < 128:
                continue

            # Check neighbors' opacity
            n = cleaned_img[y - 1, x, 3] > 128
            s = cleaned_img[y + 1, x, 3] > 128
            e = cleaned_img[x + 1, y, 3] > 128
            w_ = cleaned_img[x - 1, y, 3] > 128
            ne = cleaned_img[y - 1, x + 1, 3] > 128
            nw = cleaned_img[y - 1, x - 1, 3] > 128
            se = cleaned_img[y + 1, x + 1, 3] > 128
            sw = cleaned_img[y + 1, x - 1, 3] > 128
            
            opaque_orth = sum([n, s, e, w_])
            opaque_diag = sum([ne, nw, se, sw])
            
            # Remove pixel if it has no orthogonal neighbors and only one diagonal one
            if opaque_orth == 0 and opaque_diag == 1:
                cleaned_img[y, x] = [0, 0, 0, 0]

    return cleaned_img


def alpha_binarization(image_np, threshold=128):
    """
    Sets alpha channel to 0 or 255 based on a threshold.
    Mimics the JS implementation. Returns a copy.
    """
    if image_np.shape[2] != 4:
        return image_np
    
    print(f"Binarizing alpha channel with threshold {threshold}...")
    image_copy = image_np.copy()
    alpha_channel = image_copy[:, :, 3]
    mask = alpha_channel >= threshold
    alpha_channel[mask] = 255
    alpha_channel[~mask] = 0
    return image_copy


def find_optimal_crop(gray_image, scale):
    """
    Finds the best grid offset by analyzing Sobel gradient profiles.
    A python port of the logic in our JS utils.
    """
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    profile_x = np.sum(np.absolute(sobel_x), axis=0)
    profile_y = np.sum(np.absolute(sobel_y), axis=1)

    def find_best_offset(profile, s):
        best_offset, max_score = 0, -1
        for offset in range(s):
            score = np.sum(profile[offset::s])
            if score > max_score:
                max_score, best_offset = score, offset
        return best_offset

    best_dx = find_best_offset(profile_x, scale)
    best_dy = find_best_offset(profile_y, scale)
    print(f"Found optimal crop offset: x={best_dx}, y={best_dy}")
    return best_dx, best_dy

def find_most_interesting_tile(image_gray_np, tile_size=150):
    """
    Finds the tile with the most variance (detail) to analyze,
    ensuring we have a content-rich area for demonstration.
    """
    print("Searching for the most interesting tile...")
    max_variance = -1
    best_tile_coords = (0, 0)
    h, w = image_gray_np.shape

    # Iterate with a step to not check every single pixel start
    step = tile_size // 4
    for y in range(0, h - tile_size, step):
        for x in range(0, w - tile_size, step):
            tile = image_gray_np[y:y + tile_size, x:x + tile_size]
            # Standard deviation is a good measure of variance/detail
            variance = np.std(tile)
            if variance > max_variance:
                max_variance = variance
                best_tile_coords = (x, y)
    
    print(f"Found most interesting tile at {best_tile_coords} with variance {max_variance:.2f}")
    return best_tile_coords


def create_introductory_images(image_color_np, image_gray_np, output_dir, scale=43):
    """
    Generates the core images for the article's introduction.
    """
    print("\n--- Generating introductory images ---")
    
    h, w = image_gray_np.shape
    img_bgra = cv2.cvtColor(image_color_np, cv2.COLOR_RGBA2BGRA)

    # 1. Misaligned Grid Overlay
    grid_misaligned = img_bgra.copy()
    for x in range(0, w, scale): cv2.line(grid_misaligned, (x, 0), (x, h), (0, 0, 255, 255), 2)
    for y in range(0, h, scale): cv2.line(grid_misaligned, (0, y), (w, y), (0, 0, 255, 255), 2)
    path = os.path.join(output_dir, 'intro_grid_overlay_misaligned.png')
    cv2.imwrite(path, grid_misaligned)
    print(f"Saved misaligned grid to '{path}'")

    # 2. Aligned Grid Overlay
    offset_x, offset_y = find_optimal_crop(image_gray_np, scale)
    grid_aligned = img_bgra.copy()
    for x in range(offset_x, w, scale): cv2.line(grid_aligned, (x, 0), (x, h), (0, 255, 0, 255), 2) # Green lines
    for y in range(offset_y, h, scale): cv2.line(grid_aligned, (0, y), (w, y), (0, 255, 0, 255), 2)
    path = os.path.join(output_dir, 'intro_grid_overlay_aligned.png')
    cv2.imwrite(path, grid_aligned)
    print(f"Saved aligned grid to '{path}'")

    # 3. Nearest Neighbor Downscale and Upscale for comparison
    h, w, _ = image_color_np.shape
    new_w, new_h = w // scale, h // scale
    
    # Use the COLOR image for this operation
    nn_downscaled = cv2.resize(image_color_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Upscale back to a large, viewable size
    upscaled_w, upscaled_h = 1024, 1024 # Make it big enough to see pixels
    nn_upscaled = cv2.resize(nn_downscaled, (upscaled_w, upscaled_h), interpolation=cv2.INTER_NEAREST)
    
    # Convert back to BGRA for saving with OpenCV
    nn_upscaled_bgra = cv2.cvtColor(nn_upscaled, cv2.COLOR_RGBA2BGRA)
    path = os.path.join(output_dir, 'intro_nearest_neighbor_upscaled.png')
    cv2.imwrite(path, nn_upscaled_bgra)
    print(f"Saved upscaled nearest neighbor to '{path}'")


def create_analysis_images(image_gray_np, output_dir, tile_size=150):
    """
    Generates images for the scale detection analysis chapter.
    """
    print("\n--- Generating analysis images ---")
    
    # 1. Analysis Tile - Use the new function to find a content-rich tile
    tile_x, tile_y = find_most_interesting_tile(image_gray_np, tile_size)
    tile = image_gray_np[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]
    tile_path = os.path.join(output_dir, 'analysis_tile.png')
    cv2.imwrite(tile_path, tile)
    print(f"Saved analysis tile to '{tile_path}'")

    # 2. Sobel Gradients
    sobel_x = cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(tile, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_x_scaled = np.uint8(255 * np.absolute(sobel_x) / np.max(np.absolute(sobel_x)))
    sobel_y_scaled = np.uint8(255 * np.absolute(sobel_y) / np.max(np.absolute(sobel_y)))
    
    sobel_x_path = os.path.join(output_dir, 'analysis_sobel_x.png')
    cv2.imwrite(sobel_x_path, sobel_x_scaled)
    print(f"Saved Sobel X to '{sobel_x_path}'")

    sobel_y_path = os.path.join(output_dir, 'analysis_sobel_y.png')
    cv2.imwrite(sobel_y_path, sobel_y_scaled)
    print(f"Saved Sobel Y to '{sobel_y_path}'")
    
    return tile

def downscale_by_dominant_color(image_color_np, scale, threshold=0.05):
    """
    Downscales an image by finding the dominant color in each block,
    with a fallback to the mean color if dominance is below a threshold.
    Handles alpha channel correctly. A python port of the core JS logic.
    """
    h, w, channels = image_color_np.shape
    has_alpha = channels == 4
    
    # Ensure the new dimensions are integers
    new_h, new_w = h // scale, w // scale
    output_image = np.zeros((new_h, new_w, channels), dtype=np.uint8)

    print(f"Downscaling with Dominant Color method (threshold: {threshold*100}%)...")
    for r in range(new_h):
        for c in range(new_w):
            block = image_color_np[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            
            if has_alpha:
                pixels_rgba = block.reshape(-1, 4)
                # Consider only pixels that are mostly opaque for color calculation
                opaque_pixels_rgb = pixels_rgba[pixels_rgba[:, 3] > 128][:, :3]
                all_alpha_values = pixels_rgba[:, 3]
            else:
                opaque_pixels_rgb = block.reshape(-1, 3)

            # Determine final alpha value for the output pixel
            final_alpha = 0
            if has_alpha and all_alpha_values.size > 0:
                # Use median alpha, binarized
                final_alpha = 255 if np.median(all_alpha_values) > 128 else 0
            
            # If no opaque pixels, the block is transparent
            if opaque_pixels_rgb.shape[0] == 0:
                if has_alpha:
                    output_image[r, c] = [0, 0, 0, 0]
                else:
                    output_image[r, c] = [0, 0, 0]
                continue

            # Find dominant color
            unique_colors, counts = np.unique(opaque_pixels_rgb, axis=0, return_counts=True)
            dominant_idx = counts.argmax()
            
            # Decide whether to use dominant or mean
            if (counts[dominant_idx] / opaque_pixels_rgb.shape[0]) >= threshold:
                final_rgb = unique_colors[dominant_idx]
            else:
                mean_color = np.mean(opaque_pixels_rgb, axis=0)
                final_rgb = np.round(mean_color).astype(np.uint8)
            
            if has_alpha:
                output_image[r, c] = [*final_rgb, final_alpha]
            else:
                output_image[r, c] = final_rgb
            
    return output_image

def find_most_interesting_block(image_color_np, scale):
    """
    Finds a block with a close "election" between colors to best demonstrate
    the dominant color algorithm.
    """
    print("Searching for the most interesting block for demo...")
    best_block_info = {'score': float('inf'), 'coords': (0, 0), 'block': None, 'variance': 0}
    h, w, _ = image_color_np.shape

    for r in range(h // scale):
        for c in range(w // scale):
            block = image_color_np[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            
            # We need RGB for color analysis. We only consider opaque pixels.
            has_alpha = block.shape[2] == 4
            if has_alpha:
                pixels_rgba = block.reshape(-1, 4)
                pixels_rgb = pixels_rgba[pixels_rgba[:, 3] > 128][:, :3]
                if pixels_rgb.shape[0] < 10: # Skip almost transparent blocks
                    continue
            else:
                pixels_rgb = block.reshape(-1, 3)

            unique_colors, counts = np.unique(pixels_rgb, axis=0, return_counts=True)

            # We need at least 4 colors for a good demo.
            if len(counts) < 4:
                continue

            # The top color should not be overwhelmingly dominant (e.g., > 80%)
            if np.max(counts) / len(pixels_rgb) > 0.8:
                continue

            # Sort counts to find the top 2
            sorted_counts = np.sort(counts)[::-1]
            
            # Calculate ratio of top 2 colors. Perfect score is 1.0 (counts are equal).
            ratio = sorted_counts[0] / sorted_counts[1]
            
            # Score is how far the ratio is from a "close call" (e.g., 1.5).
            # We want to minimize this score.
            score = abs(ratio - 1.5) 

            # We also favor higher variance to avoid flat, boring blocks.
            variance = np.std(cv2.cvtColor(block, cv2.COLOR_RGB2GRAY))
            if variance < 15: # Skip low-variance blocks
                continue

            # We want the best score (close race) but will use variance as a tie-breaker.
            if score < best_block_info['score'] or \
               (score == best_block_info['score'] and variance > best_block_info['variance']):
                best_block_info['score'] = score
                best_block_info['coords'] = (r, c)
                best_block_info['block'] = block
                best_block_info['variance'] = variance

    r_idx, c_idx = best_block_info['coords']
    best_block = best_block_info['block']
    if best_block is None: # Fallback if no suitable block is found
        print("Warning: Could not find an ideal demo block. Falling back to original method.")
        return find_most_interesting_block_by_variance(image_color_np, scale)

    # Recalculate stats for logging
    final_pixels = best_block.reshape(-1, 3)
    uc, cnts = np.unique(final_pixels, axis=0, return_counts=True)
    sorted_cnts = np.sort(cnts)[::-1]
    final_ratio = sorted_cnts[0] / sorted_cnts[1] if len(sorted_cnts) > 1 else -1

    print(f"Found best demo block at row {r_idx}, col {c_idx} with top-2 color ratio: {final_ratio:.2f}")
    return best_block

def find_most_interesting_block_by_variance(image_color_np, scale):
    """Fallback to find the block with most variance, used if the main method fails."""
    # This is a simplified version of the original "find_most_interesting_block"
    max_variance = -1
    best_block = None
    h, w, _ = image_color_np.shape
    for r in range(h // scale):
        for c in range(w // scale):
            block = image_color_np[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            variance = np.std(cv2.cvtColor(block, cv2.COLOR_RGB2GRAY))
            if variance > max_variance:
                max_variance = variance
                best_block = block
    return best_block


def create_dominant_color_demo(image_color_np, output_dir, scale=43):
    """
    Creates a multi-part visualization for the dominant color process
    on a single, interesting block. Also generates a Russian version of the tally chart.
    """
    print("\n--- Generating Dominant Color demo images ---")
    
    # Find an interesting block in the cropped image
    block = find_most_interesting_block(image_color_np, scale)

    # 1. Save the magnified block
    # Upscale the block itself for viewing
    block_mag = cv2.resize(block, (256, 256), interpolation=cv2.INTER_NEAREST)
    path = os.path.join(output_dir, 'chapter2_demo_block.png')
    cv2.imwrite(path, cv2.cvtColor(block_mag, cv2.COLOR_RGB2BGR))
    print(f"Saved demo block to '{path}'")

    # 2. Create and save a color tally chart (EN)
    pixels = block.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Sort by count
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]

    # Convert RGB colors to hex for matplotlib
    hex_colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in unique_colors]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    y_pos = np.arange(len(unique_colors))
    ax.barh(y_pos, counts, align='center', color=hex_colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{c} ({p:.1f}%)' for c, p in zip(hex_colors, 100 * counts / counts.sum())])
    ax.invert_yaxis()
    ax.set_xlabel('Pixel Count')
    ax.set_title('Color Distribution in Demo Block')
    fig.tight_layout()
    path = os.path.join(output_dir, 'chapter2_demo_tally.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved demo tally chart to '{path}'")

    # 2b. Russian version
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.barh(y_pos, counts, align='center', color=hex_colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{c} ({p:.1f}%)' for c, p in zip(hex_colors, 100 * counts / counts.sum())])
    ax.invert_yaxis()
    ax.set_xlabel('Количество пикселей')
    ax.set_title('Распределение цветов в демо-блоке')
    fig.tight_layout()
    path_ru = os.path.join(output_dir, 'chapter2_demo_tally_ru.png')
    fig.savefig(path_ru)
    plt.close(fig)
    print(f"Saved demo tally chart (RU) to '{path_ru}'")

    # 3. Save the dominant color winner
    dominant_color = unique_colors[0]
    winner_img = np.full((256, 256, 3), dominant_color, dtype=np.uint8)
    path = os.path.join(output_dir, 'chapter2_demo_winner.png')
    cv2.imwrite(path, cv2.cvtColor(winner_img, cv2.COLOR_RGB2BGR))
    print(f"Saved demo winner to '{path}'")


def plot_palette(colors, title, output_path):
    """Plots a list of colors as a horizontal bar."""
    if not colors:
        print(f"No colors to plot for '{title}'. Skipping.")
        return
    # The list might be nested, so flatten it if necessary
    if isinstance(colors[0], list) or isinstance(colors[0], np.ndarray):
        colors = colors[0]

    fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
    # Reshape the color list into a 1-pixel high image
    palette_img = np.array(colors, dtype=np.uint8).reshape(1, -1, 3)
    ax.imshow(palette_img, aspect='auto')
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved palette plot to '{output_path}'")


def quantize_image(image_color_np, n_colors=16):
    """
    Helper to just quantize an image without saving intermediate palettes.
    """
    print(f"Quantizing image to {n_colors} colors...")
    h, w, channels = image_color_np.shape
    has_alpha = channels == 4
    pixels_for_fit = image_color_np.reshape(-1, channels)[:, :3]

    if len(np.unique(pixels_for_fit, axis=0)) < n_colors:
        print("Skipping quantization, image already has fewer colors than target.")
        return image_color_np.copy()
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(pixels_for_fit)
    new_palette = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.predict(pixels_for_fit)
    quantized_rgb = new_palette[labels].reshape(h, w, 3)

    if has_alpha:
        alpha_channel = image_color_np[:, :, 3]
        quantized_image = cv2.merge([quantized_rgb[:,:,0], quantized_rgb[:,:,1], quantized_rgb[:,:,2], alpha_channel])
    else:
        quantized_image = quantized_rgb
    return quantized_image


def quantize_image_and_create_images(image_color_np, output_dir, n_colors=16):
    """
    Quantizes an image to n_colors and saves palette comparison visuals.
    Returns the quantized image. Handles RGBA images correctly.
    """
    print(f"\n--- Running Pipeline Step: Quantizing to {n_colors} colors ---")
    
    h, w, channels = image_color_np.shape
    has_alpha = channels == 4

    # The RGB data to be used for fitting the quantizer
    pixels_for_fit = image_color_np.reshape(-1, channels)[:, :3]
    
    # 1. Show the palette BEFORE quantization
    # We create the 'before' palette from opaque pixels only for a cleaner look
    if has_alpha:
        opaque_pixels_rgb = image_color_np.reshape(-1, 4)[image_color_np.reshape(-1, 4)[:, 3] > 0, :3]
        unique_colors_before = np.unique(opaque_pixels_rgb, axis=0)
    else:
        unique_colors_before = np.unique(pixels_for_fit, axis=0)
    
    unique_colors_before_sorted = sorted(list(map(tuple, unique_colors_before)), key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])
    plot_palette([np.array(unique_colors_before_sorted)], 'Original Palette ({} colors)'.format(len(unique_colors_before_sorted)), os.path.join(output_dir, 'chapter3_palette_before.png'))

    # 2. Quantize the palette using K-Means on all pixels' RGB data
    print(f"Quantizing {pixels_for_fit.shape[0]} pixels to {n_colors} colors using K-Means...")
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(pixels_for_fit)
    new_palette = kmeans.cluster_centers_.astype(np.uint8)
    
    # Create the quantized image by predicting on the same RGB data
    labels = kmeans.predict(pixels_for_fit)
    quantized_rgb = new_palette[labels].reshape(h, w, 3)

    if has_alpha:
        # Re-attach the original alpha channel
        alpha_channel = image_color_np[:, :, 3]
        # Need to use cv2.merge, as np.dstack is tricky with shapes
        quantized_image = cv2.merge([quantized_rgb[:,:,0], quantized_rgb[:,:,1], quantized_rgb[:,:,2], alpha_channel])
    else:
        quantized_image = quantized_rgb

    # 3. Show the quantized palette
    new_palette_sorted = sorted(list(map(tuple, new_palette)), key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])
    plot_palette([np.array(new_palette_sorted)], f'Quantized Palette ({n_colors} colors)', os.path.join(output_dir, 'chapter3_palette_after.png'))

    # Save the intermediate quantized result, ensuring we handle alpha
    save_path = os.path.join(output_dir, 'chapter3_final_quantized_unscaled.png')
    if has_alpha:
        cv2.imwrite(save_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(save_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
    print(f"Saved intermediate quantized result to '{save_path}'")
    
    return quantized_image


def process_single_image_for_article(image_path, output_dir, n_colors=16):
    """
    Runs the full unfake pipeline on a single image and saves the result and a comparison image.
    This is for generating the comparison images for the final section of the article.
    """
    print(f"\n--- Processing example image: {image_path} ---")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Load and pre-process
    image_color_pil = Image.open(image_path).convert('RGBA')
    image_color_np = np.array(image_color_pil)
    image_color_np = alpha_binarization(image_color_np)
    image_gray_np = cv2.cvtColor(image_color_np, cv2.COLOR_RGBA2GRAY)

    # 2. Scale detection (now aligned with JS logic)
    scale = edge_aware_detect_tiled_logic(image_gray_np)
    print(f"Detected scale for {os.path.basename(image_path)}: {scale}")
        
    # 3. Crop
    cropped_image = image_color_np
    if scale > 1:
        h, w = image_gray_np.shape
        offset_x, offset_y = find_optimal_crop(image_gray_np, scale)
        new_w = (w - offset_x) // scale * scale
        new_h = (h - offset_y) // scale * scale
        if new_w > 0 and new_h > 0:
            cropped_image = image_color_np[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        else:
            print("Warning: Crop resulted in zero-sized image. Using original.")
            
    # 3.5 Morphological Cleanup
    cleaned_image = morphological_cleanup(cropped_image)

    # 4. Quantize
    quantized_image = quantize_image(cleaned_image, n_colors=n_colors)

    # 5. Downscale
    downscaled_image = quantized_image
    if scale > 1:
        downscaled_image = downscale_by_dominant_color(quantized_image, scale, threshold=0.05)
    
    # 5.5 Finalize and Jaggy Clean
    finalized_image = finalize_pixels(downscaled_image)
    jaggy_cleaned_image = jaggy_cleaner(finalized_image)
        
    # 6. Create side-by-side comparison and save it
    comparison_output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    create_comparison_image(image_path, jaggy_cleaned_image, comparison_output_path)


def create_comparison_image(original_path, result_np, output_path):
    """
    Creates a side-by-side comparison image: Original | Result
    """
    original_pil = Image.open(original_path).convert('RGBA')
    original_np = np.array(original_pil)
    h_orig, w_orig, _ = original_np.shape
    h_res, w_res, _ = result_np.shape
    
    # Upscale the small result to match the original's height for comparison
    # Use nearest neighbor to preserve pixel art style
    if h_res == 0 or w_res == 0:
        print(f"Error: Result image for {original_path} has zero dimension. Cannot create comparison.")
        return
        
    scale_factor = h_orig // h_res
    upscaled_result = cv2.resize(result_np, (w_res * scale_factor, h_res * scale_factor), interpolation=cv2.INTER_NEAREST)
    h_new, w_new, _ = upscaled_result.shape
    
    # Add a small border between them
    border_width = 15
    border_color = [220, 220, 220, 255] # Light gray
    
    # Create a canvas to hold both images
    # Pad with border color if heights don't match perfectly
    target_h = max(h_orig, h_new)
    comparison_img = np.full((target_h, w_orig + w_new + border_width, 4), border_color, dtype=np.uint8)
    
    # Place original and result on the canvas
    comparison_img[0:h_orig, 0:w_orig] = original_np
    comparison_img[0:h_new, w_orig + border_width:] = upscaled_result

    # Add labels
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.2
    font_color = (255, 255, 255, 255)
    stroke_color = (0, 0, 0, 255)
    thickness = 2
    stroke_thickness = 4
    
    cv2.putText(comparison_img, 'Original', (15, 40), font, font_scale, stroke_color, stroke_thickness, cv2.LINE_AA)
    cv2.putText(comparison_img, 'Original', (15, 40), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    cv2.putText(comparison_img, 'Result', (w_orig + border_width + 15, 40), font, font_scale, stroke_color, stroke_thickness, cv2.LINE_AA)
    cv2.putText(comparison_img, 'Result', (w_orig + border_width + 15, 40), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Save it
    cv2.imwrite(output_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGBA2BGRA))
    print(f"Saved comparison image to '{output_path}'")


def save_analysis_report(report_data, output_path):
    """
    Сохраняет текстовый отчёт по результатам анализа в указанный файл.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for key, value in report_data.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved analysis report to '{output_path}'")


def generate_all_article_images():
    """
    Main function to generate all images for the article, following the
    logic of the unfake.js browser tool for consistency.
    """
    print("Starting article image generation...")

    # --- Setup ---
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    image_path = 'article/images/goose_ai.png'
    output_dir = 'article/images'
    os.makedirs(output_dir, exist_ok=True)
    report_data = {}
    # ГЛОБАЛЬНАЯ инициализация переменных для полного анализа
    full_peaks_x = np.array([])
    full_spacings_x = np.array([])
    try:
        # === STAGE 1: Load Image & Initial Analysis ===
        print("\n--- STAGE 1: Initial Analysis & Problem Visualization ---")
        image_color_pil = Image.open(image_path).convert('RGBA')
        image_color_np_original = np.array(image_color_pil)
        image_color_np = alpha_binarization(image_color_np_original.copy())
        image_gray_np = cv2.cvtColor(image_color_np, cv2.COLOR_RGBA2GRAY)
        create_introductory_images(image_color_np_original, image_gray_np, output_dir, scale=43)
        analysis_tile_gray = create_analysis_images(image_gray_np, output_dir, tile_size=150)
        profile_x_for_detection = np.sum(np.absolute(cv2.Sobel(analysis_tile_gray, cv2.CV_64F, 1, 0, ksize=3)), axis=0)
        scale_for_graphs, peaks_x, spacings_x = detect_scale_js_logic(profile_x_for_detection)
        # === EXTRA: Найти самый регулярный тайл ===
        print("\n--- EXTRA: Searching for the most regular tile ---")
        best_tile_info = find_most_regular_tile(image_gray_np, tile_size=150, stride=50)
        if best_tile_info:
            x, y = best_tile_info['x'], best_tile_info['y']
            tile = best_tile_info['tile']
            spacings = best_tile_info['spacings']
            mode = best_tile_info['mode']
            mode_count = best_tile_info['mode_count']
            # Сохраняем картинку тайла
            tile_path = os.path.join(output_dir, 'most_regular_tile.png')
            cv2.imwrite(tile_path, tile)
            print(f"Saved most regular tile at ({x},{y}) to '{tile_path}' (mode: {mode}, count: {mode_count})")
            # Строим гистограмму spacings
            hist_path = os.path.join(output_dir, 'most_regular_tile_hist.png')
            plot_spacings_histogram(spacings, f"Most Regular Tile Spacing Distribution (mode: {mode})", "Spacing (pixels)", "Frequency", "Most common spacing (Mode):\n{mode_val} pixels", hist_path)
        else:
            print("Could not find a regular tile for extra analysis.")
        scale = edge_aware_detect_tiled_logic(image_gray_np)
        print(f"\n>>> Detected scale from full image analysis: {scale} <<< \n")
        report_data['Scale detected'] = scale
        # === STAGE 2: Execute Main Processing Pipeline ===
        print("\n--- STAGE 2: Executing Processing Pipeline & Generating Chapter Visuals ---")
        print("\n--- Running Pipeline Step: Surgical Cropping ---")
        offset_x, offset_y = find_optimal_crop(image_gray_np, scale)
        h, w, _ = image_color_np.shape
        new_w = (w - offset_x) // scale * scale
        new_h = (h - offset_y) // scale * scale
        cropped_image = image_color_np[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        report_data['Optimal crop offset'] = f"x={offset_x}, y={offset_y}"
        report_data['Cropped image size'] = f"{cropped_image.shape[1]} x {cropped_image.shape[0]}"
        path = os.path.join(output_dir, 'chapter2_cropped.png')
        cv2.imwrite(path, cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA))
        cropped_with_grid_bgra = cv2.cvtColor(cropped_image.copy(), cv2.COLOR_RGBA2BGRA)
        h_c, w_c, _ = cropped_with_grid_bgra.shape
        for x in range(0, w_c, scale): cv2.line(cropped_with_grid_bgra, (x, 0), (x, h_c), (0, 255, 0, 255), 2)
        for y in range(0, h_c, scale): cv2.line(cropped_with_grid_bgra, (0, y), (w_c, y), (0, 255, 0, 255), 2)
        path = os.path.join(output_dir, 'chapter2_cropped_with_grid.png')
        cv2.imwrite(path, cropped_with_grid_bgra)
        cleaned_image = morphological_cleanup(cropped_image)
        # --- Step 2.2: Color Quantization (for Chapter 3) ---
        if cleaned_image.shape[2] == 4:
            opaque_pixels_rgb = cleaned_image.reshape(-1, 4)[cleaned_image.reshape(-1, 4)[:, 3] > 0, :3]
        else:
            opaque_pixels_rgb = cleaned_image.reshape(-1, 3)
        unique_colors_before = np.unique(opaque_pixels_rgb, axis=0)
        report_data['Original palette size'] = f"{len(unique_colors_before)} colors"
        quantized_image = quantize_image_and_create_images(cleaned_image, output_dir, n_colors=16)
        if quantized_image.shape[2] == 4:
            opaque_pixels_rgb_q = quantized_image.reshape(-1, 4)[quantized_image.reshape(-1, 4)[:, 3] > 0, :3]
        else:
            opaque_pixels_rgb_q = quantized_image.reshape(-1, 3)
        unique_colors_after = np.unique(opaque_pixels_rgb_q, axis=0)
        report_data['Quantized palette size'] = f"{len(unique_colors_after)} colors"
        # --- Step 2.3: Content-Aware Downscaling (for Chapter 4) ---
        print("\n--- Running Pipeline Step: Content-Aware Downscaling ---")
        create_dominant_color_demo(cv2.cvtColor(quantized_image, cv2.COLOR_RGBA2RGB), output_dir, scale)
        downscaled_image = downscale_by_dominant_color(quantized_image, scale, threshold=0.05)
        blocks_x = downscaled_image.shape[1]
        blocks_y = downscaled_image.shape[0]
        report_data['Blocks'] = f"{blocks_x} x {blocks_y}"
        # Анализ доминирующего цвета в демо-блоке
        demo_block = find_most_interesting_block(cv2.cvtColor(quantized_image, cv2.COLOR_RGBA2RGB), scale)
        pixels = demo_block.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dominant_idx = counts.argmax()
        dominant_color = unique_colors[dominant_idx]
        dominant_percent = 100 * counts[dominant_idx] / counts.sum()
        report_data['Demo block dominant color'] = f"#{dominant_color[0]:02x}{dominant_color[1]:02x}{dominant_color[2]:02x} ({dominant_percent:.1f}%)"
        report_data['Fallback to mean color'] = 'Yes' if (dominant_percent < 5) else 'No'
        # === BONUS: Full Image Analysis for Rich Statistics ===
        print("\n--- BONUS: Analyzing full image for comprehensive statistics ---")
        try:
            full_sobel_x = cv2.Sobel(image_gray_np, cv2.CV_64F, 1, 0, ksize=3)
            full_profile_x = np.sum(np.absolute(full_sobel_x), axis=0)
            _, full_peaks_x, full_spacings_x = detect_scale_js_logic(full_profile_x)
            print(f"Full image analysis: {len(full_peaks_x)} peaks found, {len(full_spacings_x)} spacings calculated")
            # Create 2D gradient magnitude visualization
            full_sobel_y = cv2.Sobel(image_gray_np, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(full_sobel_x**2 + full_sobel_y**2)
            create_gradient_heatmap(gradient_magnitude, output_dir)
        except Exception as e:
            print(f"Warning: Full image analysis failed: {e}")
            print("Continuing with tile-based analysis only...")
        # Add full image analysis stats
        report_data['Full image peaks found'] = len(full_peaks_x)
        report_data['Full image spacings analyzed'] = len(full_spacings_x)
        if len(full_spacings_x) > 0:
            full_mode = Counter(full_spacings_x).most_common(1)[0]
            report_data['Full image mode spacing'] = f"{full_mode[0]} pixels ({full_mode[1]} occurrences)"
            report_data['Full image spacing range'] = f"{min(full_spacings_x)}-{max(full_spacings_x)} pixels"
        else:
            report_data['Full image mode spacing'] = "Analysis failed or no peaks found"
            report_data['Full image spacing range'] = "N/A"
        finalized_image = finalize_pixels(downscaled_image)
        jaggy_cleaned_image = jaggy_cleaner(finalized_image)
        path_small = os.path.join(output_dir, 'chapter2_final_result.png')
        cv2.imwrite(path_small, cv2.cvtColor(jaggy_cleaned_image, cv2.COLOR_RGBA2BGRA))
        final_upscaled = cv2.resize(jaggy_cleaned_image, (jaggy_cleaned_image.shape[1]*10, jaggy_cleaned_image.shape[0]*10), interpolation=cv2.INTER_NEAREST)
        path_upscaled_ch2 = os.path.join(output_dir, 'chapter2_final_result_upscaled.png')
        cv2.imwrite(path_upscaled_ch2, cv2.cvtColor(final_upscaled, cv2.COLOR_RGBA2BGRA))
        path_upscaled_ch3 = os.path.join(output_dir, 'chapter3_final_quantized_upscaled.png')
        cv2.imwrite(path_upscaled_ch3, cv2.cvtColor(final_upscaled, cv2.COLOR_RGBA2BGRA))
        # --- Визуализация финальной палитры ---
        plot_final_palette(jaggy_cleaned_image, os.path.join(output_dir, 'final_palette.png'))
        # === STAGE 3: Process Example Images for Final Section ===
        print("\n--- STAGE 3: Processing examples of where the algorithm struggles ---")
        example_images = [
            'article/images/example_1.png',
            'article/images/example_2.png',
            'article/images/example_3.png',
            'article/images/example_4.png',
        ]
        for img_path in example_images:
            process_single_image_for_article(img_path, output_dir, n_colors=16)
        # === STAGE 4: Generate Analysis Graphs for Chapter 1 ===
        print("\n--- STAGE 4: Generating Final Analysis Graphs ---")
        profile_y_for_detection = np.sum(np.absolute(cv2.Sobel(analysis_tile_gray, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
        _, peaks_y, spacings_y = detect_scale_js_logic(profile_y_for_detection)
        
        # === BONUS: Full Image Analysis for Rich Statistics ===
        print("\n--- BONUS: Analyzing full image for comprehensive statistics ---")
        # переменные уже инициализированы пустыми массивами выше
        for lang, t in TRANSLATIONS.items():
            print(f"--- Generating graphs for language: {lang.upper()} ---")
            plot_profile_with_peaks(
                profile_x_for_detection, peaks_x, t['profile_x_title'], t['profile_xlabel_x'], t['profile_ylabel'], 
                (t['profile_legend_profile'], t['profile_legend_peaks']), 
                os.path.join(output_dir, f'matplotlib_profile_x_peaks_{lang}.png')
            )
            plot_spacings_histogram(
                spacings_x, t['hist_x_title'], t['hist_xlabel'], t['hist_ylabel'], 
                t['hist_mode_text'], os.path.join(output_dir, f'matplotlib_spacings_x_hist_{lang}.png')
            )
            plot_profile_with_peaks(
                profile_y_for_detection, peaks_y, t['profile_y_title'], t['profile_xlabel_y'], t['profile_ylabel'],
                (t['profile_legend_profile'], t['profile_legend_peaks']),
                os.path.join(output_dir, f'matplotlib_profile_y_peaks_{lang}.png')
            )
            plot_spacings_histogram(
                spacings_y, t['hist_y_title'], t['hist_xlabel'], t['hist_ylabel'],
                t['hist_mode_text'], os.path.join(output_dir, f'matplotlib_spacings_y_hist_{lang}.png')
            )
            
            # BONUS: Full image histogram with rich statistics (only if we have data)
            if len(full_spacings_x) > 0:
                full_hist_title = {
                    'en': 'Full Image Peak Spacing Distribution (Complete Statistics)',
                    'ru': 'Распределение Расстояний по Всему Изображению (Полная Статистика)'
                }[lang]
                plot_spacings_histogram(
                    full_spacings_x, full_hist_title, t['hist_xlabel'], t['hist_ylabel'],
                    t['hist_mode_text'], os.path.join(output_dir, f'matplotlib_spacings_full_image_{lang}.png')
                )
            else:
                print(f"Skipping full image histogram for {lang} - no spacing data available")
        # --- Сохраняем текстовый отчёт ---
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        save_analysis_report(report_data, report_path)
        print("\nAll article images and graphs generated successfully!")
    except FileNotFoundError:
        print(f"\nERROR: Could not find the image at '{image_path}'.")
        print("Please make sure the image exists and you are running the script from the project root directory ('unfake.js/').")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

def create_gradient_profile_graphs():
    """
    Loads an image, calculates Sobel gradients for a specific tile,
    and generates high-quality profile graphs using Matplotlib for multiple languages.
    
    DEPRECATED: This function is now a wrapper for the new main entry point.
    """
    generate_all_article_images()


def plot_profile_with_peaks(profile_data, peaks, title, xlabel, ylabel, legends, output_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(profile_data, color='royalblue', linewidth=1.5, label=legends[0])
    if peaks.size > 0:
        ax.plot(peaks, profile_data[peaks.astype(int)], "x", color='crimson', markersize=8, label=legends[1])
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, len(profile_data) - 1)
    ax.set_ylim(0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved graph to '{output_path}'")

def create_gradient_heatmap(gradient_magnitude, output_dir):
    """
    Creates a 2D heatmap showing gradient magnitude across the full image.
    This visualizes where the AI created the strongest "pixel boundaries".
    Also generates a Russian version with translated labels.
    """
    print("Generating gradient magnitude heatmap...")
    
    # Normalize and enhance contrast
    normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = np.power(normalized / 255.0, 0.7) * 255  # Gamma correction for better visibility
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    
    # Original gradient magnitude
    im1 = ax1.imshow(gradient_magnitude, cmap='hot', aspect='equal')
    ax1.set_title('Raw Gradient Magnitude\n(Edge Detection Response)', fontsize=14, weight='bold')
    ax1.set_xlabel('Pixel Column')
    ax1.set_ylabel('Pixel Row')
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # Enhanced version for better visibility
    im2 = ax2.imshow(enhanced, cmap='viridis', aspect='equal')
    ax2.set_title('Enhanced Visualization\n(AI Pixel Grid Pattern)', fontsize=14, weight='bold')
    ax2.set_xlabel('Pixel Column')
    ax2.set_ylabel('Pixel Row')
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    fig.suptitle('Full Image Gradient Analysis: Where AI Created "Pixel" Boundaries', 
                fontsize=16, weight='bold', y=0.95)
    fig.tight_layout()
    
    path = os.path.join(output_dir, 'full_image_gradient_heatmap.png')
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved gradient heatmap to '{path}'")

    # --- Russian version ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    im1 = ax1.imshow(gradient_magnitude, cmap='hot', aspect='equal')
    ax1.set_title('Модуль градиента (сырые данные)', fontsize=14, weight='bold')
    ax1.set_xlabel('Колонка пикселей')
    ax1.set_ylabel('Ряд пикселей')
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    im2 = ax2.imshow(enhanced, cmap='viridis', aspect='equal')
    ax2.set_title('Усиленная визуализация\n(Паттерн AI-пикселей)', fontsize=14, weight='bold')
    ax2.set_xlabel('Колонка пикселей')
    ax2.set_ylabel('Ряд пикселей')
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    fig.suptitle('Градиентный анализ всего изображения: где AI создал "пиксельные" границы', 
                fontsize=16, weight='bold', y=0.95)
    fig.tight_layout()
    path_ru = os.path.join(output_dir, 'full_image_gradient_heatmap_ru.png')
    fig.savefig(path_ru, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved gradient heatmap (RU) to '{path_ru}'")

def plot_spacings_histogram(spacings, title, xlabel, ylabel, mode_text_template, output_path):
    if len(spacings) == 0:
        print(f"No spacings to plot for '{title}'. Skipping.")
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    counts = Counter(spacings)
    mode_val, mode_count = counts.most_common(1)[0]
    
    # Calculate smart display range
    data_min, data_max = min(spacings), max(spacings)
    data_range = data_max - data_min
    unique_values = len(counts)
    
    # For large datasets, focus on the most common values around the mode
    if unique_values > 20 or data_range > 50:
        # Focus on values within 3 standard deviations or around mode ±15
        focus_range = min(15, max(5, data_range // 4))
        plot_min = max(data_min, mode_val - focus_range)
        plot_max = min(data_max, mode_val + focus_range)
        
        # Filter spacings to this range for cleaner display
        filtered_spacings = [s for s in spacings if plot_min <= s <= plot_max]
        bins = np.arange(plot_min, plot_max + 2) - 0.5
        
        # Add info about filtered data
        filtered_count = len(filtered_spacings)
        total_count = len(spacings)
        filter_info = f" (showing {filtered_count}/{total_count} values around mode)" if filtered_count < total_count else ""
        
    else:
        # Small datasets: show everything with some padding
        padding = max(1, data_range // 6)
        plot_min = max(1, data_min - padding)
        plot_max = data_max + padding
        filtered_spacings = spacings
        bins = np.arange(plot_min, plot_max + 2) - 0.5
        filter_info = ""
    
    # Create histogram
    n, bins_used, patches = ax.hist(filtered_spacings, bins=bins, color='skyblue', 
                                   edgecolor='black', alpha=0.7, zorder=2)
    
    # Color the mode bar in salmon
    for patch in patches:
        bin_center = patch.get_x() + patch.get_width() / 2
        if abs(bin_center - mode_val) < 0.6:
            patch.set_facecolor('salmon')
            patch.set_alpha(0.9)
    
    ax.set_title(title + filter_info, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    
    # Smart tick spacing
    tick_step = max(1, (plot_max - plot_min) // 15)
    tick_positions = np.arange(plot_min, plot_max + 1, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xlim(plot_min - 0.5, plot_max + 0.5)
    
    # Enhanced info box
    total_measurements = len(spacings)
    mode_percentage = (mode_count / total_measurements) * 100
    mode_text = mode_text_template.format(mode_val=mode_val)
    
    # Add statistics
    median_val = np.median(spacings)
    std_val = np.std(spacings)
    
    info_text = f"{mode_text}\nOccurs {mode_count}× ({mode_percentage:.1f}%)\n"
    info_text += f"Median: {median_val:.1f} pixels\n"
    info_text += f"Std dev: {std_val:.1f} pixels\n"
    info_text += f"Total: {total_measurements} measurements"
    
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.9))
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved histogram to '{output_path}' (mode: {mode_val} pixels, {mode_count}× occurrences)")

def find_most_regular_tile(gray_image, tile_size=150, stride=50):
    """
    Ищет тайл, где разница между самым частым и вторым по частоте spacing максимальна.
    Возвращает координаты (x, y), массив spacings, Counter, и сам тайл.
    """
    h, w = gray_image.shape
    best_score = -1
    best_info = None
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = gray_image[y:y+tile_size, x:x+tile_size]
            profile_x = np.sum(np.absolute(cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)), axis=0)
            _, peaks, spacings = detect_scale_js_logic(profile_x)
            if len(spacings) == 0:
                continue
            freq = Counter(spacings)
            if len(freq) == 0:
                continue
            most_common = freq.most_common(2)
            mode_count = most_common[0][1]
            second_count = most_common[1][1] if len(most_common) > 1 else 0
            score = mode_count - second_count  # Чем больше разница, тем "чище" мода
            if score > best_score:
                best_score = score
                best_info = {
                    'x': x,
                    'y': y,
                    'spacings': spacings,
                    'freq': freq,
                    'tile': tile,
                    'mode': most_common[0][0],
                    'mode_count': mode_count
                }
    return best_info

def plot_final_palette(image_np, output_path):
    """
    Строит горизонтальный бар с цветами финальной палитры.
    """
    import numpy as np
    from PIL import Image
    # Оставляем только непрозрачные пиксели
    if image_np.shape[2] == 4:
        mask = image_np[:, :, 3] > 128
        pixels = image_np[mask]
        pixels = pixels[:, :3] if pixels.ndim == 2 else pixels.reshape(-1, 3)
    else:
        pixels = image_np.reshape(-1, 3)
    # Получаем уникальные цвета
    unique_colors = np.unique(pixels, axis=0)
    n = len(unique_colors)
    bar = np.zeros((32, n * 32, 3), dtype=np.uint8)
    for i, color in enumerate(unique_colors):
        bar[:, i*32:(i+1)*32, :] = color
    Image.fromarray(bar).save(output_path)
    print(f"Saved final palette to '{output_path}'")

if __name__ == '__main__':
    generate_all_article_images()