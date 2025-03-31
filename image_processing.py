import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Discrete Fourier Transform (DFT)
def custom_dft(img):
    return np.fft.fft2(img.astype(np.float32)), img.shape

# High-Pass Filter (Edge Enhancement)
def apply_high_pass_filter(dft_shift, shape, radius=20):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = 1 - np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * radius ** 2))
    return dft_shift * mask

# Gaussian Low-Pass Filter (Noise Removal)
def apply_low_pass_filter(dft_shift, shape, radius=30):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * radius ** 2))
    return dft_shift * mask

# Enhanced Non-Local Means Denoising for color images
def enhanced_nlmeans_color(img, h=15, template_size=7, search_size=21, color_h=15):
    """Apply Non-Local Means denoising to color images with detail preservation"""
    if len(img.shape) == 2:  # Grayscale
        return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=template_size, searchWindowSize=search_size)
    else:  # Color image
        return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=color_h, 
                                              templateWindowSize=template_size, searchWindowSize=search_size)

# Enhanced Bilateral Filter (Preserve Edges)
def enhanced_bilateral_filter(img, d=11, sigma_color=75, sigma_space=75):
    """Apply bilateral filtering with parameters optimized for detail preservation"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

# Process each color channel separately for frequency domain operations
def process_color_channels(img, process_func, **kwargs):
    """Apply a processing function to each color channel separately"""
    if len(img.shape) == 2:  # Grayscale
        return process_func(img, **kwargs)
    
    # Process each channel
    result = np.zeros_like(img)
    for i in range(img.shape[2]):
        result[:,:,i] = process_func(img[:,:,i], **kwargs)
    
    return result

# Improved Horizontal Stripe Removal with detail preservation
def remove_horizontal_stripes(img, filter_strength=10):
    """Remove horizontal stripes while preserving details"""
    # Get image dimensions
    if len(img.shape) == 2:  # Grayscale
        rows, cols = img.shape
    else:  # Color image
        rows, cols = img.shape[:2]
    
    def process_channel(channel):
        # Convert to frequency domain
        dft = np.fft.fft2(channel.astype(np.float32))
        dft_shift = np.fft.fftshift(dft)
        
        # Create a mask to remove horizontal frequency components
        mask = np.ones((rows, cols), dtype=np.complex64)
        
        # Get center coordinates
        crow, ccol = rows // 2, cols // 2
        
        # Create a more targeted mask for horizontal stripes
        # Instead of just removing specific rows, create a smoother transition
        y_coords = np.arange(rows)
        
        # Calculate distance from center row (in frequency domain)
        y_dist = np.abs(y_coords - crow)
        
        # Convert distances to row-wise filter strength
        # Focus on horizontal frequency components (near center row)
        for i in range(rows):
            # Check if we're at a frequency that likely corresponds to stripes
            # Target frequencies near center but not at the DC component
            if 0 < y_dist[i] < 5:
                # Use a softer filter to preserve more details
                attenuation = 1 - (filter_strength / 15)
                mask[i, :] = max(0.25, attenuation)  # Never completely remove, just attenuate
            elif 5 <= y_dist[i] < 15:
                # Gradually reduce strength for frequencies farther from center
                attenuation = 1 - (filter_strength / 20) * (15 - y_dist[i]) / 10
                mask[i, :] = max(0.5, attenuation)
        
        # Preserve the DC component (average brightness)
        mask[crow, ccol] = 1
        
        # Apply mask and inverse transform
        dft_shift_filtered = dft_shift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(dft_shift_filtered))
        img_filtered = np.abs(img_back)
        
        # Normalize and convert to uint8
        img_filtered = np.uint8(255 * (img_filtered - np.min(img_filtered)) / 
                        (np.max(img_filtered) - np.min(img_filtered)))
        
        return img_filtered
    
    # Process each channel separately for color images
    if len(img.shape) == 2:  # Grayscale
        return process_channel(img)
    else:  # Color
        result = np.zeros_like(img)
        for i in range(img.shape[2]):
            result[:,:,i] = process_channel(img[:,:,i])
        return result

# Calculate Data Loss Function
def calculate_data_loss(original_img, processed_img):
    """
    Calculate multiple metrics to quantify data loss during denoising process
    
    Parameters:
    original_img (numpy.ndarray): Original input image
    processed_img (numpy.ndarray): Processed/denoised image
    
    Returns:
    dict: Dictionary containing various data loss metrics
    """
    # Ensure images have the same dimensions and type
    if original_img.shape != processed_img.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float for calculations
    original_float = original_img.astype(np.float32)
    processed_float = processed_img.astype(np.float32)
    
    # Calculate Mean Squared Error (MSE)
    if len(original_img.shape) == 3:  # Color image
        mse = np.mean((original_float - processed_float) ** 2, axis=(0, 1))
        mse_total = np.mean(mse)  # Average across channels
    else:  # Grayscale
        mse_total = np.mean((original_float - processed_float) ** 2)
    
    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    if mse_total == 0:  # Images are identical
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse_total)
    
    # Calculate Structural Similarity Index (SSIM)
    if len(original_img.shape) == 3:  # Color image
        # Calculate SSIM for each channel and average
        ssim_value = 0
        for i in range(original_img.shape[2]):
            ssim_channel = cv2.matchTemplate(original_img[:,:,i], processed_img[:,:,i], cv2.TM_CCORR_NORMED)[0][0]
            ssim_value += ssim_channel
        ssim_value /= original_img.shape[2]
    else:  # Grayscale
        ssim_value = cv2.matchTemplate(original_img, processed_img, cv2.TM_CCORR_NORMED)[0][0]
    
    # Calculate frequency domain loss
    def calculate_frequency_loss(img1, img2):
        # Convert to frequency domain
        fft1 = np.fft.fft2(img1.astype(np.float32))
        fft2 = np.fft.fft2(img2.astype(np.float32))
        
        # Calculate magnitude and phase
        mag1 = np.abs(fft1)
        mag2 = np.abs(fft2)
        
        # Calculate frequency magnitude loss
        mag_diff = np.abs(mag1 - mag2)
        mag_loss = np.sum(mag_diff) / (np.sum(mag1) + 1e-10) * 100  # Percentage loss
        
        return mag_loss
    
    # Calculate frequency loss for each channel
    if len(original_img.shape) == 3:  # Color image
        freq_loss = 0
        for i in range(original_img.shape[2]):
            freq_loss += calculate_frequency_loss(original_img[:,:,i], processed_img[:,:,i])
        freq_loss /= original_img.shape[2]
    else:  # Grayscale
        freq_loss = calculate_frequency_loss(original_img, processed_img)
    
    # Calculate detail loss (high-frequency component loss)
    def calculate_detail_loss(img1, img2):
        # Apply high-pass filter to extract details
        kernel = np.array([[-1, -1, -1], 
                           [-1,  8, -1], 
                           [-1, -1, -1]])
        
        details1 = cv2.filter2D(img1.astype(np.float32), -1, kernel)
        details2 = cv2.filter2D(img2.astype(np.float32), -1, kernel)
        
        # Calculate detail preservation ratio
        detail_energy1 = np.sum(np.abs(details1))
        detail_energy2 = np.sum(np.abs(details2))
        
        if detail_energy1 < 1e-10:
            return 0  # No details in original
        
        detail_preservation = detail_energy2 / detail_energy1 * 100  # Percentage preserved
        detail_loss = 100 - detail_preservation  # Percentage lost
        
        return detail_loss
    
    # Calculate detail loss for each channel
    if len(original_img.shape) == 3:  # Color image
        detail_loss = 0
        for i in range(original_img.shape[2]):
            detail_loss += calculate_detail_loss(original_img[:,:,i], processed_img[:,:,i])
        detail_loss /= original_img.shape[2]
    else:  # Grayscale
        detail_loss = calculate_detail_loss(original_img, processed_img)
    
    # Return all metrics in a dictionary
    return {
        'mse': mse_total,
        'psnr': psnr,
        'ssim': ssim_value,
        'frequency_loss_percent': freq_loss,
        'detail_loss_percent': detail_loss,
        'data_preserved_percent': 100 - detail_loss
    }

# Enhanced denoising with TV-L1 (Total Variation) - alternative to wavelets
def tv_l1_denoising(img, weight=30, iterations=30):
    """
    TV-L1 denoising - good for removing grainy noise while preserving edges
    """
    # Process grayscale images
    if len(img.shape) == 2:
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        # Apply TV-L1 denoising
        # This is an edge-preserving smoothing method
        result = cv2.denoise_TVL1(np.float32([img_float]), weight, iterations)[0]
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    # Process color images by channel
    else:
        # Split into channels
        channels = cv2.split(img)
        result_channels = []
        
        # Process each channel
        for channel in channels:
            # Convert to float
            channel_float = channel.astype(np.float32) / 255.0
            
            # Apply TV-L1 denoising
            denoised = cv2.denoise_TVL1(np.float32([channel_float]), weight, iterations)[0]
            
            # Convert back to uint8
            denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
            result_channels.append(denoised)
        
        # Merge channels back
        return cv2.merge(result_channels)

# Adaptive Denoising - Detect and apply appropriate methods with color preservation
def adaptive_denoising(img):
    # Convert to grayscale for analysis only
    if len(img.shape) == 3:  # Color image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        img_gray = img.copy()
    
    # Analyze image to determine dominant noise type
    fft_img = np.fft.fft2(img_gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft_img)
    magnitude = np.log(np.abs(fft_shift) + 1)
    
    # Check for horizontal lines in frequency domain
    rows, cols = magnitude.shape
    center_row = rows // 2
    
    # Measure horizontal components
    horizontal_energy = np.sum(magnitude[center_row-5:center_row+5, :])
    vertical_energy = np.sum(magnitude[:, center_row-5:center_row+5])
    total_energy = np.sum(magnitude)
    
    horizontal_ratio = horizontal_energy / total_energy
    
    # Measure noise uniformity (grainy noise detection)
    std_dev = np.std(img_gray)
    local_std = cv2.boxFilter(np.float32(img_gray*2), -1, (5,5)) - cv2.boxFilter(np.float32(img_gray), -1, (5,5))*2
    local_std = np.sqrt(np.maximum(local_std, 0))
    grain_ratio = np.mean(local_std) / std_dev
    
    # Apply appropriate denoising based on detected noise type
    if horizontal_ratio > 0.1 and horizontal_energy > vertical_energy:  # Horizontal stripe noise detected
        print("Detected horizontal stripes - applying specialized filter")
        # Use a gentler stripe removal to preserve details
        result_img = remove_horizontal_stripes(img, filter_strength=10)
        # Add a gentle bilateral filter to preserve edges
        result_img = enhanced_bilateral_filter(result_img, d=5, sigma_color=35, sigma_space=35)
    
    elif grain_ratio > 0.8:  # Grainy noise detected
        print("Detected grainy noise - applying detail-preserving denoising")
        # Use gentler settings for bilateral filter
        bilateral_result = enhanced_bilateral_filter(img, d=7, sigma_color=35, sigma_space=35)
        # Use lower weight for TV-L1 to preserve more detail
        result_img = tv_l1_denoising(bilateral_result, weight=20, iterations=15)
    
    else:  # General noise
        print("General noise - applying gentle NLMeans denoising")
        # Use NLMeans with detail preservation
        result_img = enhanced_nlmeans_color(img, h=10, color_h=10, template_size=7, search_size=21)
    
    return result_img

# Multi-stage stripe removal for difficult cases with detail preservation
def advanced_stripe_removal(img):
    """Apply a more gentle approach to remove horizontal stripes while preserving details"""
    # First pass - frequency domain with gentler settings
    result = remove_horizontal_stripes(img, filter_strength=10)
    
    # For color images, process differently
    if len(img.shape) == 3:
        # Convert to YUV color space to separate luminance from color
        yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
        
        # Apply morphological operations only to Y channel
        y_channel = yuv[:,:,0]
        
        # Create a horizontal kernel for morphological operations
        kernel = np.ones((1, 11), np.uint8)  # Smaller kernel to preserve more detail
        
        # Apply opening operation to remove thin horizontal lines
        opening = cv2.morphologyEx(y_channel, cv2.MORPH_OPEN, kernel)
        
        # Blend with original result to preserve details
        blended_y = cv2.addWeighted(y_channel, 0.8, opening, 0.2, 0)  # More weight to original for detail
        
        # Replace Y channel and convert back to BGR
        yuv[:,:,0] = blended_y
        blended = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Third pass - frequency domain with milder parameters
        final = remove_horizontal_stripes(blended, filter_strength=8)
        
    else:  # Grayscale processing
        # Create a horizontal kernel for morphological operations
        kernel = np.ones((1, 11), np.uint8)  # Smaller kernel
        
        # Apply opening operation to remove thin horizontal lines
        opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        # Blend with original result to preserve details
        blended = cv2.addWeighted(result, 0.8, opening, 0.2, 0)  # More weight to original
        
        # Third pass - frequency domain with milder parameters
        final = remove_horizontal_stripes(blended, filter_strength=8)
    
    return final

# Function to apply specific denoising directly without interactive tuning
def apply_denoising(img, method="adaptive", h=10, color_h=10, d=9, sigma=35, 
                   filter_strength=8, weight=35, iterations=20):
    """
    Apply a specific denoising method with given parameters
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image to denoise
    method : str
        One of "adaptive", "nlm", "bilateral", "tv_l1", "stripe", "advanced_stripe"
    h, color_h, d, sigma, filter_strength, weight, iterations : int
        Method-specific parameters
    
    Returns:
    --------
    numpy.ndarray
        Denoised image
    """
    method = method.lower()
    
    if method == "adaptive":
        return adaptive_denoising(img)
        
    elif method == "nlm":
        return enhanced_nlmeans_color(img, h=h, color_h=color_h)
        
    elif method == "bilateral":
        return enhanced_bilateral_filter(img, d=d, sigma_color=sigma, sigma_space=sigma)
        
    elif method == "tv_l1":
        return tv_l1_denoising(img, weight=weight, iterations=iterations)
        
    elif method == "stripe":
        return remove_horizontal_stripes(img, filter_strength=filter_strength)
        
    elif method == "advanced_stripe":
        return advanced_stripe_removal(img)
        
    else:
        print(f"Unknown method: {method}. Using adaptive denoising instead.")
        return adaptive_denoising(img)

# Process image or video with specified method and parameters
def process_file(file_path, output_path=None, method="adaptive", h=10, color_h=10, 
                d=9, sigma=35, filter_strength=8, weight=35, iterations=20):
    """
    Process an image or video file with specified denoising method
    
    Parameters:
    -----------
    file_path : str
        Path to input image or video
    output_path : str or None
        Path to save output. If None, creates a path based on input file
    method : str
        Denoising method to use
    Other parameters:
        Method-specific parameters
    
    Returns:
    --------
    str
        Path to output file
    """
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_denoised{ext}"
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        # Process image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Could not read image file {file_path}")
            return None
        
        # Apply denoising
        result = apply_denoising(img, method, h, color_h, d, sigma, filter_strength, weight, iterations)
        
        # Calculate metrics
        metrics = calculate_data_loss(img, result)
        
        # Save output
        cv2.imwrite(output_path, result)
        
        print(f"Image processed and saved to {output_path}")
        print("Data Loss Metrics:")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Frequency Loss: {metrics['frequency_loss_percent']:.2f}%")
        print(f"Detail Loss: {metrics['detail_loss_percent']:.2f}%")
        print(f"Data Preserved: {metrics['data_preserved_percent']:.2f}%")
        
        # Return the processed image and metrics
        return output_path, metrics
    
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {file_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output file
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        # Process each frame
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Variables to track metrics
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        total_freq_loss = 0
        total_detail_loss = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply denoising
            denoised_frame = apply_denoising(frame, method, h, color_h, d, sigma, 
                                            filter_strength, weight, iterations)
            
            # Calculate metrics
            metrics = calculate_data_loss(frame, denoised_frame)
            
            # Accumulate metrics
            total_mse += metrics['mse']
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            total_freq_loss += metrics['frequency_loss_percent']
            total_detail_loss += metrics['detail_loss_percent']
            
            # Write to output
            out.write(denoised_frame)
            
            # Progress update
            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete", end='\r')
        
        # Calculate average metrics
        avg_metrics = {
            'mse': total_mse / frame_count,
            'psnr': total_psnr / frame_count,
            'ssim': total_ssim / frame_count,
            'frequency_loss_percent': total_freq_loss / frame_count,
            'detail_loss_percent': total_detail_loss / frame_count,
            'data_preserved_percent': 100 - (total_detail_loss / frame_count)
        }
        
        # Display summary
        print("\nVideo Processing Complete")
        print(f"Output saved to {output_path}")
        print("Average Metrics:")
        print(f"MSE: {avg_metrics['mse']:.2f}")
        print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"SSIM: {avg_metrics['ssim']:.4f}")
        print(f"Frequency Loss: {avg_metrics['frequency_loss_percent']:.2f}%")
        print(f"Detail Loss: {avg_metrics['detail_loss_percent']:.2f}%")
        print(f"Data Preserved: {avg_metrics['data_preserved_percent']:.2f}%")
        
        cap.release()
        out.release()
        
        return output_path, avg_metrics
    
    else:
        print(f"Unsupported file format: {file_path}")
        return None
