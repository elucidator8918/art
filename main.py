import streamlit as st
import numpy as np
import torch as t
from PIL import Image, ImageFilter
import io
import base64
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import math

# Set page config
st.set_page_config(
    page_title="Computational Thread Art",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #667eea;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .color-swatch {
        display: inline-block;
        width: 30px;
        height: 30px;
        margin: 5px;
        border: 2px solid #000;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def dist(p0, p1):
    """Calculate distance between two points"""
    δ = np.subtract(p1, p0)
    return (δ**2).sum()**0.5

def through_pixels(p0, p1):
    """Get pixels along a line between two points"""
    δ = np.subtract(p1, p0)
    distance = (δ**2).sum()**0.5
    
    if distance == 0:
        return np.array([p0]).T
    
    pixels_in_line = p0 + np.outer((np.arange(int(distance) + 1) / distance), δ)
    return pixels_in_line.T

def get_color_hash(c):
    """Generate hash for color tensor"""
    return 256*256*c[0] + 256*c[1] + c[2]

def get_img_hash(i):
    """Generate hash for image tensor"""
    return 256*256*i[:,:,0] + 256*i[:,:,1] + i[:,:,2]

# Main classes
@dataclass
class ThreadArtParams:
    """Parameters for thread art generation"""
    name: str
    x: int
    n_nodes: int
    palette: Dict[str, List[int]]
    n_lines_per_color: List[int]
    n_random_lines: int
    darkness: float
    blur_rad: int
    group_orders: str
    shape: str = "Rectangle"
    
class ImageProcessor:
    """Handles image processing and dithering"""
    
    def __init__(self, image: Image.Image, target_x: int, palette: Dict[str, List[int]]):
        self.original = image
        self.x = target_x
        self.y = int(target_x * (image.height / image.width))
        self.palette = {k: tuple(v) for k, v in palette.items()}
        
        # Resize and convert image
        base_image = image.resize((self.x, self.y))
        self.imageRGB = t.tensor(base_image.convert("RGB").getdata()).reshape((self.y, self.x, 3))
        self.imageBW = t.tensor(base_image.convert("L").getdata()).reshape((self.y, self.x))
        
    def floyd_steinberg_dither(self, progress_callback=None):
        """Apply Floyd-Steinberg dithering"""
        image_dithered = self.imageRGB.clone().to(t.float)
        palette_tensor = t.tensor(list(self.palette.values())).to(t.float)
        palette_sq = palette_tensor.unsqueeze(1)
        
        y, x = image_dithered.shape[:2]
        
        total_pixels = y * x
        processed = 0
        
        for y_ in range(y - 1):
            row = image_dithered[y_].to(t.float)
            next_row = t.zeros_like(row)
            
            for x_ in range(x - 1):
                old_color = row[x_]
                color_diffs = (palette_sq - old_color).pow(2).sum(axis=-1)
                color = palette_tensor[color_diffs.argmin(dim=0)]
                color_diff = old_color - color
                
                row[x_] = color
                row[x_+1] += (7/16) * color_diff
                
                if x_ > 0:
                    next_row[x_-1] += (3/16) * color_diff
                next_row[x_] += (5/16) * color_diff
                next_row[x_+1] += (1/16) * color_diff
                
                processed += 1
                if progress_callback and processed % 1000 == 0:
                    progress_callback(processed / total_pixels)
            
            image_dithered[y_] = t.clamp(row, 0, 255)
            image_dithered[y_+1] += next_row
            image_dithered[y_+1] = t.clamp(image_dithered[y_+1], 0, 255)
        
        self.image_dithered = image_dithered.to(t.int)
        return self.image_dithered
    
    def generate_mono_images(self):
        """Generate monochrome images for each color"""
        mono_images = {}
        histogram = {}
        
        for color_name, color_value in self.palette.items():
            mono_image = (get_img_hash(self.image_dithered) == 
                         get_color_hash(t.tensor(color_value))).to(t.int)
            mono_images[color_name] = mono_image
            histogram[color_name] = mono_image.sum().item() / (self.x * self.y)
        
        return mono_images, histogram

def build_coordinate_system(x, y, n_nodes, shape="Rectangle"):
    """Build coordinate system for thread art nodes"""
    d_coords = {}
    
    if shape == "Rectangle":
        nx = 2 * int(n_nodes * 0.25 * x / (x + y))
        ny = 2 * int(n_nodes * 0.25 * y / (x + y))
        
        while 2 * (nx + ny) < n_nodes:
            if ny >= nx: ny += 2
            else: nx += 2
        
        xd = (x - 1) / nx
        yd = (y - 1) / ny
        
        idx = 0
        # Top edge (right to left)
        for i in range(nx):
            d_coords[idx] = (0, x - 1 - i * xd)
            idx += 1
        # Left edge (top to bottom)
        for i in range(ny):
            d_coords[idx] = (i * yd, 0)
            idx += 1
        # Bottom edge (left to right)
        for i in range(nx):
            d_coords[idx] = (y - 1, i * xd)
            idx += 1
        # Right edge (bottom to top)
        for i in range(ny):
            d_coords[idx] = (y - 1 - i * yd, x - 1)
            idx += 1
    
    elif shape == "Ellipse":
        angles = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]
        x_coords = 1 + ((0.5*x) - 2) * (1 + np.cos(angles))
        y_coords = 1 + ((0.5*y) - 2) * (1 - np.sin(angles))
        
        for i, (y_coord, x_coord) in enumerate(zip(y_coords, x_coords)):
            d_coords[i] = (y_coord, x_coord)
    
    return d_coords

# Streamlit App
def main():
    st.markdown('<h1 class="main-header">🧵 Computational Thread Art</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Transform your images into stunning thread art using computational algorithms. 
    This tool uses Floyd-Steinberg dithering and a greedy line-placement algorithm to create 
    beautiful string art patterns.
    """)
    
    # Sidebar - Parameters
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Project name
        project_name = st.text_input("Project Name", "my_thread_art")
        
        # Shape selection
        shape = st.selectbox("Shape", ["Rectangle", "Ellipse"])
        
        # Image dimensions
        st.subheader("Image Dimensions")
        target_width = st.slider("Target Width (pixels)", 400, 1500, 800, 50)
        
        # Nodes
        n_nodes = st.slider("Number of Nodes", 100, 800, 500, 50)
        
        # Color palette
        st.subheader("Color Palette")
        n_colors = st.slider("Number of Colors", 2, 6, 3)
        
        palette = {}
        color_names = []
        for i in range(n_colors):
            col1, col2 = st.columns([1, 2])
            with col1:
                color = st.color_picker(f"Color {i+1}", 
                                       ["#FFFFFF", "#FF0000", "#0000FF", "#00FF00", "#000000"][i % 5],
                                       key=f"color_{i}")
            with col2:
                name = st.text_input(f"Name {i+1}", 
                                    ["white", "red", "blue", "green", "black"][i % 5],
                                    key=f"name_{i}")
            
            # Convert hex to RGB
            rgb = tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            palette[name] = list(rgb)
            color_names.append(name)
        
        # Algorithm parameters
        st.subheader("Algorithm Parameters")
        n_random_lines = st.slider("Random Lines per Selection", 50, 300, 150, 10)
        darkness = st.slider("Darkness", 0.05, 0.30, 0.15, 0.01)
        blur_rad = st.slider("Blur Radius", 0, 10, 4, 1)
        
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🎨 Process", "🖼️ Generate", "📊 Analysis"])
    
    with tab1:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.info(f"""
                **Image Info:**
                - Size: {image.width} x {image.height}
                - Mode: {image.mode}
                - Format: {image.format}
                """)
            
            # Store in session state
            if 'image' not in st.session_state or st.session_state.image != uploaded_file:
                st.session_state.image = uploaded_file
                st.session_state.pil_image = image
                st.session_state.processed = False
    
    with tab2:
        st.markdown('<div class="section-header">Process Image</div>', unsafe_allow_html=True)
        
        if 'pil_image' not in st.session_state:
            st.warning("Please upload an image first!")
        else:
            if st.button("🎨 Apply Floyd-Steinberg Dithering", type="primary"):
                with st.spinner("Processing image..."):
                    # Create processor
                    processor = ImageProcessor(st.session_state.pil_image, target_width, palette)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"Dithering: {int(pct*100)}%")
                    
                    # Dither
                    start_time = time.time()
                    dithered = processor.floyd_steinberg_dither(update_progress)
                    process_time = time.time() - start_time
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Dithering complete in {process_time:.2f}s")
                    
                    # Generate mono images
                    mono_images, histogram = processor.generate_mono_images()
                    
                    # Store results
                    st.session_state.processor = processor
                    st.session_state.mono_images = mono_images
                    st.session_state.histogram = histogram
                    st.session_state.processed = True
                    
                    st.success("Image processed successfully!")
            
            # Display results if processed
            if st.session_state.get('processed', False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Dithered Image")
                    dithered_img = st.session_state.processor.image_dithered.numpy().astype(np.uint8)
                    st.image(dithered_img, use_column_width=True)
                
                with col2:
                    st.subheader("Color Distribution")
                    hist_df = pd.DataFrame(
                        list(st.session_state.histogram.items()),
                        columns=['Color', 'Percentage']
                    )
                    hist_df['Percentage'] = hist_df['Percentage'] * 100
                    st.bar_chart(hist_df.set_index('Color'))
                
                # Show mono images
                st.subheader("Monochrome Layers")
                cols = st.columns(len(palette))
                for idx, (color_name, mono_img) in enumerate(st.session_state.mono_images.items()):
                    with cols[idx]:
                        mono_display = (mono_img.numpy() * 255).astype(np.uint8)
                        st.image(mono_display, caption=color_name, use_column_width=True)
                
                # Suggest line counts
                st.subheader("Suggested Line Counts")
                total_lines = st.slider("Total Lines", 5000, 25000, 12000, 1000)
                
                suggested_lines = []
                for color_name in palette.keys():
                    n_lines = int(st.session_state.histogram[color_name] * total_lines)
                    suggested_lines.append(n_lines)
                
                # Adjust to match total
                darkest_idx = np.argmax([sum(v) for v in palette.values()])
                suggested_lines[darkest_idx] += (total_lines - sum(suggested_lines))
                
                line_df = pd.DataFrame({
                    'Color': list(palette.keys()),
                    'RGB': [str(tuple(v)) for v in palette.values()],
                    'Suggested Lines': suggested_lines
                })
                st.dataframe(line_df, use_container_width=True)
                
                st.session_state.suggested_lines = suggested_lines
    
    with tab3:
        st.markdown('<div class="section-header">Generate Thread Art</div>', unsafe_allow_html=True)
        
        if not st.session_state.get('processed', False):
            st.warning("Please process an image first!")
        else:
            # Line count inputs
            st.subheader("Line Counts per Color")
            n_lines_per_color = []
            cols = st.columns(len(palette))
            
            for idx, color_name in enumerate(palette.keys()):
                with cols[idx]:
                    default_val = st.session_state.get('suggested_lines', [1000]*len(palette))[idx]
                    n_lines = st.number_input(
                        color_name,
                        min_value=0,
                        max_value=10000,
                        value=default_val,
                        step=100
                    )
                    n_lines_per_color.append(n_lines)
            
            # Group orders
            st.subheader("Layer Order")
            color_letters = ''.join([name[0] for name in palette.keys()])
            group_orders = st.text_input(
                "Group Order (use first letters)",
                value=color_letters * 4,
                help="Order in which colors are layered. Repeat letters for multiple passes."
            )
            
            if st.button("🎨 Generate Thread Art", type="primary"):
                st.info("⚠️ Note: Full thread art generation requires the complete coordinate system implementation. This demo shows the setup.")
                
                with st.spinner("Building coordinate system..."):
                    # Build coordinates
                    processor = st.session_state.processor
                    d_coords = build_coordinate_system(
                        processor.x,
                        processor.y,
                        n_nodes,
                        shape
                    )
                    
                    # Visualize node placement
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    # Draw image
                    ax.imshow(processor.imageRGB.numpy())
                    
                    # Draw nodes
                    for idx, (y, x) in d_coords.items():
                        ax.plot(x, y, 'ro', markersize=2)
                    
                    ax.set_title(f"Node Placement ({n_nodes} nodes)")
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    
                    st.success(f"Coordinate system built with {len(d_coords)} nodes!")
                    
                    st.info("""
                    **Next Steps:**
                    The full implementation would:
                    1. Build the pixel dictionary for all node pairs
                    2. Apply the greedy line selection algorithm
                    3. Generate SVG output
                    4. Create PDF instructions for physical creation
                    
                    This requires significant computation time and memory.
                    """)
    
    with tab4:
        st.markdown('<div class="section-header">Analysis & Export</div>', unsafe_allow_html=True)
        
        st.info("Analysis tools and export options will appear here after generation.")
        
        # Placeholder for future features
        st.subheader("Future Features")
        st.markdown("""
        - **SVG Export**: Download vector graphics
        - **PDF Instructions**: Physical creation guide
        - **Animation**: Watch the thread art build
        - **Statistics**: Line lengths, coverage analysis
        - **Optimization**: Fine-tune parameters
        """)

if __name__ == "__main__":
    main()
