# Text-to-Image-AI

Advanced AI-powered text-to-image conversion using Stable Diffusion, CLIP embeddings, and RealESRGAN upscaling. Generate high-quality images from text prompts with multiple optimization options and quality metrics.

## 🎯 Overview

This project implements a comprehensive text-to-image generation system that combines state-of-the-art deep learning models to convert textual descriptions into high-quality, visually appealing images. The system leverages multiple advanced techniques for image generation and enhancement.

## ✨ Key Features

### Core Generation Capabilities
- **Stable Diffusion Integration**: Uses the powerful Stable Diffusion model (v1.5 or Turbo variant) for fast and efficient image generation
- **CLIP Embeddings**: Implements CLIP vision-language model for computing similarity scores between generated images and text prompts
- **RealESRGAN Upscaling**: AI-powered super-resolution enhancement for generating high-resolution images (2x or 4x upscaling)
- **Fallback PIL Upscaling**: Intelligent fallback to PIL/Lanczos resampling if RealESRGAN is unavailable

### Optimization Features
- **GPU Acceleration**: Full CUDA support with automatic device detection and memory management
- **Model Mode Selection**: Choose between "turbo" (fast generation) and "high" (best quality) modes
- **Memory Efficient**: Automatic garbage collection and cache clearing to optimize resource usage
- **Batch Processing**: UIpport for processing multiple prompts with custom parameters

### Quality Metrics
- **CLIP Similarity Scoring**: Quantitative evaluation of how well the generated image matches the text prompt
- **Flexible Guidance Scale**: Adjust the strength of prompt adherence (guidance scale: 1.0-15.0)
- **Customizable Steps**: Fine-tune generation quality and speed with adjustable inference steps (8-50)
- **Seed Control**: Reproducible image generation with configurable random seeds

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration) - optional but recommended
- At least 8GB RAM (16GB recommended for optimal performance)
- GPU with 6GB+ VRAM (for running Stable Diffusion efficiently)

### Required Libraries
```bash
pip install torch==2.2.0 torchvision==0.17.0
pip install diffusers==0.28.0 transformers==4.38.1
pip install accelerate==0.30.0
pip install huggingface-hub==0.22.2 omegaconf==2.3.0 safetensors==0.4.2
pip install numpy==1.26.4 pandas ipywidgets pillow tqdm
pip install scipy scikit-learn matplotlib seaborn
```

### Optional Libraries
```bash
pip install realesrgan  # For enhanced AI upscaling
```

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/meet2121/Text-to-Image-AI.git
cd Text-to-Image-AI
pip install -r requirements.txt
```

### 2. Running the Notebook

Launch Jupyter and open the notebook:
```bash
jupyter notebook txt_to_img.ipynb
```

### 3. Basic Usage

```python
# Simple image generation
prompt = "A beautiful sunset over mountains, oil painting style"
image = generate_image(
    prompt=prompt,
    steps=30,
    guidance=7.5,
    seed=42
)
image.show()
```

## 📖 Usage Guide

### Image Generation

#### Basic Generation
```python
from txt_to_img import generate_image

image = generate_image("A futuristic city at night")
image.show()
```

#### Advanced Generation with Parameters
```python
image = generate_image(
    prompt="A serene forest lake with mountains in background",
    steps=50,           # Higher steps = better quality but slower
    guidance=8.0,       # Higher guidance = more adherence to prompt
    seed=123            # Fixed seed for reproducibility
)
```

### Quality Assessment

```python
from txt_to_img import clip_similarity

# Calculate CLIP similarity score
score = clip_similarity(prompt, image)
print(f"Similarity Score: {score}")
```

### Image Upscaling

```python
from txt_to_img import upscale_smart

# Intelligent upscaling with RealESRGAN or PIL fallback
upscaled_image = upscale_smart(image, desired_factor=2)  # 2x upscaling
upscaled_image.save('upscaled_image.png')
```

## 🔧 Configuration

### Model Selection
- **Turbo Mode**: Fast generation (~7 seconds), suitable for quick previews
- **High Quality Mode**: Better quality (~30 seconds), suitable for final outputs

### Inference Parameters
- **Steps**: 8-50 (default: 7 for turbo, 30 for high)
- **Guidance Scale**: 1.0-15.0 (default: 7.5)
- **Upscale Factor**: 1x, 2x, or 4x
- **Seed**: Any positive integer

## 📊 Performance Optimization

### Tips for Better Results
1. **Use Descriptive Prompts**: More detailed prompts generally produce better results
2. **Adjust Guidance Scale**: Lower values (3-5) for more creative freedom, higher values (10-15) for prompt adherence
3. **Increase Steps**: Use 30-50 steps for final outputs, 7-15 for previews
4. **GPU Memory**: Monitor GPU usage and clear cache if needed

### Tips for Faster Generation
1. Use turbo mode for quick iterations
2. Reduce the number of inference steps
3. Lower the upscale factor
4. Use smaller guidance scales

## 🏗️ Project Structure

```
Text-to-Image-AI/
├── txt_to_img.ipynb          # Main Jupyter notebook with UI
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── outputs/                  # Directory for saved images
    ├── history.csv           # Generation history log
    └── [generated images]    # Generated and upscaled images
```

## 📈 Workflow Overview

```
User Input (Prompt, Parameters)
         ↓
   Text Encoding (CLIP)
         ↓
  Stable Diffusion Pipeline
         ↓
    Generate Image (512x512)
         ↓
   CLIP Similarity Scoring
         ↓
   Optional Upscaling (2x/4x)
         ↓
Save & Display Results
```

## 🎨 Example Prompts

### Art Styles
- "Oil painting of a serene landscape"
- "Watercolor illustration of a flower garden"
- "Digital artwork of a cyberpunk city"
- "Van Gogh style starry night over a small town"

### Photorealistic
- "A professional photograph of a mountain lake at sunset"
- "High-resolution portrait of a person smiling"
- "Detailed macro photography of a butterfly wing"

### Fantasy
- "A magical forest with glowing trees and mystical creatures"
- "A dragon flying over an ancient castle"
- "Underwater city with bioluminescent architecture"

## 📊 Key Technologies

### Deep Learning Models
- **Stable Diffusion v1.5**: Text-to-image diffusion model
- **CLIP (Vision-Language)**: Multi-modal embedding model
- **RealESRGAN**: Super-resolution generative adversarial network

### Libraries & Frameworks
- **PyTorch**: Deep learning framework
- **Hugging Face Diffusers**: Pre-built diffusion pipelines
- **Transformers**: CLIP model implementation
- **Pillow**: Image processing
- **IPyWidgets**: Interactive UI components

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce the number of inference steps
- Lower the upscale factor
- Use turbo mode instead of high quality
- Restart the kernel and clear GPU cache

### RealESRGAN Not Found
- Install with: `pip install realesrgan`
- If installation fails, PIL upscaling will be used automatically

### Model Download Issues
- Ensure stable internet connection
- Check Hugging Face token if using private models
- Models are cached locally after first download

## 📝 Output Files

Generated images and metadata are saved in the `outputs/` directory:
- **Low-res images**: Original 512x512 output from Stable Diffusion
- **HD upscaled images**: 1024x1024 (2x) or 2048x2048 (4x) versions
- **history.csv**: Log file containing all generation parameters and CLIP scores

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 🔄 Future Enhancements

- [ ] Batch image generation API
- [ ] Web interface using Gradio/Streamlit
- [ ] Additional upscaling models (BSRGAN, SwinIR)
- [ ] Inpainting and image editing capabilities
- [ ] ControlNet integration for guided generation
- [ ] Custom LoRA fine-tuning support
- [ ] Real-time preview during generation
- [ ] Database storage for generation history

## 📚 References

- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [RealESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Meet Brahmaniya**
- GitHub: [@meet2121](https://github.com/meet2121)
- Email: [Your Email]

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion
- OpenAI for CLIP
- Hugging Face for the Diffusers library
- Contributors and users for feedback and improvements

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/meet2121/Text-to-Image-AI/issues) page
2. Create a new issue with detailed description
3. Include error messages and system specifications

---

**Last Updated**: December 2025
**Status**: Active Development
**Version**: 1.0.0
