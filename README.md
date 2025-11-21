# My JAX Learning Journey 🚀

This repository documents my journey learning JAX from scratch - from complete beginner to building a real fraud detection system on GPU. I'm sharing everything I learned along the way to help other beginners understand this powerful framework.

**What started as personal notes became a comprehensive tutorial.** If you're new to JAX (like I was) or coming from PyTorch (like I did), this guide walks through everything step-by-step with zero assumptions about prior knowledge.

## 📁 What's Inside

I organized my learning into 6 progressive notebooks, each focusing on one core concept:

```
JAX-Practice/
├── README.md                 # This file (my learning journey overview)
└── learning-path/            # Main tutorial notebooks
    ├── requirements.txt     # All dependencies (GPU-enabled)
    ├── README.md            # Detailed guide for each notebook
    ├── 01-jax-basics.ipynb                     # Where I started: JAX fundamentals
    ├── 02-jit-compilation.ipynb                # Making code 100x faster
    ├── 03-automatic-differentiation.ipynb      # Computing gradients automatically
    ├── 04-vectorization-vmap.ipynb             # Automatic batching magic
    ├── 05-jax-vs-pytorch-comparison.ipynb      # Side-by-side framework comparison
    └── 06-neural-networks-fraud-detection.ipynb # Real ML project on GPU
```

## 📚 What I Learned (And What You'll Learn Too)

As I went through these notebooks, I discovered JAX's superpowers and built confidence with real projects:

### JAX Fundamentals (Notebooks 1-4)
- ✅ How JAX arrays work differently from NumPy (immutability!)
- ✅ Why functional programming matters for ML
- ✅ JIT compilation: making code 10-100x faster on GPU
- ✅ Automatic differentiation: never manually compute gradients again
- ✅ `vmap`: the magic transformation that handles batches automatically
- ✅ Function transformations and composability (the JAX way)

### Real Machine Learning (Notebook 6)
- ✅ Built a fraud detection neural network from scratch
- ✅ Handled extreme class imbalance (577:1 ratio - tough!)
- ✅ Used production metrics (Precision, Recall, F1, PR-AUC)
- ✅ Trained on GPU with 284K real transactions
- ✅ Understood when accuracy is a terrible metric

### JAX vs PyTorch (Notebook 5)
- ✅ When to use JAX vs PyTorch (I now know!)
- ✅ Code pattern differences (functional vs OOP)
- ✅ Performance comparisons on the same tasks
- ✅ How to translate between frameworks

---

## 📖 The Learning Path

Here's how I progressed through the material (and how I recommend you approach it):

| # | Notebook | Time | What I Built | Key Learning |
|---|----------|------|--------------|------------|
| 1 | [JAX Basics](learning-path/01-jax-basics.ipynb) | 20-30 min | Understanding arrays | Immutability, why functional programming |
| 2 | [JIT Compilation](learning-path/02-jit-compilation.ipynb) | 30-40 min | Speed optimization | 100x speedup on GPU, control flow pitfalls |
| 3 | [Auto Differentiation](learning-path/03-automatic-differentiation.ipynb) | 30-40 min | Gradient computation | `jax.grad()` magic, Jacobian, Hessian |
| 4 | [Vectorization](learning-path/04-vectorization-vmap.ipynb) | 30-40 min | Batch processing | `vmap` automatic batching |
| 5 | [JAX vs PyTorch](learning-path/05-jax-vs-pytorch-comparison.ipynb) | 30-40 min | Framework comparison | When to use each, code translation |
| 6 | [Fraud Detection](learning-path/06-neural-networks-fraud-detection.ipynb) | 40-60 min | Real neural network | Everything combined on GPU |

**Total learning time**: ~3-4 hours (I did it over a weekend)

**💡 Pro tip**: Don't rush! I spent extra time in notebooks 2-4 experimenting with the transformations. That hands-on practice made everything click.

---

## 🎯 Who This Is For

**This tutorial is for you if you're like I was:**

✅ Complete JAX beginner (I had zero JAX knowledge when I started)  
✅ Coming from PyTorch (I knew PyTorch first, wanted to learn JAX)  
✅ Comfortable with Python and basic NumPy  
✅ Want to understand ML frameworks deeply (not just use them)  
✅ Need to make informed decisions about JAX vs PyTorch  
✅ Learn by building real projects (not just toy examples)

**This might not be for you if:**

❌ You need a quick reference (check JAX docs instead)  
❌ You want advanced research topics only  
❌ You prefer watching videos to hands-on coding

---

## 💡 My Learning Philosophy (And Yours)

As I worked through these notebooks, I followed this approach (and I recommend you do too):

1. **Read first** - Understand the WHY before the HOW
2. **Run everything** - See the code execute (don't just read it)
3. **Break things** - Modify code, introduce errors, learn from them
4. **Take notes** - I kept a learning journal (these notebooks are the result!)
5. **Build progressively** - Each notebook builds on the previous one

**I explained every concept assuming "past me" was reading** - someone who knew Python and NumPy but nothing about JAX or modern ML frameworks. If something confused me, I made sure to document it clearly.

---

---

## 🚀 Getting Started - Installation & Setup

I run JAX on an NVIDIA GPU (CUDA-enabled), which makes training significantly faster. Here's exactly how I set everything up, with options for both GPU and CPU.

### System Requirements

**What I'm using:**
- OS: Windows 11 (also works on Linux/macOS - instructions below)
- GPU: NVIDIA RTX 5060 with 8GB VRAM
- CUDA: Version 13.0
- Python: 3.13
- RAM: 24GB (8GB minimum works fine)

**Don't have a GPU?** No problem! Everything works on CPU (just slower for training in notebook 6).

---

### Step 0: Create a Virtual Environment (Highly Recommended!)

I always use virtual environments to keep dependencies isolated. Here are all the options - pick what works for you:

#### Option A: Anaconda/Miniconda (What I Use)

**Why I chose this:** Great for data science, handles CUDA dependencies well, easy environment management.

**Windows:**
```powershell
# Create environment with Python 3.13
conda create -n jax-practice python=3.13 -y
conda activate jax-practice
```

**Linux/macOS:**
```bash
# Create environment with Python 3.13
conda create -n jax-practice python=3.13 -y
conda activate jax-practice
```

**Don't have Conda?** Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

---

#### Option B: venv (Python Built-in)

**Why choose this:** No extra installation needed, comes with Python, lightweight.

**Windows:**
```powershell
# Create virtual environment
python -m venv jax-practice

# Activate it
.\jax-practice\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv jax-practice

# Activate it
source jax-practice/bin/activate
```

---

#### Option C: uv (Modern & Fast)

**Why choose this:** Extremely fast, modern Python package installer, gaining popularity.

**Install uv first:**
```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Then create environment:**
```bash
# All platforms
uv venv jax-practice

# Activate (Windows)
.\jax-practice\Scripts\activate

# Activate (Linux/macOS)
source jax-practice/bin/activate
```

---

#### Option D: virtualenv (Alternative)

**Why choose this:** More features than venv, cross-Python version support.

```bash
# Install virtualenv first
pip install virtualenv

# Create environment (all platforms)
virtualenv jax-practice

# Activate (Windows)
.\jax-practice\Scripts\activate

# Activate (Linux/macOS)
source jax-practice/bin/activate
```

---

**My recommendation:** Use **Anaconda** if you're doing data science/ML (best for CUDA), or **venv** if you want simplicity (no extra install).

**Important:** Make sure your environment is activated before installing packages! You should see `(jax-practice)` in your terminal prompt.

---

### Step 1: Clone This Repository

```bash
# All platforms
git clone https://github.com/YOUR_USERNAME/JAX-Practice.git
cd JAX-Practice
```

Or **download as ZIP** if you prefer.

---

### Step 2: Check Your GPU (If You Have One)

First, I checked if my NVIDIA GPU was detected:

```bash
# Check if NVIDIA GPU is available (all platforms)
nvidia-smi
```

**What you should see:**
- GPU name (e.g., "NVIDIA GeForce RTX 5060")
- CUDA version (e.g., "CUDA Version: 13.0")
- Driver version

**If `nvidia-smi` doesn't work:**
1. You might not have an NVIDIA GPU (use CPU version instead)
2. Or you need to install NVIDIA drivers:
   - **Windows/Linux:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - **Linux:** May also install via package manager (`sudo apt install nvidia-driver-XXX` on Ubuntu)

---

### Step 3: Install CUDA Toolkit (GPU Users Only)

**JAX requires CUDA installed on your system to use GPU.** PyTorch bundles CUDA in its wheels (easier!), but JAX needs CUDA Toolkit installed separately.

I installed CUDA 13.0 for my RTX 5060:

#### Windows:

1. **Install NVIDIA Driver (if not already installed)**:
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - **Minimum driver version:** 580+ for CUDA 13 on Windows

2. **Download CUDA Toolkit 13.0**: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select: Windows → x86_64 → Version 13.x → Installer type (exe recommended)

3. **Install CUDA Toolkit**:
   - Run the installer
   - Choose "Custom" installation
   - Make sure "CUDA" is checked (default)
   - Installation takes ~10 minutes

4. **Verify installation**:
   ```powershell
   nvcc --version
   ```
   Should show: `Cuda compilation tools, release 13.x`

#### Linux:

1. **Install NVIDIA Driver (if not already installed)**:
   - **Minimum driver version:** 580+ for CUDA 13 on Linux
   ```bash
   # Ubuntu/Debian example
   sudo apt install nvidia-driver-580
   ```

2. **Install CUDA 13.0**:
   ```bash
   # Example for Ubuntu 22.04
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-13-0
   
   # Verify installation
   nvcc --version
   ```

#### macOS:

**Note:** NVIDIA CUDA is not supported on macOS with Apple Silicon (M1/M2/M3/M4). Use CPU version instead.

---

**Important Notes:**
- JAX supports NVIDIA GPUs with **SM version 7.5+** on CUDA 13 (RTX 20-series and newer)
- Make sure `LD_LIBRARY_PATH` is not set on Linux, as it can override CUDA libraries
- For detailed setup or troubleshooting, see: [JAX Installation Guide](https://docs.jax.dev/en/latest/installation.html)

---

### Step 4: Install Dependencies

I created a `requirements.txt` with everything needed (GPU versions included).

#### Option A: GPU Installation (NVIDIA GPU + CUDA 13)

**What I use with RTX 5060 + CUDA 13.0:**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install JAX with CUDA 13 support (pip wheels method - recommended by JAX team)
pip install --upgrade "jax[cuda13]"

# Install PyTorch with CUDA 13 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install remaining dependencies
pip install -r learning-path/requirements.txt
```

**Important:**
- JAX CUDA 13 wheels are only available on **Linux** (x86_64 and aarch64)
- Windows users: JAX will use CUDA via system installation
- Make sure `LD_LIBRARY_PATH` is not set (can cause version conflicts)

**For other setups or troubleshooting:**
- JAX Installation Guide: https://docs.jax.dev/en/latest/installation.html
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/

#### Option B: CPU-Only Installation (No GPU)

```bash
# Install CPU-only JAX (all platforms)
pip install jax[cpu]

# Install CPU-only PyTorch
pip install torch torchvision

# Install remaining dependencies
pip install -r learning-path/requirements.txt
```

**Why separate commands for PyTorch?** PyTorch needs special index URLs for CUDA support, so we can't put it directly in `requirements.txt` with the specific version we want.

---

### Step 5: Verify GPU Detection

After installation, I verified that JAX and PyTorch detected my GPU:

**Test JAX:**
```python
import jax
print(jax.devices())
# Should show: [cuda(id=0)] if GPU is available
# Or: [cpu(id=0)] if using CPU
```

**Test PyTorch:**
```python
import torch
print(torch.cuda.is_available())  # Should print: True (if GPU available)
print(torch.cuda.get_device_name(0))  # Should show: NVIDIA GeForce RTX 5060
```

I made sure both frameworks saw my GPU before starting the notebooks!

**Troubleshooting:** If GPU isn't detected, restart your terminal/IDE to reload environment variables after CUDA installation.

---

### Step 6: Launch Jupyter & Start Learning

```bash
# Navigate to notebooks folder (all platforms)
cd learning-path

# Launch Jupyter Notebook
jupyter notebook
```

**Your browser will open automatically.** Click on `01-jax-basics.ipynb` to start!

---

## 🎯 Quick Start Summary

**If you have NVIDIA GPU (like me):**
```bash
# 1. Create virtual environment (conda, venv, uv, etc.)
# 2. Install CUDA Toolkit 13.0 from NVIDIA website
# 3. Clone repo and install dependencies
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r learning-path/requirements.txt
cd learning-path
jupyter notebook
```

**If you're using CPU only:**
```bash
# 1. Create virtual environment
# 2. Clone repo and install dependencies
pip install jax[cpu]
pip install torch torchvision  
pip install -r learning-path/requirements.txt
cd learning-path
jupyter notebook
```

**That's it!** Open `01-jax-basics.ipynb` and start your JAX journey! 🚀

---

---

## 📊 What Makes This Different

When I started learning JAX, most tutorials either:
- Assumed I already knew advanced ML concepts (I didn't!)
- Used toy synthetic data (not realistic)
- Skipped the "why" and just showed code

**This tutorial is what I wish I had found:**

1. **Beginner-friendly**: I explain everything assuming you're like "past me"
2. **Real data**: Notebook 6 uses actual credit card fraud data (284K transactions)
3. **GPU-focused**: I show how to set up and use GPU acceleration properly
4. **Progressive**: Each notebook builds on the previous (no jumps)
5. **Dual purpose**: My learning journal + comprehensive tutorial for you
6. **Side-by-side**: JAX vs PyTorch comparison on the same tasks

---

## 🎓 What I Discovered: JAX vs PyTorch

After building the same models in both frameworks, here's what I learned:

### Use JAX When:
- You need custom operations or research algorithms
- You want composable transformations (jit, grad, vmap together!)
- You prefer functional programming style
- You need per-sample gradients or higher-order derivatives
- NumPy-like code that's GPU-ready appeals to you

### Use PyTorch When:
- You need quick prototyping with familiar OOP patterns
- You want a massive ecosystem (torchvision, pre-trained models)
- You're deploying to production (more mature tooling)
- You prefer object-oriented programming
- You want extensive community support

**The truth?** After learning both, I now use:
- **JAX** for research and custom experiments
- **PyTorch** for production deployments

Knowing both made me a better ML engineer!

---

---

## 📊 The Dataset: Credit Card Fraud Detection

The final notebook (my favorite!) uses **real-world fraud detection data**. This wasn't easy - fraud detection has unique challenges:

**Dataset Details:**
- **Name**: Credit Card Fraud Detection (from Kaggle/OpenML)
- **Size**: 284,807 real transactions from European cardholders
- **Features**: 30 numerical features (V1-V28 via PCA, plus Time and Amount)
- **Target**: Binary - fraud (1) or legitimate (0)
- **Imbalance**: Extreme! Only 492 frauds out of 284,807 (0.172% fraud rate = 577:1 ratio)

**Why this dataset challenged me:**
- Severe class imbalance (can't just use accuracy!)
- Real-world patterns (not synthetic clean data)
- Forces proper evaluation metrics (Precision, Recall, F1, PR-AUC)
- Shows how JAX handles production ML scenarios

**The dataset auto-downloads when you run notebook 6** - no manual steps needed!

This is where theory met practice for me, and where JAX's performance really shined on GPU.

---

---

## 🛠️ Technical Stack

Here's everything I use in these notebooks:

### Core Frameworks (GPU-Enabled)
- **JAX** 0.4.20+ with CUDA 12 support
  - NumPy-compatible arrays
  - Functional transformations (jit, grad, vmap, pmap)
  - Automatic GPU acceleration
  
- **PyTorch** 2.0.0+ with CUDA support
  - Deep learning framework for comparison
  - Object-oriented neural network modules
  - Mature ecosystem

### Data & Scientific Computing
- **NumPy** 1.24.0+ - Numerical computing foundation
- **Polars** 0.19.0+ - Fast dataframes (I found it 3-5x faster than Pandas!)
- **scikit-learn** 1.3.0+ - Metrics, preprocessing, dataset loading

### Development Environment
- **Jupyter** 1.0.0+ - Interactive notebooks
- **matplotlib** & **seaborn** - Visualization (for understanding results)

### Hardware I Use
- **GPU**: NVIDIA RTX 5060 with 8GB VRAM (CUDA 13.0)
- **RAM**: 24GB (8GB minimum is fine)
- **OS**: Windows 11 (also works on Linux/macOS)
- **Python**: 3.13

**CPU-only users:** Everything works! Just slower for training (~5-10x).

---

---

## ❓ Questions I Had (You Probably Have Them Too)

### Q: Do I need a GPU?
**My answer:** Not required, but highly recommended! Here's what I experienced:
- **CPU**: Everything works fine. Notebooks 1-5 are fast. Notebook 6 training takes ~5-10 minutes.
- **GPU**: Notebooks 1-5 slightly faster. Notebook 6 training takes ~30-60 seconds (10x speedup!).

**For learning, CPU is fine. For real projects, GPU makes a huge difference.**

### Q: Why GPU for JAX but not mentioned for PyTorch?
**What I learned:** Both need GPU setup!
- **JAX**: Requires CUDA Toolkit installed separately on your system
- **PyTorch**: Bundles CUDA in the wheel (easier!), but still needs NVIDIA drivers

That's why I included both JAX and PyTorch GPU setup instructions above.

### Q: I know NumPy. Will I understand JAX?
**My experience:** Yes! JAX is "NumPy with superpowers". If you know NumPy:
- 80% of JAX syntax is identical
- Main difference: immutability (can't modify arrays in-place)
- The transformations (jit, grad, vmap) are new, but I explain them from scratch

### Q: Should I learn JAX or PyTorch?
**Short answer:** Both! But if you must choose:
- **Learn JAX first if:** You're doing research, want fine-grained control, or prefer functional programming
- **Learn PyTorch first if:** You're building production apps, want quick results, or prefer OOP

See notebook 5 for my detailed comparison after using both extensively.

### Q: How long did it take you to complete this?
**Honestly?** About 2-3 days total:
- Day 1: Notebooks 1-3 (fundamentals)
- Day 2: Notebooks 4-5 (vectorization and comparison)
- Day 3: Notebook 6 (fraud detection project)

I recommend taking breaks between notebooks to let concepts sink in!

### Q: What if I get stuck?
**What helped me:**
1. Read error messages carefully (JAX errors are actually informative!)
2. Check the notebook documentation (I explain most gotchas)
3. Review previous notebooks (concepts build progressively)
4. Use `jax.devices()` to verify GPU is detected
5. Check [JAX FAQ](https://jax.readthedocs.io/en/latest/faq.html) and [PyTorch Forums](https://discuss.pytorch.org/)

### Q: Can I skip notebooks?
**My recommendation:**
- Notebooks 1-4: DON'T skip! They build on each other.
- Notebook 5: Can skip if you only care about JAX (but comparison is insightful)
- Notebook 6: Requires understanding from 1-4

I tried jumping ahead once - ended up confused and had to go back!

---

---

## 🤝 Contributing

Found something confusing? Have suggestions? I'd love to hear from you!

**This started as my personal learning journal** - if it helped you, let's make it even better:

1. **Found a typo or error?** Open an issue or PR
2. **Have a better explanation?** Share it!
3. **Built something cool with JAX?** I'd love to see it!

**Please keep the beginner-friendly tone** - remember, we're helping "past us" who knew nothing about JAX.

---

## 📜 License & Attribution

- **Code**: Free to use, modify, and share (MIT License)
- **Dataset**: Credit Card Fraud Detection from Kaggle/OpenML (public domain)
- **This is a learning project** - use it however helps your JAX journey!

If you share or build upon this, a link back is appreciated (but not required). 🙏

---

## 📖 Resources That Helped Me

### Official Docs (I referenced these constantly)
- [JAX Documentation](https://jax.readthedocs.io/) - The official guide
- [JAX GitHub](https://github.com/google/jax) - Source code and issues
- [PyTorch Documentation](https://pytorch.org/docs/) - For comparison
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Great for understanding concepts

### JAX Ecosystem (For when you outgrow this tutorial)
- [Flax](https://flax.readthedocs.io/) - High-level neural network library
- [Optax](https://optax.readthedocs.io/) - Advanced optimizers
- [Haiku](https://dm-haiku.readthedocs.io/) - DeepMind's neural network library
- [Awesome JAX](https://github.com/n2cholas/awesome-jax) - Curated list of JAX resources

### CUDA & GPU Setup
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration
- [JAX GPU Installation](https://jax.readthedocs.io/en/latest/installation.html) - Official GPU guide
- [PyTorch GPU Setup](https://pytorch.org/get-started/locally/) - GPU installation guide

---

## 🎉 Ready to Start Your Journey?

Here's your onboarding checklist:

**Phase 1: Setup (15-30 minutes)**
- [ ] Check if you have NVIDIA GPU (`nvidia-smi`)
- [ ] Install CUDA Toolkit (GPU users) or skip (CPU users)
- [ ] Clone this repository
- [ ] Install dependencies with `learning-path/requirements.txt` (GPU or CPU version)
- [ ] Verify GPU detection (JAX and PyTorch)
- [ ] Launch Jupyter Notebook

**Phase 2: Learning (3-4 hours)**
- [ ] Notebook 1: JAX Basics (20-30 min)
- [ ] Notebook 2: JIT Compilation (30-40 min)
- [ ] Notebook 3: Automatic Differentiation (30-40 min)
- [ ] Notebook 4: Vectorization (30-40 min)
- [ ] Notebook 5: JAX vs PyTorch (30-40 min)
- [ ] Notebook 6: Fraud Detection (40-60 min)

**Phase 3: Beyond**
- [ ] Experiment with the code (break things!)
- [ ] Try your own datasets
- [ ] Explore JAX ecosystem (Flax, Optax)
- [ ] Build your own project

**Installation reminder:**

```bash
# GPU version (NVIDIA + CUDA 13.0)
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r learning-path/requirements.txt

# CPU version (no GPU)
pip install jax[cpu]
pip install torch torchvision
pip install -r learning-path/requirements.txt

# Then start
cd learning-path
jupyter notebook
```

**The journey from beginner to JAX practitioner starts now!** 🚀

---

## ⭐ If This Helped You

This project started as my personal learning notes. If it helped you too:

- ⭐ **Star this repo** - It helps others find it
- 📢 **Share with friends** - Help others learning JAX
- 💬 **Give feedback** - Issues and PRs welcome!
- 🔗 **Link to it** - If you reference it in your work

**Happy learning, and welcome to the JAX community!** 🎓

---

*Last updated: November 2025 | Built with ❤️ while learning JAX on Windows 11 + NVIDIA RTX GPU*