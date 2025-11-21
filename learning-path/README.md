# My JAX Learning Journey - The Notebooks

This folder contains the 6 notebooks I worked through while learning JAX from scratch. Each notebook represents a key milestone in my journey, starting from "what even is JAX?" to building a real fraud detection system on GPU.

**I'm sharing these as both:**
- 📓 **My personal learning journal** - documenting what worked, what confused me, and breakthrough moments
- 📚 **A tutorial for you** - so you can learn alongside "past me" without the same struggles

No prior JAX or PyTorch knowledge required. Just Python and curiosity!

---

## 🗺️ My Learning Path (Your Roadmap)

Here's how I progressed through the material. Each notebook built on the previous one:

```
01-jax-basics.ipynb
    ↓ "Wait, I can't modify arrays? Why??"
02-jit-compilation.ipynb
    ↓ "Holy speedup Batman! 100x faster?!"
03-automatic-differentiation.ipynb
    ↓ "Gradients... automatically?! This changes everything."
04-vectorization-vmap.ipynb
    ↓ "Write for one example, batch for free. MIND BLOWN."
05-jax-vs-pytorch-comparison.ipynb
    ↓ "Okay, now I see when to use JAX vs PyTorch."
06-neural-networks-fraud-detection.ipynb
    ✓ "I just built a real ML system in JAX on GPU!"
```

**Total time**: I spent 2-3 days on this (with breaks!). You can do it faster or slower - go at your pace.

---

---

## 📓 My Journey Through Each Notebook

### Notebook 1: JAX Basics - "So... it's just NumPy?"
**Time I spent**: 20-30 minutes | **Starting point**: Knew NumPy, zero JAX

**What I learned:**
- JAX looked like NumPy but had a twist: **immutability** (can't modify arrays!)
- Initially frustrated ("Why can't I just do `x[0] = 5`?!"), but then understood why functional programming matters
- Array creation, basic ops - all felt familiar if you know NumPy
- The `.at[].set()` syntax for "modifying" arrays (actually creating new ones)

**Breakthrough moment**: Realizing immutability makes JIT compilation and autodiff possible

**Prerequisites:** Just basic Python

**My advice**: Don't fight immutability - embrace it. It's weird at first but becomes natural.

---

### Notebook 2: JIT Compilation - "This is INSANELY fast"
**Time I spent**: 30-40 minutes (more because I experimented a lot) | **Mind blown**: Yes

**What I learned:**
- One decorator (`@jax.jit`) made my code 10-100x faster on GPU
- **But there's a catch**: data-dependent control flow breaks everything
- Spent 20 minutes debugging why `if` statements didn't work in JIT
- Learned `jnp.where`, `jax.lax.cond` - the JAX way of control flow

**Most painful moment**: "Why isn't my `print()` statement working?!" (It's traced away during compilation)

**Prerequisites:** Notebook 1

**My advice**: Read the pitfalls section carefully. I learned the hard way!

---

### Notebook 3: Automatic Differentiation - "No more manual calculus!"
**Time I spent**: 30-40 minutes | **Flashback**: High school calculus trauma avoided

**What I learned:**
- `jax.grad(fn)` computes **exact** derivatives automatically
- I wrote the forward function, JAX handled the backward pass
- Gradient descent became trivial to implement
- Jacobian, Hessian - intimidating names but JAX makes them easy

**Favorite part**: `value_and_grad()` - get both function value and gradient in one go (efficient!)

**Prerequisites:** Notebook 1, vague memories of calculus (but JAX does the math)

**My advice**: Don't try to understand the math deeply. Trust that JAX is right (it is).

---

### Notebook 4: Vectorization with vmap - "Wait, where's the loop?"
**Time I spent**: 25-35 minutes | **Realization**: I've been writing unnecessary loops my whole life

**What I learned:**
- `vmap` = write code for ONE example, automatically batch it
- No more manually tracking batch dimensions or writing loops
- Per-sample gradients? One line: `jax.vmap(jax.grad(loss_fn))`
- Performance boost was massive (especially combined with JIT)

**"Aha!" moment**: You can compose transformations! `jit(vmap(grad(fn)))` - mind blown!

**Prerequisites:** Notebooks 1-2 (understanding JIT helps)

**My advice**: Start simple with `vmap`, then experiment with `in_axes`. The power will dawn on you.

---

### Notebook 5: JAX vs PyTorch - "Now I get when to use each"
**Time I spent**: 30-40 minutes | **Perspective shift**: Yes

**What I learned:**
- PyTorch: OOP, mutable, batteries-included (nn.Module, optimizers, etc.)
- JAX: functional, immutable, bring-your-own (but composable transformations!)
- Same tasks, different philosophies - side-by-side code comparison was eye-opening
- Now I know which to reach for depending on the project

**Key insight**: It's not "JAX vs PyTorch" - it's "JAX AND PyTorch" (know both!)

**Prerequisites:** Notebooks 1-3 (or come from PyTorch background)

**My advice**: Don't skip this even if you're only interested in JAX. Comparison clarifies JAX's design choices.

---

### Notebook 6: Fraud Detection - "I built something REAL!"
**Time I spent**: 40-60 minutes (+ extra time experimenting) | **Proud moment**: Absolutely

**What I learned:**
- Built a neural network from scratch (no high-level libraries!)
- Trained on 284K real transactions on my RTX GPU
- Handled extreme class imbalance (577:1 ratio - accuracy was useless!)
- Learned when precision matters more than recall (and vice versa)
- Everything from notebooks 1-4 came together: JIT + grad + vmap = magic

**Toughest challenge**: Class imbalance - had to use weighted loss and proper metrics (F1, PR-AUC)

**Most satisfying**: Watching training converge in ~30 seconds on GPU (would've been 10 minutes on CPU)

**Prerequisites:** All previous notebooks (this is the finale!)

**My advice**: Don't rush this. It's where theory meets practice. Try different hyperparameters!

---

## 🚀 Getting Started (How I Set Things Up)

### Step 1: Install Dependencies

I'm running on GPU (NVIDIA RTX), so I installed the CUDA versions. If you don't have a GPU, use the CPU versions instead.

**GPU version (what I use):**
```bash
# From the JAX-Practice root directory
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"  # CUDA 13 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r learning-path/requirements.txt
```

**CPU version (works great too, just slower):**
```bash
pip install jax[cpu]
pip install torch torchvision
pip install -r learning-path/requirements.txt
```

**Note**: See the main README for detailed GPU setup instructions (CUDA Toolkit installation, etc.)

### Step 2: Launch Jupyter

```bash
# Navigate to this folder
cd learning-path

# Start Jupyter
jupyter notebook
```

### Step 3: Start Your Journey!

Open `01-jax-basics.ipynb` and start reading + running cells. That's how I did it!

---

## 📖 How I Used These Notebooks (And How You Can Too)

### If You're a Complete Beginner (Like I Was)
- Start with notebook 1 and go in order
- Run EVERY cell - don't just read
- When something confuses you, re-read that section (I did this a lot!)
- Modify code and break things - that's how you learn
- Take breaks between notebooks to let concepts sink in

### If You Know PyTorch (Like I Did)
- Jump straight to notebook 5 (JAX vs PyTorch) - it'll orient you
- Then do notebooks 1-4 to understand JAX-specific features
- Finish with notebook 6 to see everything in action
- You'll appreciate the functional programming style more this way

### If You're Just Exploring JAX
- Notebooks 1-4 cover all JAX fundamentals
- Skip notebook 5 if you don't care about PyTorch
- Notebook 6 shows how to apply everything to real problems

**My recommendation**: Whatever path you choose, actually RUN the code. I learned 10x more by experimenting than just reading.

---

## 🎯 What You'll Gain (What I Gained)

After completing these 6 notebooks, I can now:

✅ Write GPU-accelerated code using JAX arrays  
✅ Make code 100x faster with JIT compilation  
✅ Compute gradients automatically (never manually deriving again!)  
✅ Use `vmap` for elegant batching (goodbye, messy loops!)  
✅ Understand when to use JAX vs PyTorch (and why)  
✅ Build neural networks from scratch in both frameworks  
✅ Handle real-world ML challenges (class imbalance, proper metrics)  
✅ Make informed framework decisions for my projects  

**Bonus**: I'm now confident reading JAX research papers and implementing custom algorithms!  

---

## 💡 Tips That Helped Me (Will Help You Too)

1. **Run EVERY code cell** - I learned way more by executing than reading. Seeing output matters!
2. **Break things on purpose** - Introduce errors, see what happens. That's how I learned JIT pitfalls.
3. **Read error messages** - JAX errors are actually informative (unlike some frameworks 😄)
4. **Take breaks** - I did one notebook per session, not all at once. Your brain needs processing time.
5. **Experiment** - Modify examples, try different values, test edge cases. That's where "aha!" moments happen.
6. **Keep a learning journal** - I wrote notes about confusing parts. This repo IS that journal!
7. **GPU vs CPU** - If you have GPU, verify it's detected (`jax.devices()`). Made a huge difference for me in notebook 6.

---

## 🆘 Common Issues & Solutions

Here are issues that beginners often encounter (some I experienced, others are common pitfalls):

### "ImportError: No module named 'jax'"
**Solution:**
```bash
cd learning-path
pip install -r requirements.txt
```

### "Code runs slow even with JIT!"
**What's happening:**
- First run is always slow (compilation)
- Second run onwards should be fast
- Check `jax.devices()` - make sure GPU is detected
- Small functions might have JIT overhead (see notebook 2)

### "Print statements don't show up in my JIT function!"
**Common mistake:** Using regular `print()` inside `@jax.jit` decorated functions  
**Solution:** See notebook 2 for debugging JIT functions

### "CUDA errors or GPU not detected"
**Solutions that work:**
- Verify `nvidia-smi` works in terminal
- Check `jax.devices()` shows `[cuda(id=0)]` not `[cpu(id=0)]`
- Restart terminal/IDE after CUDA installation
- CPU version works fine if GPU setup is too complex
- See main README for detailed GPU setup

### "Notebook kernel crashes on notebook 6"
**Fix:** Reduce batch size (default is 1024, try 512 or 256 if RAM is limited)

### "Array shapes don't match after vmap"
**Common solution:** Print shapes at each step. Understanding `in_axes` and `out_axes` (notebook 4) resolves most shape issues.

### "CUDA version mismatch warnings"
**What to know:** Minor CUDA version differences are usually fine. JAX requires CUDA 13 Toolkit installed on your system.

### Need more help?
**Official documentation:**
- JAX Installation: https://docs.jax.dev/en/latest/installation.html
- PyTorch Installation: https://pytorch.org/get-started/locally/

---

## 📊 About the Fraud Detection Dataset (Notebook 6)

This was my first time working with severely imbalanced data - challenging but rewarding!

**Dataset Details:**
- **Source**: Kaggle/OpenML (auto-downloads, no manual setup needed)
- **Size**: 284,807 real credit card transactions
- **Features**: 30 numerical (V1-V28 via PCA + Time + Amount)
- **Target**: Binary - fraud (1) or legitimate (0)
- **Imbalance**: Only 492 frauds out of 284,807 (0.172% fraud rate)

**What made this interesting:**
- Can't just use accuracy (99.8% accuracy = predicting "all legitimate"!)
- Forced me to learn proper metrics (Precision, Recall, F1, PR-AUC)
- Real-world class imbalance patterns (not synthetic clean data)
- Great showcase of JAX's performance on GPU with real data

**Fun fact**: Training this on my RTX 3060 took ~30 seconds. Same code on CPU took ~5 minutes!

---

## 🔗 Resources That Helped Me (Will Help You Too)

**Official docs I referenced constantly:**
- [JAX Documentation](https://jax.readthedocs.io/) - The source of truth
- [JAX GitHub](https://github.com/google/jax) - For understanding internals
- [PyTorch Documentation](https://pytorch.org/docs/) - For comparison work
- [JAX Tutorial (Official)](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) - Good complement to these notebooks

**For going deeper:**
- [Flax](https://flax.readthedocs.io/) - When you outgrow manual neural networks
- [Optax](https://optax.readthedocs.io/) - When you need fancier optimizers than SGD
- [Awesome JAX](https://github.com/n2cholas/awesome-jax) - Curated resources

**My recommendation**: Start with these notebooks, THEN explore the ecosystem. I tried jumping to Flax too early and got confused.

---

## ✨ Why I Created These Notebooks

When I started learning JAX, I struggled to find tutorials that:
- Assumed zero JAX knowledge (most assumed ML research background)
- Explained the "why" not just the "what"
- Used real data instead of toy examples
- Compared JAX with PyTorch side-by-side

**So I documented my learning journey as I went**, writing the tutorial I wish I had found.

If you're reading this, you're getting the benefit of my struggles, "aha!" moments, and hard-won insights. I hope it makes your JAX journey smoother than mine was!

---

## 🎉 Ready to Start?

**Installation:**
```bash
cd learning-path
pip install -r requirements.txt  # See main README for GPU setup
jupyter notebook
```

**Then open `01-jax-basics.ipynb` and begin your journey!**

Remember: Every expert was once a beginner. These notebooks are me teaching "past me" - and now, teaching you.

**Let's learn JAX together!** 🚀

---

*These notebooks represent about 2-3 days of intensive learning, experimentation, and documentation. I hope they save you weeks of trial and error!*
