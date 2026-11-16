# Combinatorial Weight Superposition (CWS)

**Zero-Overhead Cryptographic Leak Attribution and Anti-Monoculture Defense in Neural Malware Classifiers**

**Author:** Manish Kumar Parihar  
**ORCID:** [0009-0002-1900-8945](https://orcid.org/0009-0002-1900-8945)  
**Dataset:** [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html) ($58,596$ Samples)  
**Framework:** PyTorch  

---

## Table of Contents
1. [The Problem: Why Standard AI Antivirus Deployment Fails](#1-the-problem-why-standard-ai-antivirus-deployment-fails)
2. [Our Solution: Combinatorial Weight Superposition (CWS)](#2-our-solution-combinatorial-weight-superposition-cws)
3. [Deep Dive: How CWS Works Step-by-Step](#3-deep-dive-how-cws-works-step-by-step)
   - [Phase 1: Server-Side Superposition Training & `min_gap` Constraint](#phase-1-server-side-superposition-training--min_gap-constraint)
   - [Phase 2: Input-Gradient Decorrelation & Anti-Transfer Defense](#phase-2-input-gradient-decorrelation--anti-transfer-defense)
   - [Phase 3: Cryptographic Deployment (`HMAC-SHA256`) with Zero Overhead](#phase-3-cryptographic-deployment-hmac-sha256-with-zero-overhead)
   - [Phase 4: Forensic Leak Attribution](#phase-4-forensic-leak-attribution)
4. [Why CWS Beats Existing Research](#4-why-cws-beats-existing-research)
5. [Real Experimental Benchmarks (CIC-MalMem-2022)](#5-real-experimental-benchmarks-cic-malmem-2022)
6. [Visual Architecture & Experimental Diagrams](#6-visual-architecture--experimental-diagrams)
7. [Repository Setup & Step-by-Step Reproduction](#7-repository-setup--step-by-step-reproduction)
8. [Citation](#8-citation)

---

## 1. The Problem: Why Standard AI Antivirus Deployment Fails

In modern cybersecurity, antivirus and Endpoint Detection and Response (EDR) vendors train a single deep neural network on their cloud servers and ship the exact same weight file to millions of user computers. While convenient, this uniform deployment creates **two critical security failures**:

1. **Algorithmic Monoculture (One Exploit Breaks Every Computer):**  
   Because every user runs identical neural network weights, every user's model shares the exact same mathematical decision boundaries and gradient weak points. If an attacker obtains one copy of the antivirus software and crafts an adversarial malware mutation (e.g., via byte padding or feature perturbation) that tricks that single computer, **that exact same mutation bypasses every other computer in the world**.
2. **Unattributable Model Theft & Leakage:**  
   Training proprietary malware detectors requires massive telemetry pipelines and compute budgets. However, because every subscriber receives an identical `.pt` or `.onnx` weight binary, if a malicious insider or compromised client leaks the model file online, **there is no cryptographic way to prove which user account leaked it**.

Existing defenses only solve one half of this problem—and usually at a heavy performance cost:
- **Traditional Weight Watermarking** embeds a hidden signature into the weights, but every user still receives identical weights (zero protection against bypass mutations).
- **Ensemble / Moving Target Defenses (MTD)** deploy multiple different models per computer so attacks do not transfer, but forcing client machines to store and run multiple neural networks multiplies memory usage and slows down scanning.

---

## 2. Our Solution: Combinatorial Weight Superposition (CWS)

**Combinatorial Weight Superposition (CWS)** eliminates both vulnerabilities simultaneously while adding **0.00% memory or runtime overhead** on the user's computer.

Instead of learning one fixed weight per neural connection, the training server maintains a **Master Superposition Table** where every parameter slot stores $K = 4$ distinct candidate weights separated by a strict minimum distance (`min_gap = 0.06`). During training, the server randomly collapses the table into different sub-networks on every batch and trains them so that:
- **Any random selection of weights produces the exact same clean prediction** ($99.91\%$ accuracy, $99.92\%$ cross-user agreement).
- **Different weight selections rely on orthogonal input features**, preventing a malware mutation crafted for User #1 from bypassing Users #2–#21 ($99.63\%$ cross-user protection).
- **Each deployed user model is a completely normal single-weight neural network** ($108,160$ scalar parameters)—meaning the user's computer sees zero extra layers, zero extra RAM usage, and zero slowdown.

For a network with $N = 108,160$ parameter slots and $K = 4$ candidates per slot, a single training run creates a combinatorial keyspace of:

$$4^{108,160} \approx 10^{65,118} \text{ unique, valid neural networks}$$

---

## 3. Deep Dive: How CWS Works Step-by-Step

### Phase 1: Server-Side Superposition Training & `min_gap` Constraint

In a standard neural network with $N = 108,160$ parameters, each connection holds a single scalar value $w_i$. In CWS, each parameter slot $i \in \{1, \dots, N\}$ holds a sorted bucket of $K = 4$ candidate weights:

$$\mathbf{W}_i = [w_{i,0},\; w_{i,1},\; w_{i,2},\; w_{i,3}]$$

If we trained all four candidates simultaneously by averaging them, they would collapse into four identical numbers—destroying both cryptographic uniqueness and adversarial diversity. Instead, on every training step:
1. The server samples **two independent random masks** $\mathbf{m}^{(1)}, \mathbf{m}^{(2)} \in \{0, 1, 2, 3\}^N$, picking one weight per slot to form **Model 1** and **Model 2**.
2. Both models classify the same batch of files, and backpropagation updates **only the weights picked by Mask 1 and Mask 2**.
3. **Minimum-Gap Projection (`min_gap`):** Immediately after each optimizer step, the server sorts each 4-weight bucket and enforces a hard minimum separation $\delta = 0.06$ between adjacent candidates:

$$w_{i,k} \leftarrow \max\left(w_{i,k},\; w_{i,k-1} + \delta\right), \quad \forall k \in \{1, 2, 3\}$$

This guarantees that the four weights in every slot never overlap or merge during training, preserving clean forensic separability.

---

### Phase 2: Input-Gradient Decorrelation & Anti-Transfer Defense

To ensure that an evasion attack crafted against one user fails against other users, CWS optimizes two security objectives alongside standard classification loss:

1. **Prediction Consistency Loss ($\mathcal{L}_{\text{cons}}$):** Forces Model 1 and Model 2 to output identical class probabilities on clean samples so every user enjoys identical detection accuracy.
2. **Input-Gradient Decorrelation ($\mathcal{L}_{\text{decorr}}$):** Computes the input sensitivity gradients $\mathbf{g}^{(1)} = \nabla_{\mathbf{x}} \mathcal{L}_{\text{CE}}^{(1)}$ and $\mathbf{g}^{(2)} = \nabla_{\mathbf{x}} \mathcal{L}_{\text{CE}}^{(2)}$, and penalizes their squared cosine similarity:

$$\mathcal{L}_{\text{decorr}} = \left( \frac{\langle \mathbf{g}^{(1)}, \mathbf{g}^{(2)} \rangle}{\|\mathbf{g}^{(1)}\|_2 \|\mathbf{g}^{(2)}\|_2 + \epsilon} \right)^2$$

This forces different weight masks to rely on different combinations of the $512$ input features.
3. **Cross-Mask Anti-Transfer Training ($\mathcal{L}_{\text{anti}}$):** Generates an adversarial perturbation targeting Model 1 ($\tilde{\mathbf{x}}^{(1)} = \mathbf{x} + \eta\,\text{sign}(\mathbf{g}^{(1)})$) and explicitly trains Model 2 to still classify $\tilde{\mathbf{x}}^{(1)}$ correctly as malware.

---

### Phase 3: Cryptographic Deployment (`HMAC-SHA256`) with Zero Overhead

When provisioning software for a specific subscriber $\text{ID}_u$ (e.g., `user_001`), the vendor uses a private server key $K_{\text{sec}}$ and counter-mode `HMAC-SHA256` to generate a deterministic $216,320$-bit stream ($2$ bits per parameter slot):

$$\mathbf{B}_u = \text{HMAC-SHA256}(K_{\text{sec}}, \text{ID}_u \parallel 0) \parallel \text{HMAC-SHA256}(K_{\text{sec}}, \text{ID}_u \parallel 1) \parallel \dots$$

- Each 2-bit pair (`00`, `01`, `10`, `11`) selects index `0`, `1`, `2`, or `3` from the corresponding superposition slot.
- The selected scalar weights are exported into a standard `512 -> 160 -> 160 -> 2` PyTorch model file (`108,160` parameters).
- The user receives a completely normal weight file with **no extra code, no extra metadata, and 0.00% runtime slowdown**.

---

### Phase 4: Forensic Leak Attribution

If a leaked model file $\hat{\theta}$ appears on an unauthorized forum or malware testing service, the vendor compares each weight $\hat{\theta}_i$ against the four stored candidates in slot $i$ of the Master Superposition Table:

$$\hat{m}_i = \arg\min_{k \in \{0, 1, 2, 3\}} \left| \hat{\theta}_i - w_{i,k} \right|$$

Because every candidate in a slot is separated by at least `min_gap = 0.06`, nearest-neighbor matching recovers **100.0000% of the 216,320-bit cryptographic signature**, uniquely identifying the exact subscriber account that leaked the model.

---

## 4. Why CWS Beats Existing Research

| Capability / Requirement | Standard Neural Net | Weight Watermarking *(Uchida et al. / DeepMarks)* | Backdoor Watermarking *(Adi et al.)* | Ensemble / MTD *(Kariyappa / Morphence)* | **Proposed CWS ($K=4$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Traces Leaked Raw Weight File?** | No ($0$ bits) | Global or low-capacity | No (requires API query) | No ($0$ bits) | **Yes ($216,320$ bits, $100\%$)** |
| **Stops Single Exploit Across Users?** | No ($16.50\%$ catch) | No (identical boundaries) | No (identical boundaries) | Yes | **Yes ($99.63\%$ catch)** |
| **Client Memory & Latency Overhead** | $0.00\%$ ($1\times$ size) | $\approx 0.00\%$ | $\approx 0.00\%$ | **High** ($M\times$ models) | **$0.00\%$ ($1\times$ scalar file)** |
| **Unique Models per Training Run** | $1$ static model | $1$ static model | $1$ static model | Small pool ($5\text{--}20$) | **$4^{108,160}$ unique models** |

---

## 5. Real Experimental Benchmarks (CIC-MalMem-2022)

We evaluated CWS on the **CIC-MalMem-2022** dataset ($58,596$ balanced benign and obfuscated malware samples across Trojan, Ransomware, and Spyware families; $46,876$ training / $11,720$ test records) using a `512 -> 160 -> 160 -> 2` residual classifier (`RMSNorm` + `GELU` + residual skip connection) with **$108,160$ scalar parameters**:

| Evaluation Benchmark | Standard Baseline (Monoculture) | Proposed CWS ($K=4$, `min_gap=0.06`) | What This Proves |
| :--- | :---: | :---: | :--- |
| **Clean Test Accuracy** ($50$ Random Collapses) | `99.99%` | **`99.91%`** *(Min: `99.86%`, Max: `99.95%`)* | Virtually zero accuracy loss across $4^{108,160}$ combinations |
| **Cross-User Prediction Agreement** | `100.00%` *(trivial)* | **`99.92%`** | Different users' weight files make consistent decisions |
| **Client Memory & Speed Overhead** | `0.00%` | **`0.00%`** ($108,160$ scalar parameters) | Zero extra inference latency or RAM usage on endpoints |
| **Cryptographic Signature Recovery** | `0.00%` *(none)* | **`100.0000%`** ($216,320 / 216,320$ bits) | Exact bit-for-bit forensic leak attribution from `.pt` files |
| **Targeted Endpoint Catch Rate** (User #1, PGD) | `16.50%` | **`87.00%`** ($26/200$ mutations bypassed) | White-box PGD evasion crafted specifically against User #1 |
| **All Other Endpoints Catch Rate** (Users #2–#21) | `16.50%` | **`99.63%`** | **$>6\times$ higher** protection across non-targeted users |

---

## 6. Visual Architecture & Experimental Diagrams

### 1. Experimental Benchmark Results on CIC-MalMem-2022
![Real Experimental Benchmark on CIC-MalMem-2022](assets/3_3.png)

### 2. Three-Phase Pipeline: Server Training, User Deployment & Leak Verification
![Three-Phase System Flow](assets/0_2.jpg)

### 3. Inside a Superposition Slot ($K=4$) & Minimum-Gap Rule
![Combinatorial Weight Superposition](assets/2_2.jpg)

### 4. Deployed Collapsed Model Architecture ($108,160$ Parameters)
![Deployed Model Architecture](assets/3_2.jpg)

### 5. Input-Gradient Decorrelation During Training
![Gradient Decorrelation](assets/5_2.jpg)

### 6. Cryptographic `HMAC-SHA256` Bit Mapping & Keyspace Scaling
![Cryptographic Weight Selection](assets/4_2.jpg)

### 7. Anti-Monoculture Defense: Standard vs. Superposition Antivirus
![Antivirus Comparison](assets/1_2.jpg)

---

## 7. Repository Setup & Step-by-Step Reproduction

### Folder Structure
Create an `assets/` folder in your repository and place the six diagrams and benchmark screenshot inside it so all images display automatically:

```text
├── assets/
│   ├── 0_2.jpg          # Three-phase server training, deployment & leak verification flow
│   ├── 1_2.jpg          # Standard vs. Superposition antivirus defense comparison
│   ├── 2_2.jpg          # K=4 superposition bucket & min_gap illustration
│   ├── 3_2.jpg          # Deployed 512 -> 160 -> 160 -> 2 network architecture
│   ├── 4_2.jpg          # HMAC-SHA256 bit selection & keyspace scaling
│   ├── 5_2.jpg          # Input-gradient decorrelation diagram
│   └── 3_3.png          # CIC-MalMem-2022 experimental benchmark screenshot & chart
├── cws_experiment.ipynb # Complete PyTorch / Google Colab notebook
└── README.md            # Project documentation
```

### Installation & Running the Notebook
1. **Install dependencies:**
   ```bash
   pip install torch scikit-learn pandas matplotlib numpy
   ```
2. **Run `cws_experiment.ipynb` (in Google Colab or locally):**
   - **Cell 1 (Dataset Preparation):** Downloads `Obfuscated-MalMem2022.csv` ($58,596$ samples), standardizes all features, projects them into $512$ static feature channels, and builds a stratified $80\% / 20\%$ train/test split.
   - **Cell 2 (Training):** Trains both the parameter-matched Standard Baseline (`108,160` weights) and the Master CWS model (`108,160` slots $\times 4 = 432,640$ master weights, `min_gap = 0.06`) with dual-mask consistency, orthogonal gradient loss, and cross-mask anti-transfer training.
   - **Cell 3 (Verification & Attack Suite):** Evaluates $50$ random weight collapses, exports User #1's model via `HMAC-SHA256` to `leaked_user_001.pt` to verify $100\%$ recovery of all $216,320$ bits, and runs a $20$-step white-box PGD evasion attack ($\epsilon = 0.18$) against User #1 to measure cross-user protection across Users #2–#21.

---

## 8. Citation

If you use this codebase, diagrams, or methodology in your research, please cite:

```bibtex
@article{parihar2026cws,
  title   = {Combinatorial Weight Superposition: Zero-Overhead Cryptographic Leak Attribution and Anti-Monoculture Defense in Neural Malware Classifiers},
  author  = {Parihar, Manish Kumar},
  year    = {2026},
  note    = {ORCID: 0009-0002-1900-8945}
}
```