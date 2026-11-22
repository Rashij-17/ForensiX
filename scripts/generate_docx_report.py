import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page Margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("ForensiX: Multimodal Digital Media Forensics & Authenticity Intelligence")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(16)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Comprehensive Technical Documentation, System Workflow & Team Contributions\n"
                            "Fundamentals of Computing for Data Science Lab (26B16CS318) — B.Tech V Semester\n"
                            "Jaypee Institute of Information Technology (JIIT), Noida")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x44, 0x55, 0x66)

    # Horizontal Rule
    p_hr = doc.add_paragraph()
    p_hr.paragraph_format.space_after = Pt(14)
    run_hr = p_hr.add_run("―" * 55)
    run_hr.font.color.rgb = RGBColor(0x99, 0xAA, 0xBB)
    p_hr.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Project metadata block
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row_0 = meta_table.rows[0].cells
    row_0[0].text = "Project Team Members:"
    row_0[1].text = "Supervisors / Faculty Mentors:"
    row_1 = meta_table.rows[1].cells
    row_1[0].text = "• Rashi Johari (2401030072)\n• Akshat Singh (2401030075)\n• Sriza Goel (2401030094)"
    row_1[1].text = "• Dr. Amit Mishra\n• Dr. Megh Singhal\n• Dr. Somya Jain"
    
    for row in meta_table.rows:
        for cell in row.cells:
            set_cell_background(cell, "F4F6F9")
            set_cell_margins(cell, 80, 80, 120, 120)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(3)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # SECTION 1: PROJECT OVERVIEW
    h1 = doc.add_heading("1. Project Executive Overview", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "ForensiX is an end-to-end multimodal forensic intelligence platform engineered to detect manipulated media "
        "(deepfakes, synthetic media, tampered files) across three primary digital modalities: Images, Audio, and Video. "
        "Unlike commercial or naive deepfake detectors that output an opaque, binary 'Real' or 'Fake' label with no verifiable "
        "justification, ForensiX adheres strictly to the principles of forensic science and trustworthy machine learning:"
    )

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Multimodal by Design: ")
    r.bold = True
    p.add_run("Evaluates image artifacts (spatial, frequency, noise), audio anomalies (spectral, MFCC, pitch jitter), and video inconsistencies (temporal jerks, optical-flow motion coherence, warp error).")

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Evidence-Backed & Transparent: ")
    r.bold = True
    p.add_run("Every single detection is justified using interpretable forensic features and explained via SHAP (SHapley Additive exPlanations), proving which specific biological or digital artifacts drove the verdict.")

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Statistically Validated: ")
    r.bold = True
    p.add_run("Features are verified with rigorous hypothesis tests (Welch t-test, Mann-Whitney U, Kolmogorov-Smirnov) and Benjamini-Hochberg False Discovery Rate (FDR) control to guarantee that differences are statistically significant.")

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Realistic Robustness Testing: ")
    r.bold = True
    p.add_run("Models are evaluated under severe real-world social-media degradations (JPEG compression, Gaussian noise, blur, downsampling, MP3 re-encoding, and H.264 video compression).")

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Calibrated Verdicts: ")
    r.bold = True
    p.add_run("Reports reject definitive absolutes. Results are categorised into calibrated confidence bands: '<35% Authentic', '35–65% Inconclusive', and '>65% Potentially Manipulated', protecting legal integrity.")

    p = doc.add_paragraph(style='List Bullet')
    r = p.add_run("Production MLOps Lifecycle: ")
    r.bold = True
    p.add_run("Integrated with MLflow experiment tracking, automated model registry with stage gates, Population Stability Index (PSI) drift monitoring, and an automated retraining pipeline.")

    # SECTION 2: SYSTEM ARCHITECTURE
    h2 = doc.add_heading("2. System Architecture & Dual-Mode Execution", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "ForensiX is structured into a clean three-tier architecture with an intelligent dual-mode fallback designed "
        "for maximum operational resilience:"
    )

    doc.add_paragraph(
        "1. Presentation Layer (Streamlit Web Workspace :8501): An interactive forensic workstation featuring eight dedicated "
        "views: Analyze (instant file classification), Evidence (deep signal & frequency inspection), Compare (side-by-side delta "
        "visualisation), Forensic Lab (real-time interactive degradation sandbox), Investigations (case management & audit logs), "
        "Reports (formal forensic certificate generation), Benchmarks (model comparison & statistical rigor), and MLOps (model registry, drift, and retraining).\n"
        "2. API & Service Layer (FastAPI Backend :8000 & Shared Services): High-performance REST endpoints (/analyze, /explain, /report, "
        "/mlops) wrapped in a typed Python SDK. Crucially, if the external FastAPI server is offline, the Streamlit app automatically falls "
        "back to running the identical service layer in-process without losing functionality.\n"
        "3. Core Machine Learning & Data Pipeline: The underlying data ingestion, multi-stage preprocessing, forensic feature extraction, "
        "statistical validation, model training, SHAP explainers, and SQLite-backed case stores."
    )

    # SECTION 3: 12-STAGE PIPELINE WORKFLOW
    h3 = doc.add_heading("3. The End-to-End 12-Stage Forensic Workflow", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "The system executes a strictly sequential, leak-free pipeline where each stage has a distinct forensic purpose and CLI command:"
    )

    stages = [
        ("Stage 1: Data Acquisition & Construction", "Scans raw files, standardises metadata, auto-infers labels and generator sources from paths, and computes cryptographic MD5 hashes and perceptual difference hashes (dHash). Enforces leak-free 70/15/15 stratified train/val/test splits.", "forensix/data/dataset.py", "make ingest"),
        ("Stage 2: Multimodal Preprocessing", "Rejects corrupted, undecodable, or silent samples. Resizes images to 256x256 float [0, 1]; resamples audio to 16 kHz mono, peak-normalises to 0.95 and segments into 4s chunks; extracts video frames at fixed intervals (128x128).", "forensix/preprocess/", "make ingest"),
        ("Stage 3: Exploratory Data Analysis (EDA)", "Conducts class-conditional data analysis strictly on the training split: generates intensity histograms, top-feature box plots, Spearman correlation matrices, 2D FFT radial spectra, audio spectrograms/MFCCs, and temporal difference charts.", "forensix/eda.py", "make eda"),
        ("Stage 4: Forensic Feature Engineering", "Extracts 34 image features (spatial, edge, noise residual, 2D FFT, DCT blockiness), 47 audio features (MFCCs, spectral roll-off/flux, pitch jitter, dynamic range), and 33 video features (inter-frame diffs, optical flow, warp error).", "forensix/features/", "make features"),
        ("Stage 5: Statistical Validation", "Runs Welch t-tests, Mann-Whitney U, two-sample Kolmogorov-Smirnov tests, Cohen's d effect sizes, and Benjamini-Hochberg FDR correction. Features with q < 0.05 are validated as genuine markers of manipulation.", "forensix/stats/", "make stats"),
        ("Stage 6: Multi-Model Training", "Trains four distinct candidate models per modality: Logistic Regression, Support Vector Machine (RBF kernel), Random Forest, and XGBoost. Linear models receive StandardScaler. Winner chosen strictly on validation split F1-score.", "forensix/models/train.py", "make train"),
        ("Stage 7: Model Explainability (SHAP)", "Computes exact Shapley attribution values using TreeExplainer (RF/XGBoost) and probability-space PermutationExplainer (LogReg/SVM). Every prediction yields a breakdown of which features pushed the verdict toward manipulated or authentic.", "forensix/explain/", "make explain"),
        ("Stage 8: Comprehensive Model Evaluation", "Computes Accuracy, Precision, Recall, F1-score, ROC AUC, Average Precision, Confusion Matrices, ROC curves, and Precision-Recall curves exclusively on the unseen test split.", "forensix/models/train.py", "make train"),
        ("Stage 9: Robustness & Degradation Stress-Testing", "Evaluates production models under 8 realistic degradations per modality (e.g. JPEG quality 70/30, audio noise 20/10 dB SNR, H.264 CRF 28/38). Compares clean vs. degraded F1 and counts failed decodes without silent drops.", "forensix/robustness/", "make robustness"),
        ("Stage 10: Source & Generator-Wise Analysis", "Calculates manipulation recall for each known synthetic generator (e.g., Stable Diffusion, Midjourney, TTS engines, deepfake encoders) with Wilson 95% confidence intervals, identifying blind spots across different generative engines.", "forensix/models/generators.py", "make generators"),
        ("Stage 11: Deployment & Interactive Workstation", "Deploys the dual-tier FastAPI microservice and Streamlit workspace, exposing interactive investigation tools, file analysis, forensic lab tampering, and PDF/HTML report generators.", "api/, frontend/", "make start"),
        ("Stage 12: Continuous MLOps & Retraining Gate", "Maintains versioned dataset metadata, logs model parameters and metrics to MLflow, manages model promotion (staging -> production), monitors feature drift via PSI, and gates automated retraining on untouched test data.", "forensix/mlops/", "make monitor")
    ]

    table_stages = doc.add_table(rows=len(stages) + 1, cols=4)
    table_stages.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table_stages.rows[0].cells
    hdr[0].text = "Stage"
    hdr[1].text = "Key Forensic Activities"
    hdr[2].text = "Core Module"
    hdr[3].text = "Command"
    
    for c in hdr:
        set_cell_background(c, "1B365D")
        set_cell_margins(c, 100, 100, 120, 120)
        for p in c.paragraphs:
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for idx, (stg, act, mod, cmd) in enumerate(stages):
        row = table_stages.rows[idx + 1].cells
        row[0].text = stg
        row[1].text = act
        row[2].text = mod
        row[3].text = cmd
        bg = "F9FAFB" if idx % 2 == 0 else "FFFFFF"
        for c in row:
            set_cell_background(c, bg)
            set_cell_margins(c, 80, 80, 100, 100)
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(2)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # SECTION 4: 4.12 CONTRIBUTION OF TEAM MEMBERS
    h4 = doc.add_heading("4. Section 4.12: Contribution of Team Members", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Each member wrote and tested code in a separate part of the system and took part in integration, testing, "
        "and documentation. The team followed a strict separation of concerns, ensuring high modularity, automated test "
        "coverage, and seamless interoperability between components."
    )

    doc.add_paragraph("Table 4.5: Code contributions").runs[0].font.bold = True

    table_45 = doc.add_table(rows=4, cols=3)
    table_45.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_hdr = table_45.rows[0].cells
    t_hdr[0].text = "Member"
    t_hdr[1].text = "Modules implemented"
    t_hdr[2].text = "Main files"

    for c in t_hdr:
        set_cell_background(c, "1B365D")
        set_cell_margins(c, 100, 100, 120, 120)
        for p in c.paragraphs:
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    contrib_data = [
        ("Rashi Johari\n(2401030072)",
         "Dataset construction, duplicate detection and splits; image, audio and video preprocessing; feature extraction and feature catalogue; exploratory data analysis",
         "data/, preprocess/, features/, eda.py"),
        ("Akshat Singh\n(2401030075)",
         "Shared service layer, inference engine and verdict bands, FastAPI backend and client library, Streamlit application, MLOps (tracking, registry, versioning, monitoring, retraining), report builder, case store, tests and CI",
         "services.py, inference.py, api/, frontend/, sdk/, mlops/, report.py, cases.py, tests/"),
        ("Sriza Goel\n(2401030094)",
         "Statistical validation, model training and evaluation, SHAP explanations, robustness transformations and evaluation, generator-wise analysis, forensic lab",
         "stats/, models/, explain/, robustness/, lab.py")
    ]

    for idx, (m, mod, f) in enumerate(contrib_data):
        row = table_45.rows[idx + 1].cells
        row[0].text = m
        row[1].text = mod
        row[2].text = f
        bg = "F9FAFB" if idx % 2 == 0 else "FFFFFF"
        for c in row:
            set_cell_background(c, bg)
            set_cell_margins(c, 100, 100, 120, 120)
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(2)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Detailed summary of Akshat and Sriza
    p_ak = doc.add_paragraph()
    r = p_ak.add_run("Akshat Singh (2401030075) — Systems, Full-Stack & MLOps Architecture: ")
    r.bold = True
    p_ak.add_run(
        "Architected the overarching dual-mode service layer (services.py) and unified inference engine (inference.py). "
        "Engineered the asynchronous FastAPI REST API (api/) with interactive Swagger docs and designed the client SDK. "
        "Built the full 8-view Streamlit forensic suite (frontend/), SQLite investigation case store (cases.py), forensic "
        "report generator (report.py), automated continuous integration suite (tests/), and the entire MLOps lifecycle "
        "(MLflow tracking, model registry, PSI drift monitoring, and automated retraining gates)."
    )

    p_sr = doc.add_paragraph()
    r = p_sr.add_run("Sriza Goel (2401030094) — Modeling, Statistical Rigor & Explainability: ")
    r.bold = True
    p_sr.add_run(
        "Implemented the statistical hypothesis testing framework (stats/) executing Welch t, Mann-Whitney U, KS tests, and FDR correction. "
        "Trained, cross-validated, and evaluated 4 machine learning algorithms per modality (models/). Built the SHAP explainability engine "
        "(explain/) providing local and global feature attribution. Programmed the real-world robustness perturbation suite (robustness/), "
        "conducted generator-wise consistency analysis, and implemented the Forensic Lab tamper testing sandbox (lab.py)."
    )

    # SECTION 5: RASHI'S CONTRIBUTION IN EASY WORDS
    h5 = doc.add_heading("5. Detailed Breakdown: Rashi Johari's Contribution in Plain English", level=1)
    h5.paragraph_format.space_before = Pt(16)
    h5.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "In plain and simple words, Rashi Johari served as the Data & Forensic Feature Foundation Engineer of ForensiX. "
        "Machine learning models cannot work on raw, messy files. Before any model can learn or detect a deepfake, someone has to "
        "collect the media, clean it, ensure there is zero cheating/leakage, standardize the formats, extract mathematical forensic "
        "measurements (features) from the signals, and visualize how real files differ from fake ones. "
        "That entire foundational half of the project was built, tested, and owned by Rashi."
    )

    # Submodule 1
    h5_1 = doc.add_heading("Module 1: Dataset Construction & Leak-Free Splitting (forensix/data/)", level=2)
    h5_1.paragraph_format.space_before = Pt(10)
    h5_1.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "What Rashi did here:\n"
        "• Raw Directory Scanning & Auto-Inference (dataset.py): Wrote code to recursively scan raw media folders. Instead of requiring manual labeling, her script inspects folder names to automatically detect if a sample is 'real' or 'fake', and which AI generator created it (e.g., Stable Diffusion, Midjourney, TTS).\n"
        "• Strict Duplicate Detection (MD5 & Perceptual dHash): In data science, having the same image in both training and testing is called 'data leakage' (it makes models artificially look like they have 100% accuracy). Rashi implemented exact MD5 hash checking for all files, PLUS a 64-bit difference hash (dHash) for images to catch near-identical cropped/resized images. Duplicate copies are marked as 'status=duplicate' and automatically isolated.\n"
        "• Leak-Free Stratified Splits: Created a 70% Train, 15% Validation, and 15% Test split stratified across real/fake classes. Her unit test (test_splits_are_disjoint_and_cover_all_usable_samples) mathematically proves that zero duplicate files ever straddle the train and test sets.\n"
        "• Unified Metadata Registry: Standardized every file's dimensions, duration, fps, hash, and status into a single comprehensive metadata table.\n"
        "• Synthetic Data Generator (synthetic.py): Wrote the demo generator that allows anyone to clone the repo and run the full end-to-end pipeline in 60 seconds without needing to download massive external gigabyte datasets."
    )

    # Submodule 2
    h5_2 = doc.add_heading("Module 2: Multimodal Preprocessing (forensix/preprocess/)", level=2)
    h5_2.paragraph_format.space_before = Pt(10)
    h5_2.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "What Rashi did here:\n"
        "Different media types have different formats and shapes. Rashi wrote the cleaning pipelines for all three modalities:\n"
        "• Image Preprocessing (image.py): Catches corrupted/unreadable images. Resizes valid images to a uniform 256x256 resolution using area/cubic interpolation, converts color space to standard RGB, and normalizes pixel values from [0, 255] integers to [0.0, 1.0] float32 arrays.\n"
        "• Audio Preprocessing (audio.py): Validates sound files, rejecting corrupt, undecodable, or silent audio (under -55 dBFS). Resamples all audio to 16,000 Hz mono, strips DC offset, peak-normalizes volume to 0.95 (so loudness doesn't trick the model), and cuts long audio into uniform 4-second chunks.\n"
        "• Video Preprocessing (video.py): Probes video streams using OpenCV, verifies valid fps and frame counts, samples frames at fixed intervals (every 2nd frame up to 24 frames), resizes them to 128x128, and formats consecutive frame pairs to analyze motion jumps."
    )

    # Submodule 3
    h5_3 = doc.add_heading("Module 3: Forensic Feature Extraction & Catalog (forensix/features/)", level=2)
    h5_3.paragraph_format.space_before = Pt(10)
    h5_3.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "What Rashi did here:\n"
        "This is the scientific heart of Rashi's work. Instead of treating media as random pixels, Rashi extracted 114 forensic features based on how AI generators leave mathematical fingerprints:\n"
        "• 34 Image Features (image.py): Gray-level variance, skewness, kurtosis, Shannon entropy, RGB color correlations; edge density and Laplacian sharpness; high-pass noise residuals and 1-pixel/2-pixel autocorrelation (which reveal AI upsampling artifacts); 2D Fast Fourier Transform (FFT) radial power slopes and periodic high-frequency peaks; and Discrete Cosine Transform (DCT) 8x8 blockiness (compression traces).\n"
        "• 47 Audio Features (audio.py): 13 Mel-Frequency Cepstral Coefficients (MFCC means and standard deviations) capturing vocal tract acoustics; spectral centroid, bandwidth, flatness, contrast, and flux; zero-crossing rate; pitch mean, variance, jitter (unnatural pitch smoothness reveals synthetic voices); and high-frequency energy ratios (above 4 kHz and 7 kHz).\n"
        "• 33 Video Features (video.py): Inter-frame temporal brightness and texture changes; sudden frame differences and jerk; temporal noise correlation across frames; optical-flow motion vectors and motion coherence; and motion-compensated warp prediction error (revealing face warping inconsistencies).\n"
        "• Forensic Catalog (catalog.py): Mapped every single feature to an indicator group (e.g., 'Frequency', 'Noise', 'Pitch', 'Motion') and a plain-English human description. This catalog is what Sriza and Akshat's SHAP reports use to explain verdicts to non-technical users."
    )

    # Submodule 4
    h5_4 = doc.add_heading("Module 4: Exploratory Data Analysis (forensix/eda.py)", level=2)
    h5_4.paragraph_format.space_before = Pt(10)
    h5_4.paragraph_format.space_after = Pt(4)

    doc.add_paragraph(
        "What Rashi did here:\n"
        "• Class-Conditional Visualizations: Automated the generation of comparative plots strictly on the training set (to avoid test data snooping).\n"
        "• Generated Histograms & Box Plots showing the exact distributions of real vs. manipulated files for top features.\n"
        "• Spearman Feature Correlation Matrices: Generated correlation heatmaps to discover redundant feature pairs.\n"
        "• Signal Evidence Figures: Generated mean 2D FFT spectrum comparisons for images, audio spectrograms & MFCC heatmaps, and video temporal frame differences.\n"
        "• Automated Reporting: Produced summary.json files detailing class balance, sample counts, and top separating features."
    )

    # SECTION 6: VIVA & PRESENTATION GUIDE FOR RASHI
    h6 = doc.add_heading("6. Viva & Presentation Guide: What Rashi Has to Say & Do", level=1)
    h6.paragraph_format.space_before = Pt(16)
    h6.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "When explaining your contribution during the lab evaluation or viva, follow this structured narrative:"
    )

    doc.add_paragraph(
        "Your 1-Minute Elevator Pitch:\n"
        "\"In ForensiX, I was responsible for the entire Data & Forensic Feature Foundation. My job was to take raw, messy multimodal media "
        "(images, audio, video), clean and standardize them, eliminate data leakage through cryptographic and perceptual deduplication, "
        "extract 114 domain-specific forensic features (frequency spectra, noise residuals, MFCCs, pitch jitter, optical flow), catalog "
        "their plain-English meanings for SHAP explainability, and conduct exploratory data analysis on the training set. "
        "My output is the clean, leak-free feature table that Sriza used to train the models and Akshat deployed to production.\""
    ).runs[0].font.italic = True

    doc.add_paragraph("Top 5 Viva Questions You Will Be Asked & How to Answer:").runs[0].font.bold = True

    qa = [
        ("Q1: How did you ensure there is no data leakage between your train and test sets?",
         "Answer: I implemented a two-tier deduplication check in forensix/data/dataset.py. First, exact cryptographic MD5 hashing to catch identical files. Second, a 64-bit difference hash (dHash) with a Hamming distance threshold of <= 2 to catch near-duplicates (like resized or cropped versions). Groups of duplicates are assigned a single duplicate group, and only one copy is kept. Our splits (70/15/15) are strictly stratified, and our automated unit tests assert that no duplicate group or identical MD5 straddles across splits."),
        
        ("Q2: Why did you extract handcrafted forensic features instead of using an end-to-end Deep Neural Network?",
         "Answer: Deep neural networks act as black boxes—they might classify an image based on background noise or specific camera models rather than actual manipulation, and they fail when compressed. By extracting explicit forensic features—such as 2D FFT high-frequency energy, noise residual autocorrelation, MFCCs, pitch jitter, and optical-flow warp error—we know exactly what physical and mathematical artifacts are being measured. This enables hypothesis testing, SHAP explainability, and calibrated reporting."),

        ("Q3: What preprocessing steps did you perform on audio and why?",
         "Answer: In forensix/preprocess/audio.py, I first rejected corrupted or silent recordings (below -55 dBFS). Then I resampled all audio to 16 kHz mono. Crucially, I removed DC offset and peak-normalized amplitude to 0.95 so models wouldn't be biased by differences in recording volume. Finally, I segmented long audio into uniform 4-second chunks so that feature extraction is consistent."),

        ("Q4: Can you explain one image feature and one audio feature you extracted?",
         "Answer for Image: 'noise_autocorr_h1' computes the 1-pixel horizontal autocorrelation of the noise residual after high-pass filtering. Generative AI models (like GANs or Diffusion models) perform upsampling/transposed convolutions that leave regular periodic autocorrelation traces in the noise pattern that natural camera sensors do not produce.\n"
         "Answer for Audio: 'pitch_jitter' measures the frame-to-frame micro-variations in vocal cord vibration frequency. Natural human speech has subtle, organic pitch jitter. Synthetic voices generated by TTS engines often have unnaturally smooth or mathematically robotic pitch contours."),

        ("Q5: Where in the codebase is your code tested?",
         "Answer: My modules are covered by automated pytest suites in tests/test_preprocess_features.py and tests/test_dataset_stats.py. These tests verify corrupt file rejection, resolution normalisation, deterministic feature extraction, absence of NaN/infinite values, and mathematical verification that train/test splits are strictly disjoint.")
    ]

    for q, a in qa:
        p_q = doc.add_paragraph()
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        
        p_a = doc.add_paragraph(a)
        p_a.paragraph_format.space_after = Pt(8)

    output_path = "ForensiX_Project_Report_and_Workflow.docx"
    doc.save(output_path)
    print(f"Successfully generated Word Document at: {output_path}")

if __name__ == "__main__":
    create_document()
