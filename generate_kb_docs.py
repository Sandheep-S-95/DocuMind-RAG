"""
generate_kb_docs.py
--------------------
Creates 5 medical-student-level study PDFs in ./knowledge_base_dir so you
have a ready-made knowledge base to demo the RAG app with.

Run: python generate_kb_docs.py
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base_dir")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("DocTitle", parent=styles["Title"], spaceAfter=18)
h_style = ParagraphStyle("H", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)


def build_pdf(filename: str, title: str, sections: list[tuple[str, list[str]]]):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [Paragraph(title, title_style), Spacer(1, 6)]
    for heading, paragraphs in sections:
        story.append(Paragraph(heading, h_style))
        for para in paragraphs:
            story.append(Paragraph(para, body_style))
    doc.build(story)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# 1. Cardiovascular Physiology
# ---------------------------------------------------------------------------
build_pdf(
    "Cardiovascular_Physiology.pdf",
    "Cardiovascular Physiology",
    [
        ("Cardiac Anatomy Overview", [
            "The heart is a four-chambered muscular organ consisting of two atria and two ventricles. "
            "The right atrium receives deoxygenated blood from the superior and inferior vena cavae, "
            "passing it through the tricuspid valve into the right ventricle, which pumps it through the "
            "pulmonary valve into the pulmonary arteries for gas exchange in the lungs. Oxygenated blood "
            "returns via the pulmonary veins to the left atrium, crosses the bicuspid (mitral) valve into "
            "the left ventricle, and is ejected through the aortic valve into the systemic circulation. "
            "The left ventricle has a thicker myocardial wall than the right ventricle because it must "
            "generate enough pressure to perfuse the entire systemic circuit, whereas the right ventricle "
            "only needs to drive blood through the lower-resistance pulmonary circuit.",
        ]),
        ("The Cardiac Conduction System", [
            "The sinoatrial (SA) node, located in the wall of the right atrium near the entrance of the "
            "superior vena cava, is the heart's natural pacemaker, initiating electrical impulses at an "
            "intrinsic rate of 60-100 beats per minute. This impulse spreads across both atria, causing "
            "atrial contraction, and reaches the atrioventricular (AV) node near the interatrial septum. "
            "The AV node delays the impulse by roughly 0.1 seconds, allowing the atria to finish "
            "contracting and the ventricles to fill before ventricular contraction begins. From the AV "
            "node, the impulse travels down the Bundle of His, splits into the left and right bundle "
            "branches, and terminates in the Purkinje fibers, which rapidly distribute the signal "
            "throughout the ventricular myocardium, producing a coordinated, near-simultaneous contraction.",
            "If the SA node fails, the AV node can act as a backup pacemaker at an intrinsic rate of "
            "40-60 bpm; if both fail, the Purkinje fibers can sustain a much slower ventricular escape "
            "rhythm of 20-40 bpm. This hierarchy explains why AV nodal block produces bradycardia rather "
            "than immediate cardiac arrest.",
        ]),
        ("The Cardiac Cycle", [
            "The cardiac cycle is divided into systole (contraction) and diastole (relaxation). During "
            "ventricular diastole, the AV valves are open and the ventricles fill passively, followed by "
            "an active atrial 'kick' that contributes the final 20-30% of ventricular filling. Isovolumetric "
            "contraction begins when the ventricles start contracting with all valves closed, raising "
            "pressure until it exceeds arterial pressure and the semilunar valves open, beginning "
            "ventricular ejection. As the ventricles relax, pressure falls below arterial pressure, the "
            "semilunar valves close (producing the second heart sound, S2), and isovolumetric relaxation "
            "occurs before the AV valves reopen (often preceded by closure sounds forming S1) to begin the "
            "next cycle.",
            "Stroke volume (SV) is the volume of blood ejected per beat, calculated as end-diastolic "
            "volume (EDV) minus end-systolic volume (ESV). Cardiac output (CO) equals stroke volume "
            "multiplied by heart rate (CO = SV x HR), and is the primary determinant of tissue perfusion.",
        ]),
        ("Regulation of Blood Pressure", [
            "Mean arterial pressure (MAP) is approximately equal to cardiac output multiplied by total "
            "peripheral resistance (MAP = CO x TPR). Short-term blood pressure regulation is dominated by "
            "the baroreceptor reflex: stretch receptors in the carotid sinus and aortic arch detect changes "
            "in arterial pressure and relay this information via the glossopharyngeal and vagus nerves to "
            "the medullary cardiovascular center, which adjusts sympathetic and parasympathetic outflow to "
            "the heart and vasculature within seconds.",
            "Long-term blood pressure regulation relies heavily on the renin-angiotensin-aldosterone system "
            "(RAAS). A drop in renal perfusion pressure triggers renin release from juxtaglomerular cells, "
            "which converts angiotensinogen to angiotensin I. Angiotensin-converting enzyme (ACE), mainly in "
            "the pulmonary vasculature, converts angiotensin I to angiotensin II, a potent vasoconstrictor "
            "that also stimulates aldosterone release from the adrenal cortex, promoting sodium and water "
            "retention in the renal collecting ducts and thereby raising blood volume and pressure over "
            "hours to days.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 2. Respiratory System
# ---------------------------------------------------------------------------
build_pdf(
    "Respiratory_System.pdf",
    "The Respiratory System",
    [
        ("Airway Anatomy and Mechanics of Breathing", [
            "Air travels from the nasal cavity or mouth through the pharynx and larynx into the trachea, "
            "which branches into the right and left main bronchi, then progressively smaller bronchioles, "
            "terminating in alveoli where gas exchange occurs. The right main bronchus is wider, shorter, "
            "and more vertical than the left, making it the more common site for aspirated foreign bodies.",
            "Inspiration is an active process driven primarily by contraction of the diaphragm, which "
            "flattens and increases thoracic volume, along with the external intercostal muscles, which "
            "elevate the ribs. This increase in thoracic volume decreases intrapleural and intrapulmonary "
            "pressure below atmospheric pressure, drawing air into the lungs (Boyle's Law: pressure and "
            "volume are inversely related at constant temperature). Quiet expiration is passive, relying on "
            "elastic recoil of the lungs and chest wall as the diaphragm relaxes; forced expiration recruits "
            "the internal intercostals and abdominal muscles.",
        ]),
        ("Lung Volumes and Capacities", [
            "Tidal volume (TV) is the air moved in a single normal breath (~500 mL in adults). Inspiratory "
            "reserve volume (IRV) is the additional air that can be forcibly inhaled after a normal tidal "
            "inspiration, and expiratory reserve volume (ERV) is the additional air that can be forcibly "
            "exhaled after a normal tidal expiration. Residual volume (RV) is the air remaining in the "
            "lungs after maximal expiration and cannot be measured by spirometry alone because it is never "
            "exhaled.",
            "Functional residual capacity (FRC = ERV + RV) is the volume remaining in the lungs after a "
            "normal tidal expiration, and represents the resting balance point between the inward elastic "
            "recoil of the lung and the outward recoil of the chest wall. Vital capacity (VC = TV + IRV + "
            "ERV) is the maximum volume that can be voluntarily moved in a single breath, and total lung "
            "capacity (TLC = VC + RV) is the volume in the lungs after a maximal inspiration.",
        ]),
        ("Gas Exchange and the Oxygen-Hemoglobin Dissociation Curve", [
            "Gas exchange across the alveolar-capillary membrane occurs by simple diffusion, driven by "
            "partial pressure gradients, and is favored by the large surface area and thin membrane of the "
            "alveoli. Oxygen diffuses from alveolar air (PAO2 ~100 mmHg) into pulmonary capillary blood "
            "(PvO2 ~40 mmHg), while carbon dioxide diffuses in the opposite direction, from blood (~45 "
            "mmHg) into alveolar air (~40 mmHg).",
            "Most oxygen in blood is carried bound to hemoglobin, and the relationship between PaO2 and "
            "hemoglobin saturation is described by the sigmoid oxygen-hemoglobin dissociation curve. A "
            "rightward shift of this curve -- caused by increased temperature, increased PCO2, decreased "
            "pH (the Bohr effect), or increased 2,3-BPG -- decreases hemoglobin's affinity for oxygen, "
            "promoting oxygen unloading at the tissues, which is physiologically useful during exercise or "
            "in metabolically active tissue. A leftward shift has the opposite effect, increasing affinity "
            "and impairing oxygen delivery to tissues.",
        ]),
        ("Control of Breathing", [
            "Automatic control of breathing originates in the medulla oblongata (the dorsal and ventral "
            "respiratory groups), with rhythm modulated by the pontine respiratory centers. The dominant "
            "chemical drive to breathe comes from central chemoreceptors in the medulla, which respond to "
            "changes in cerebrospinal fluid pH driven by arterial CO2 (CO2 diffuses freely across the "
            "blood-brain barrier and is hydrated to carbonic acid). Peripheral chemoreceptors in the carotid "
            "and aortic bodies respond primarily to hypoxemia (low PaO2), becoming an important drive to "
            "breathe only when PaO2 falls substantially, and also respond to pH and PCO2 changes with a "
            "faster response time than central chemoreceptors.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 3. Renal Physiology and Acid-Base Balance
# ---------------------------------------------------------------------------
build_pdf(
    "Renal_Physiology_AcidBase.pdf",
    "Renal Physiology and Acid-Base Balance",
    [
        ("The Nephron: Structure and Overview", [
            "The nephron is the functional unit of the kidney and consists of the renal corpuscle "
            "(glomerulus plus Bowman's capsule) and a renal tubule made up of the proximal convoluted "
            "tubule, the loop of Henle (descending and ascending limbs), the distal convoluted tubule, and "
            "the collecting duct. Blood is filtered at the glomerulus based on size and charge selectivity, "
            "producing an ultrafiltrate that is essentially plasma without large proteins or cells; this "
            "filtrate is then modified by reabsorption and secretion as it passes through the tubule.",
        ]),
        ("Filtration, Reabsorption, and Secretion", [
            "Glomerular filtration rate (GFR) is the volume of filtrate formed per unit time and is "
            "determined by the net filtration pressure across the glomerular capillaries, which balances "
            "glomerular capillary hydrostatic pressure (favoring filtration) against Bowman's capsule "
            "hydrostatic pressure and glomerular capillary oncotic pressure (both opposing filtration).",
            "The proximal convoluted tubule reabsorbs approximately 65% of filtered sodium and water, "
            "along with essentially all filtered glucose and amino acids under normal conditions, largely "
            "via sodium-coupled cotransport. The loop of Henle establishes the medullary concentration "
            "gradient via the countercurrent multiplier: the thick ascending limb is impermeable to water "
            "but actively reabsorbs sodium, potassium, and chloride via the Na-K-2Cl cotransporter (the "
            "target of loop diuretics such as furosemide), diluting the tubular fluid while concentrating "
            "the medullary interstitium. The distal nephron and collecting duct fine-tune sodium, potassium, "
            "and water balance under the influence of aldosterone and antidiuretic hormone (ADH), "
            "respectively.",
        ]),
        ("Acid-Base Physiology", [
            "Arterial blood pH is normally tightly regulated between 7.35 and 7.40. The Henderson-"
            "Hasselbalch equation relates pH to the ratio of bicarbonate to dissolved CO2: pH = 6.1 + "
            "log([HCO3-]/(0.03 x PaCO2)). The lungs regulate the respiratory component (PaCO2) on a "
            "timescale of minutes, while the kidneys regulate the metabolic component (HCO3-) over hours to "
            "days by adjusting bicarbonate reabsorption in the proximal tubule and net acid excretion "
            "(as titratable acid and ammonium) in the distal nephron.",
            "A primary respiratory disturbance is one in which PaCO2 changes first (respiratory acidosis: "
            "elevated PaCO2 from hypoventilation; respiratory alkalosis: decreased PaCO2 from "
            "hyperventilation), while a primary metabolic disturbance is one in which HCO3- changes first "
            "(metabolic acidosis: decreased HCO3-, for example from diabetic ketoacidosis or diarrhea; "
            "metabolic alkalosis: increased HCO3-, for example from prolonged vomiting). In each case, the "
            "unaffected organ system attempts partial compensation: the lungs compensate for metabolic "
            "disturbances within minutes to hours by changing ventilation, while the kidneys compensate for "
            "respiratory disturbances over one to several days by changing bicarbonate handling.",
        ]),
        ("The Anion Gap", [
            "The anion gap (AG = Na+ - [Cl- + HCO3-]), normally about 8-12 mEq/L, is used to classify "
            "metabolic acidosis. A high anion gap metabolic acidosis occurs when an unmeasured acid "
            "accumulates (for example lactic acidosis, diabetic ketoacidosis, or toxic alcohol ingestion), "
            "while a normal anion gap (hyperchloremic) metabolic acidosis occurs when bicarbonate is lost "
            "directly, and chloride rises to maintain electroneutrality (for example severe diarrhea or "
            "renal tubular acidosis).",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 4. Endocrine System and Diabetes Mellitus
# ---------------------------------------------------------------------------
build_pdf(
    "Endocrine_Diabetes.pdf",
    "The Endocrine Pancreas and Diabetes Mellitus",
    [
        ("The Endocrine Pancreas", [
            "The endocrine pancreas consists of the islets of Langerhans, which make up roughly 1-2% of "
            "pancreatic mass but are richly vascularized to allow rapid hormone release into the "
            "bloodstream. Beta cells, the most numerous islet cell type, secrete insulin; alpha cells "
            "secrete glucagon; delta cells secrete somatostatin, which locally inhibits both insulin and "
            "glucagon release.",
            "Insulin is synthesized as preproinsulin, cleaved to proinsulin, and then to insulin plus "
            "C-peptide, which are co-secreted in equimolar amounts -- C-peptide levels are therefore used "
            "clinically to estimate endogenous insulin production, particularly in patients already taking "
            "exogenous insulin.",
        ]),
        ("Insulin and Glucagon Physiology", [
            "Insulin is released from beta cells in response to rising blood glucose: glucose enters the "
            "beta cell via GLUT2 transporters, is metabolized to generate ATP, and the resulting rise in the "
            "ATP/ADP ratio closes ATP-sensitive potassium channels, depolarizing the cell membrane and "
            "opening voltage-gated calcium channels; calcium influx triggers exocytosis of insulin-"
            "containing granules. Insulin promotes glucose uptake into skeletal muscle and adipose tissue "
            "via translocation of GLUT4 transporters to the cell membrane, promotes glycogen synthesis in "
            "the liver and muscle, and inhibits lipolysis and gluconeogenesis.",
            "Glucagon, released from alpha cells in response to low blood glucose (hypoglycemia) and during "
            "fasting, has largely opposing effects: it stimulates hepatic glycogenolysis and "
            "gluconeogenesis, raising blood glucose. Insulin and glucagon therefore act as a reciprocal "
            "regulatory pair maintaining glucose homeostasis.",
        ]),
        ("Type 1 vs Type 2 Diabetes Mellitus", [
            "Type 1 diabetes mellitus (T1DM) results from autoimmune destruction of pancreatic beta cells, "
            "typically mediated by T-cells and associated with autoantibodies such as anti-GAD65 and "
            "anti-islet cell antibodies, leading to absolute insulin deficiency. It most commonly presents "
            "in children and young adults, often with acute symptoms (polyuria, polydipsia, weight loss) "
            "and can present with diabetic ketoacidosis (DKA) as the first sign, since without any insulin "
            "the body cannot suppress lipolysis and ketogenesis.",
            "Type 2 diabetes mellitus (T2DM) is characterized primarily by peripheral insulin resistance -- "
            "reduced responsiveness of muscle, liver, and adipose tissue to insulin's effects -- combined "
            "with a progressive relative deficiency in insulin secretion as beta cells fail to compensate "
            "over time. Because insulin resistance, not absence, is the initial problem, peripheral tissues "
            "take up less glucose despite normal or even elevated circulating insulin levels early in the "
            "disease course. T2DM is strongly associated with obesity, physical inactivity, and genetic "
            "predisposition, and usually develops gradually in adults, though rising childhood obesity has "
            "increased pediatric incidence.",
        ]),
        ("Chronic Complications of Diabetes", [
            "Chronic hyperglycemia damages small blood vessels (microvascular disease) and large vessels "
            "(macrovascular disease). Microvascular complications include diabetic retinopathy (a leading "
            "cause of adult blindness), diabetic nephropathy (progressive kidney damage often first "
            "detected as microalbuminuria), and diabetic peripheral neuropathy (commonly a symmetric "
            "'stocking-glove' sensory loss that increases the risk of unnoticed foot injuries and "
            "ulceration). Macrovascular complications include accelerated atherosclerosis, increasing the "
            "risk of myocardial infarction, stroke, and peripheral arterial disease. Glycated hemoglobin "
            "(HbA1c) reflects average blood glucose over the preceding 2-3 months and is used both to "
            "diagnose diabetes and to monitor long-term glycemic control.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 5. Pharmacology of Antibiotics
# ---------------------------------------------------------------------------
build_pdf(
    "Pharmacology_Antibiotics.pdf",
    "Pharmacology: Antibiotic Classes and Mechanisms",
    [
        ("General Principles", [
            "Antibiotics are classified by their mechanism of action into several broad categories: "
            "inhibitors of cell wall synthesis, inhibitors of protein synthesis, inhibitors of nucleic acid "
            "synthesis, and disruptors of cell membrane integrity. An antibiotic is described as "
            "bactericidal if it kills bacteria directly (e.g. beta-lactams, aminoglycosides, "
            "fluoroquinolones) or bacteriostatic if it inhibits bacterial growth and relies on the host "
            "immune system to clear the infection (e.g. macrolides, tetracyclines, most cases of "
            "sulfonamides). Selective toxicity -- the ability to harm the pathogen while sparing the host "
            "-- is achieved by targeting structures or processes unique to bacteria, such as the cell wall "
            "or bacterial-type ribosomes.",
        ]),
        ("Cell Wall Synthesis Inhibitors", [
            "Beta-lactam antibiotics, including penicillins, cephalosporins, and carbapenems, share a "
            "beta-lactam ring that binds and inhibits penicillin-binding proteins (transpeptidases), "
            "blocking the cross-linking of peptidoglycan strands needed for a structurally sound bacterial "
            "cell wall; this causes osmotic lysis, particularly during active bacterial growth. Bacterial "
            "resistance commonly arises via beta-lactamase enzymes that hydrolyze the beta-lactam ring, "
            "which is why some formulations combine a beta-lactam with a beta-lactamase inhibitor (for "
            "example amoxicillin-clavulanate).",
            "Vancomycin, a glycopeptide, inhibits cell wall synthesis by binding the D-Ala-D-Ala terminus "
            "of peptidoglycan precursors, blocking transglycosylation; because its target is different from "
            "beta-lactams, it retains activity against many beta-lactam-resistant organisms such as "
            "methicillin-resistant Staphylococcus aureus (MRSA), though vancomycin-resistant enterococci "
            "(VRE) have emerged through altered precursor termini (D-Ala-D-Lac).",
        ]),
        ("Protein Synthesis Inhibitors", [
            "Bacterial ribosomes (70S, composed of 30S and 50S subunits) differ structurally from the "
            "human 80S ribosome, allowing selective inhibition. Aminoglycosides (e.g. gentamicin) bind the "
            "30S subunit, causing misreading of mRNA and are bactericidal, but carry a risk of "
            "nephrotoxicity and ototoxicity. Tetracyclines also bind the 30S subunit, blocking aminoacyl-"
            "tRNA binding, and are bacteriostatic. Macrolides (e.g. azithromycin) and clindamycin bind the "
            "50S subunit, blocking translocation, and are generally bacteriostatic. Linezolid binds the 50S "
            "subunit at an early stage, preventing formation of the initiation complex, and is notable for "
            "retaining activity against many multi-drug-resistant Gram-positive organisms.",
        ]),
        ("Nucleic Acid Synthesis Inhibitors and Antimetabolites", [
            "Fluoroquinolones (e.g. ciprofloxacin) inhibit bacterial DNA gyrase (topoisomerase II) and "
            "topoisomerase IV, enzymes required to relieve supercoiling during DNA replication, and are "
            "bactericidal. Rifampin inhibits bacterial DNA-dependent RNA polymerase, blocking transcription, "
            "and is a cornerstone of anti-tuberculosis therapy, notably able to penetrate into "
            "macrophages and caseous granulomas.",
            "Sulfonamides and trimethoprim act as antimetabolites in the bacterial folate synthesis "
            "pathway: sulfonamides competitively inhibit dihydropteroate synthase (which uses para-"
            "aminobenzoic acid, PABA), while trimethoprim inhibits dihydrofolate reductase at a later step. "
            "Because these drugs act sequentially in the same pathway, they are often combined (as "
            "trimethoprim-sulfamethoxazole) for a synergistic bactericidal effect, and selective toxicity "
            "arises because humans obtain folate from the diet rather than synthesizing it de novo.",
        ]),
        ("Mechanisms of Antibiotic Resistance", [
            "Bacteria acquire resistance through several general mechanisms: enzymatic drug inactivation "
            "(e.g. beta-lactamases), target site modification (e.g. altered penicillin-binding proteins in "
            "MRSA, altered ribosomal binding sites), decreased drug accumulation via reduced permeability or "
            "efflux pumps, and bypass pathways that circumvent the inhibited step. Resistance genes spread "
            "both vertically (during bacterial replication) and horizontally between bacteria via "
            "conjugation, transformation, or transduction, which is why resistance can spread rapidly across "
            "and even between bacterial species, particularly under the selective pressure of antibiotic "
            "overuse.",
        ]),
    ],
)

print("\nAll 5 knowledge base PDFs generated in:", OUT_DIR)
