# Activation-steering method fact sheet (source-checked)

Source tags: [P] = full-text markdown in conceptors/docs/lit_review/papers/, [S] = local summary only, [A] = arXiv abstract page only. "not stated" means the consulted source does not say it. Equation/section/table numbers are as numbered in the cited source.

## 1. CAST, conditional activation steering (2409.05907) [S: summaries/2409.05907.md; full text not local, so numbers are the summary's table citations]
- **Operator:** Sec. 3.1: h' = h + f(sim(h, proj_c h)) . alpha . v, with proj_c h = (c c^T / c.c) h and sim = cosine; in practice sim(h, tanh(proj_c h)). f is a step function (1 if sim > theta, else 0). v is the behavior (refusal) vector, added at a band of layers.
- **When it acts:** gate is checked once, on the first forward pass over the prompt, at a chosen condition layer (first half of the network; App. A.2, Fig. 11). If it fires, v is added on every later token. Conditions combine by OR; flipping the comparison gives NOT; AND is not stated.
- **Fit on:** condition vector and behavior vector are first principal components of mean-centred contrastive hidden states (Sec. 3.3). Layer, threshold and comparison direction come from an F1 grid search (App. C.2). Alpha and layers are tuned by hand.
- **Evidence:** Table 3, Qwen1.5-1.8B: always-on refusal steering gives 96.40% refusal on harmless prompts; with the gate, harmless 2.20% and harmful 90.67%. Single-turn, 500 Alpaca plus 450 Sorry-Bench prompts, seven small to mid models.
- **Toy-2D trap:** the gate reads the prompt once, not the moving state, and it thresholds a cosine (norm-blind), so a 2D "enter the region, get pushed" picture overstates it.

## 2. DSAS, dynamically scaled activation steering (2512.03661) [P: arxiv_2512.03661v2.md]
- **Operator:** Eq. 3.1: T_DSAS(t) = (1 - h_l(t)) t + h_l(t) T_l(t; lambda). It wraps any steering map T_l (CAA, ITI, LinEAS) by interpolating between the original and steered embedding.
- **When it acts:** every intervened layer and every token, with a soft gate h_l(t) in [0,1] from a per-layer logistic regressor on the current token embedding (Eq. 3.4, after PCA). Optional skip if layer classifier accuracy is below tau (Sec. 3.2).
- **Fit on:** a source set S (undesired) against a control set C, using mean-pooled embeddings per input (Eq. 3.2) reduced by PCA (5 components, Sec. 4.1). The target set shapes T_l, not the gate. An end-to-end variant trains the gate with LinEAS (Sec. 3.3).
- **Evidence:** Table 1, Qwen2.5-1.5B toxicity (ToxTET%): CAA 13.67 to CAA+DSAS 8.64; LinEAS 9.78 to 3.98; MMLU 59.19 to 59.49 for LinEAS. Single-turn RealToxicityPrompts, three models (Qwen2.5 1.5B and 7B, Gemma 2 2B).
- **Toy-2D trap:** the gate is a hyperplane classifier in a PCA space trained on pooled sentence vectors, then applied token by token, so it is a soft ramp, not a sharp boundary on the single point h.

## 3. GAPS, dimension-level gates (2609.01878) [P: arxiv_2609.01878v1.md]
- **Operator:** Eq. 7: h_t <- h_t + alpha g(h_t) (m_sep ⊙ m_post(h_t)) ⊙ v, g being CAST's or DSAS's token gate. m_sep (Eq. 3) keeps neuron d if |AUROC_d - 1/2| > tau_z sigma_0 (tau_z = 7). m_post (Eq. 6) keeps d if p(src | h_t,d) > 1/2 under per-neuron Gaussians (Eq. 4-5).
- **When it acts:** token gate g decides when; m_post is re-evaluated each step per neuron; update applies at the final token position of the running context, at layer 15 (App. E has layer 18).
- **Fit on:** the same contrastive activations used to build v; per-neuron AUROC and per-concept mean and variance, one offline pass.
- **Evidence:** abstract, Gemma-3 4B toxicity under a fixed capability budget: DSAS+GAPS 0.48% vs DSAS alone 3.52% (table row also lists 0.48). Single-turn RealToxicityPrompts and OneSeC concept removal, Gemma-3 4B and Qwen-3 1.7B.
- **Toy-2D trap:** the mask is per coordinate axis, so the update direction depends on the basis and need not point along v.

## 4. LiSeCo, linear semantic control (2405.15454) [P: arxiv_2405.15454v5.md]
- **Operator:** Sec. 3.2.2 / Theorem 4.1 (Table 1): x~_t = x_t + theta_t. With probe f_t = nu(W_t^T x), if nu(W^T x) > alpha_max then theta = (nu^-1(alpha_max) - W^T x) W / ||W||^2; if below alpha_min, the same with alpha_min; otherwise theta = 0. This is the minimum-norm move onto the allowed region (Eq. 8). Logistic case: Corollary 4.2 (Eq. 10).
- **When it acts:** at each controlled layer and each generated token, only when the probe score is outside [alpha_min, alpha_max] (Algorithm 2).
- **Fit on:** a linear probe per layer on last-token hidden states of a labelled constraint set (30k Jigsaw comments for toxicity; FLORES-Plus for language), with labels from a scoring model (Sec. 5.1).
- **Evidence:** Table 2, English-to-Spanish, in-distribution mean P(Spanish): Llama-8B 0.987, Gemma-2B 0.913, Mistral-7B 0.988. Fig. 4: probe scores land inside the target range. Single-turn continuations.
- **Toy-2D trap:** the guarantee holds for the probe score at that layer only; output-level guarantees need probe validity everywhere (Sec. 6.3, App. C), and the result lands on the boundary, not beyond it.

## 5. MiMiC / Representation Surgery, affine steering (2402.09631) [P: Representation_surgery_Theory_and_practice_of_aff.md]
- **Operator:** Sec. 4.1, Prop. 4.1 (mean matching): s*(H) = H + mu_c' - mu_c for H of concept c. Sec. 4.2, Prop. 4.2, Eq. 10-11 (MiMiC): s*(H) = W* H + b*, W* = Sigma_c^(-1/2) (Sigma_c^(1/2) Sigma_c' Sigma_c^(1/2))^(1/2) Sigma_c^(-1/2), b* = -W* mu_c + mu_c'. Same map as Gaussian optimal transport (Prop. in Sec. 4.2).
- **When it acts:** only on representations judged to be in concept c. In generation (Sec. 5.2) it is applied at each inference step to states closer to mu_c than mu_c'. Layer: not stated in the sections read.
- **Fit on:** concept-conditional means and covariances from labelled sentences (Toxic Comments data; last-token states).
- **Evidence:** Table 2, GPT-2 large, RealToxicityPrompts: expected max toxicity 0.39 to 0.33 (mean matching) and 0.29 (mean+covariance); toxic prob 0.25 to 0.09; fluency 24.66 to 30.7. Single-turn, 20-token completions; below fine-tuned/decoding baselines such as DExperts (0.21).
- **Toy-2D trap:** a pure translation picture is only the mean-matching case; MiMiC also rotates/reshapes by W*.

## 6. SEA, spectral editing of activations (2405.09719) [P: arxiv_2405.09719v3.md]
- **Operator:** Sec. 2.3-2.4: SVD of cross-covariances Omega+ = (1/n) sum h (h+)^T and Omega- (Eq. 2). Keep top-k+ left singular vectors U+ and bottom U-. z~+ = U+ U+^T z, z~- = U- U-^T z (Eq. 3), merged per coordinate with a norm-preserving rescaling (Eq. 4; formula garbled in the markdown, so check the PDF before reproducing it). Non-linear variant Phi-SEA uses a feature map phi (Sec. 2.5).
- **When it acts:** always, on MLP outputs of the last L layers, at every token position.
- **Fit on:** last-token activations of positive, negative and neutral demonstrations (n = 1000-2000); k set by variance fraction K.
- **Evidence:** Table 1, LLaMA-2-Chat-7B TruthfulQA, single-turn: MC1 36.96 (ICL) to 39.41 (SEA, N=2000, K=99.8%, L=21); MC2 54.68 to 57.15.
- **Toy-2D trap:** it projects (shrinks components), it does not add a vector, and the two projections are merged non-additively.

## 7. Contrastive covariance subspaces, CCS (2610.14025) [P: arxiv_2610.14025.md]
- **Operator:** operator is plain additive; the vector is new. Eq. 1/7: h_steer = h + alpha ||h||_2 v_hat, with v_t = P_t delta_t = P_t (mu_t - mu_G) (Eq. 6). P_t = W_t W_t^T, W_t the k leading eigenvectors of A_t = S_t - (r/|N_t|) sum_c S_c (Eq. 5), S_c the class scatter about the grand mean.
- **When it acts:** always, every token, one layer; alpha is relative to each token's norm (App. A.3).
- **Fit on:** mean-pooled prompt activations of several concept classes; the other classes act as nuisance (topic, style). Tuned k and r.
- **Evidence:** abstract: beats all baselines in 12 of 12 model-domain settings; AxBench gains of 48-70% over the next best (Sec. 4.2). Sec. 4.5: Qwen3-8B, 20-turn dialogues, joint concept-and-fluent rate 93% to 80%, whereas CAA falls 36% to 10% and a system prompt 91% to 71%.
- **Toy-2D trap:** the improvement comes from which direction is chosen in high dimensions; 2D cannot show nuisance removal.

## 8. PaSS, persona subspaces via INLP (2610.04676) [P: arxiv_2610.04676v1.md]
- **Operator:** Sec. 3-4.2: INLP gives P_content = P^(K)...P^(1) (Eq. 3), P^(k) = I - w w^T/|w|^2 (Eq. 2); P_persona = I - P_content (Eq. 4). Then h~_t = h_content + beta h_persona (Eq. 7), i.e. h + (beta - 1) h_persona; beta = 1 is identity.
- **When it acts:** on the activations of the tokens in the prompt, at a chosen layer, "prior to generation" (Sec. 4.2); whether it is also applied to newly generated tokens is not stated. Needs persona text already in context ("modulation").
- **Fit on:** mean-pooled activations with vs without the persona instruction on 350 MATH-500 items; probes iterated until accuracy reaches chance.
- **Evidence:** Table 2, Llama3-8B-Instruct layer 20, emoji persona score change: PaSS -54.0 / +71.4 at beta -1 / +1, DiffMeans vector -16.67 / +30. Single-turn, 150 MATH-500 questions; also Qwen2.5-7B-Instruct.
- **Toy-2D trap:** the push scales with the persona component already present, so it is multiplicative, and the subspace has several dimensions.

## 9. Persona Dosing / PersonaDose, FLAS controller (2609.36388) [P: arxiv_2609.36388v1.md]
- **Operator:** Sec. 3.2: dh/dt = v_theta(h(t), t, c), h(0) = h_0, h' = h(T) (Eq. 2); three Euler steps h_{k+1} = h_k + (T/3) v_theta(h_k, kT/3, c) (Eq. 3). v_theta is a learned module (cross-attention to the trait description c, plus an MLP).
- **When it acts:** a decoder hook at zero-based layer 20 on all attended positions, in prompt processing and generation (App. A.1, A.3). Always on; strength is the flow time T, mapped from a requested score by isotonic calibration (Eq. 4).
- **Fit on:** supervised persona responses (score >= 50), 1,229 training rows over seven traits, with a cross-entropy loss and a concept-diversity term. No gate.
- **Evidence:** Table 1, core-trait expression at coherence >= 75: PersonaDose 75.1 / 82.3 / 86.0 vs CAA 41.9 / 64.0 / 68.2 (Llama-3.1-8B / Qwen3-8B / Gemma-3-4B). Single-turn question-answer, 256 new tokens.
- **Toy-2D trap:** it is a trained nonlinear flow conditioned on text, not a fixed vector or projection, and it is not a learned-gain version of CAA.

## 10. Activation steering with a feedback controller, PID steering (2510.04309) [P: arxiv_2510.04309.md]
- **Operator:** Eq. 17-18: u(k) = K_p r(k) + K_i sum_{j<k} r(j) + K_d (r(k) - r(k-1)), applied by rho_steer (ActAdd x + alpha u, or directional ablation) at layer k. r(k) is the difference-in-means at layer k, non-sequential (Eq. 3) or sequential (Eq. 5-6).
- **When it acts:** every layer, every token. The "time" index is layer depth, not token or turn.
- **Fit on:** difference-in-means from a source and target set (80% of AdvBench plus 512 Alpaca for jailbreak). No runtime measurement of h, so it is not a closed loop over tokens.
- **Evidence:** Table 2 (Sec. 5.2): attack success rate Llama-3.1-8B 94.85 PID vs 93.26 DIM; Qwen2.5-14B 92.65 vs 90.38 (as stated in the text; the table's model labels are scrambled in the markdown). Single-turn refusal removal; tinyBenchmarks nearly unchanged.
- **Toy-2D trap:** "feedback" acts across layers on precomputed means; a 2D picture of a trajectory over time would misrepresent it.

## 11. Persona as a dynamical system, CAL (INLG 2026) [P: Persona_as_a_Dynamical_System_Attractor-Based_Cha.md]
- **Operator:** Sec. 4.2: s_t = A z_{t-1} + beta P (h_t - mu) (Eq. 2); z_t = (1 - lambda) s_t + lambda c (Eq. 3); h'_t = h_t + alpha W z_t (Eq. 4). Also a logit bias delta_t(v) = gamma <u_v, p_t>, p_t = W(c - z_t) (Eq. 5). Latent z has 14 named axes; configured rho(A) = 0.85, lambda = 0.25, alpha = 1 (App. B).
- **When it acts:** every generated token, at one mid-depth block (layer 18/36 for Qwen3-8B). z carries over across tokens.
- **Fit on:** read/write maps from 2,277 reframing items and 873 Dreaddit posts (capped at 200 per persona); persona centres c from labelled corpora (Table 2). No gradient training.
- **Evidence:** Table 4 human ratings, post-shock recovery: CAL 4.07, ActAdd 3.42, prompt 2.77. Sec. 8.2: Qwen3-8B recovery time constant 1.49 vs 0.17 without restoring force (8.8x). 100-turn dialogues, shock at turn 50, Qwen3-8B and Mistral-7B.
- **Toy-2D trap:** the restoring pull acts on a separate 14-d latent with memory, then is written back, so h is not pulled toward a point directly.

## 12. Angular steering (2510.26243) [A: arXiv abstract only; no local file]
- **Operator:** abstract: rotates activations within a fixed two-dimensional subspace toward or away from a target behaviour direction, and is presented as generalising addition and directional ablation. Formula, plane construction and angle parameterisation: not stated on the abstract page. A secondary mention (CCS App. A.3) says it rotates in a fixed 2D plane and sweeps over a rotation angle.
- **When it acts:** the base method: not stated. Adaptive Angular Steering rotates only activations aligned with the target feature; gate computation not stated.
- **Fit on:** not stated.
- **Evidence:** abstract gives no numbers. Settings named: refusal and emotion steering, multiple model families (NeurIPS 2025 Spotlight).
- **Toy-2D trap:** norm preservation, and how components outside the plane are handled, are not stated here; read the PDF.

## 13. BILLY, merging persona vectors (2510.10157) [A: arXiv abstract only; no local file]
- **Operator:** abstract: extracts several distinct persona vectors, blends them in activation space, and steers generation at inference with the merged vector. Merge rule, weights, layer and strength: not stated.
- **When it acts:** at inference; whether always or gated: not stated.
- **Fit on:** training-free; the data used to extract the persona vectors: not stated.
- **Evidence:** abstract reports no numbers; claims better than single-model prompting and multi-LLM approaches on creativity benchmarks with less inference time.
- **Toy-2D trap:** the merge is not specified, so do not draw it as a plain vector sum without checking the PDF.

## 14. INLP, "Null It Out" (Ravfogel et al., 2020) [A, plus PaSS Sec. 3 restating it]
- **ID and title (confirmed from the abstract page):** arXiv 2004.07667, "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection", ACL 2020.
- **Operator:** abstract: repeatedly train a linear classifier for the property, then project representations onto its null space, until classifiers cannot linearly separate the data. Formula not on the abstract page; PaSS Eq. 2-3 restate it as P^(k) = I - w w^T/|w|^2, composed as P^(K)...P^(1).
- **When it acts:** a fixed post-hoc projection on every representation; not a gate.
- **Fit on:** labelled representations carrying the protected attribute (the abstract does not give the data sizes).
- **Evidence:** abstract gives no numbers; reports reduced bias in word embeddings and improved fairness in a multi-class classification setting.
- **Toy-2D trap:** it deletes directions linearly, it does not move a point along an axis, and a nonlinear classifier can still recover the property (affine-guardedness remark in Representation Surgery, Sec. 4.2).
